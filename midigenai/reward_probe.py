"""
Linear probe reward: preferences fitted on the music model's own hidden
states instead of ten hand-written metrics.

The metric reward can only see summary statistics of a continuation — how
dense it is, how repetitive, how wide. It cannot see whether the continuation
actually *fits its prompt*, because nothing it measures refers to the prompt.
The model's last-layer activations do encode that, so a linear probe on them
sees far more while staying small enough to fit on a few hundred pairs:

    features  = [ mean-pooled last hidden state over the continuation (d_model),
                  mean log p(continuation) under the model (1) ]
    P(A beats B) = sigmoid(w · (f(A) - f(B)))        # same Bradley-Terry loss

d_model+1 parameters (769 for v3) with strong L2, chosen by
leave-one-prompt-out CV — the same protocol the metric reward is judged by,
so the two numbers are comparable.

The features are a property of the checkpoint that produced them, so the spec
records it and `ProbeReward` refuses to load against a different one.

    python -m midigenai.reward_probe fit \\
        --labels evals/labeling_v3_same/labels.jsonl \\
        --checkpoint <base.pt> --tokenizer <tokenizer.json> \\
        --out evals/reward/probe_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from midigenai.reward_align import fit_bt


def _hidden_and_logprob(model, seq: torch.Tensor, n_prompt: int):
    """(mean last-layer hidden state over the continuation, mean log-prob)."""
    grabbed = {}
    handle = model.norm.register_forward_hook(
        lambda _m, _i, out: grabbed.__setitem__("h", out))
    try:
        with torch.no_grad():
            logits, _ = model(seq)
    finally:
        handle.remove()
    hidden = grabbed["h"][0]                       # (T, d_model)
    cont = hidden[n_prompt - 1:-1] if seq.shape[1] > n_prompt else hidden[-1:]
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    picked = logp.gather(-1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)[n_prompt - 1:]
    return cont.float().mean(0).cpu().numpy(), float(picked.mean().cpu())


def feature_vector(model, prompt_ids, cont_ids, device) -> np.ndarray | None:
    if len(cont_ids) < 8:
        return None
    seq = torch.tensor([list(prompt_ids) + list(cont_ids)], dtype=torch.long,
                       device=device)
    if seq.shape[1] > model.cfg.max_seq_len:
        seq = seq[:, -model.cfg.max_seq_len:]
    h, lp = _hidden_and_logprob(model, seq, len(prompt_ids))
    v = np.concatenate([h, [lp]])
    return v if np.isfinite(v).all() else None


def load_pairs(labels_path: Path):
    """(prompt_ids, winner_ids, loser_ids, prompt-group) per decided vote."""
    pairs_dir = labels_path.parent / "pairs"
    out = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("choice") not in ("left", "right"):
            continue
        meta_p = pairs_dir / f"{r['pair_id']}.json"
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text())
        win = r.get("preferred") or (r["left_is"] if r["choice"] == "left" else r["right_is"])
        lose = "b" if win == "a" else "a"
        if f"cont_{win}_ids" not in m:
            continue
        out.append((m.get(f"prompt_ids_{win}") or m["prompt_ids"],
                    m[f"cont_{win}_ids"],
                    m.get(f"prompt_ids_{lose}") or m["prompt_ids"],
                    m[f"cont_{lose}_ids"],
                    Path(m["prompt_file"]).name))
    return out


def logo_accuracy(diffs: np.ndarray, groups: list[str], l2: float) -> float:
    garr = np.array(groups)
    correct = 0
    for g in sorted(set(groups)):
        test = garr == g
        w = fit_bt(diffs[~test], l2=l2)
        correct += int(((diffs[test] @ w) > 0).sum())
    return correct / len(diffs)


def fit(args) -> None:
    from midigenai.grpo import _device, load_policy
    from midigenai.tokenizer import load_tokenizer

    device = _device(args.device)
    model, _ = load_policy(args.checkpoint, device)
    model.eval()
    load_tokenizer(args.tokenizer)          # validates the tokenizer exists

    pairs = load_pairs(args.labels)
    print(f"[probe] {len(pairs)} decided pairs")
    diffs, groups = [], []
    for pw, cw, pl, cl, g in pairs:
        fw = feature_vector(model, pw, cw, device)
        fl = feature_vector(model, pl, cl, device)
        if fw is None or fl is None:
            continue
        diffs.append(fw - fl)
        groups.append(g)
    diffs = np.array(diffs)
    print(f"[probe] {len(diffs)} usable pairs, {len(set(groups))} prompt groups, "
          f"{diffs.shape[1]} features")
    if len(diffs) < 20:
        raise SystemExit("not enough pairs to fit")

    std = diffs.std(axis=0)
    std[std == 0] = 1.0
    diffs = diffs / std

    best = None
    for l2 in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        acc = logo_accuracy(diffs, groups, l2)
        print(f"[probe]   l2={l2:<9g} leave-one-prompt-out accuracy {acc:.3f}")
        if best is None or acc > best[1]:
            best = (l2, acc)
    l2, acc = best
    w = fit_bt(diffs, l2=l2)
    train_acc = float(((diffs @ w) > 0).mean())
    print(f"[probe] best l2={l2:g}: held-out {acc:.3f}, train {train_acc:.3f}")

    spec = {"kind": "probe", "checkpoint": str(Path(args.checkpoint).resolve()),
            "d_model": int(diffs.shape[1] - 1), "l2": l2,
            "weights": w.tolist(), "diff_std": std.tolist(),
            "heldout_accuracy": acc, "train_accuracy": train_acc,
            "n_pairs": int(len(diffs)), "n_prompt_groups": len(set(groups))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(spec))
    print(f"[probe] wrote {args.out}")


class ProbeReward:
    """Same interface as reward.Reward, but needs the model that produced the
    features (and therefore the prompt as well as the continuation)."""

    def __init__(self, spec: dict, model, device):
        self.weights = np.asarray(spec["weights"], dtype=np.float64)
        self.diff_std = np.asarray(spec["diff_std"], dtype=np.float64)
        self.heldout_accuracy = spec.get("heldout_accuracy")
        self.self_consistency = spec.get("self_consistency")
        self.features = [f"h{i}" for i in range(spec["d_model"])] + ["mean_logprob"]
        self.model = model
        self.device = device

    @classmethod
    def load(cls, path, model, device) -> "ProbeReward":
        return cls(json.loads(Path(path).read_text()), model, device)

    def score(self, tokenizer, cont_ids, prompt_ids=None) -> float | None:
        if prompt_ids is None:
            raise ValueError("ProbeReward needs prompt_ids (features are prompt-relative)")
        v = feature_vector(self.model, prompt_ids, cont_ids, self.device)
        if v is None:
            return None
        return float(self.weights @ (v / self.diff_std))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--checkpoint", type=Path, required=True)
    f.add_argument("--tokenizer", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--device", default="")
    a = p.parse_args()
    fit(a)


if __name__ == "__main__":
    main()
