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


DEFAULT_LAYERS = ("norm",)   # what every probe before layer selection used


def _layer_module(model, name: str):
    """"norm" is the final RMSNorm; "blockN" is transformer block N's output."""
    if name == "norm":
        return model.norm
    if name.startswith("block"):
        return model.blocks[int(name[len("block"):])]
    raise ValueError(f"unknown layer {name!r}: expected 'norm' or 'block<N>'")


def fit_to_context(prompt_ids, cont_ids, max_seq_len: int, device):
    """(seq, n_prompt) with the prompt boundary corrected for truncation.

    A sequence longer than the context is cut from the FRONT, so the
    continuation always survives whole and it is the prompt that loses
    tokens. The boundary has to move with the cut. Keeping the original
    length instead pools activations from the wrong positions -- and once
    the cut is deeper than the prompt, `slice(n_prompt - 1, -1)` selects a
    region that is partly or wholly continuation, or nothing at all, which
    reaches the caller as a NaN and silently drops the pair.

    Reachable wherever prompt + continuation passes the context: an
    accompaniment prompt plus up to 64*bars+64 new tokens gets there.
    Clamped to 1 so a prompt cut away entirely still leaves one position
    of context rather than an empty slice.
    """
    seq = torch.tensor([list(prompt_ids) + list(cont_ids)], dtype=torch.long,
                       device=device)
    n_prompt = len(prompt_ids)
    if seq.shape[1] > max_seq_len:
        n_prompt = max(1, n_prompt - (seq.shape[1] - max_seq_len))
        seq = seq[:, -max_seq_len:]
    return seq, n_prompt


def _hidden_and_logprob(model, seq: torch.Tensor, n_prompt: int,
                        layers=DEFAULT_LAYERS):
    """(concat of mean continuation activations from each layer, mean log-prob).

    Which layer to read matters: sweeping every block on both the 113M and
    the 202M, the preference is most linearly readable a few blocks up, and
    the final norm output — the only thing this ever read before — sits
    3-4 points below the best block. Layers are concatenated in the order
    given, then the log-prob, so a spec fitted from a feature cache scores
    identically here.
    """
    grabbed = {}
    handles = []
    for name in layers:
        mod = _layer_module(model, name)
        handles.append(mod.register_forward_hook(
            lambda _m, _i, out, k=name: grabbed.__setitem__(
                k, out[0] if isinstance(out, tuple) else out)))
    try:
        with torch.no_grad():
            logits, _ = model(seq)
    finally:
        for h in handles:
            h.remove()
    sl = slice(n_prompt - 1, -1) if seq.shape[1] > n_prompt else slice(-1, None)
    pooled = [grabbed[name][0][sl].float().mean(0).cpu().numpy() for name in layers]
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    picked = logp.gather(-1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)[n_prompt - 1:]
    return np.concatenate(pooled), float(picked.mean().cpu())


def feature_vector(model, prompt_ids, cont_ids, device,
                   layers=DEFAULT_LAYERS) -> np.ndarray | None:
    if len(cont_ids) < 8:
        return None
    seq, n_prompt = fit_to_context(prompt_ids, cont_ids,
                                   model.cfg.max_seq_len, device)
    h, lp = _hidden_and_logprob(model, seq, n_prompt, layers)
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
    if not out:
        # Every row was filtered. The usual causes are a labels file whose
        # `choice` is not left/right (label_app's schema) and pair metas
        # without cont_<side>_ids. Loading zero pairs and fitting on an empty
        # array fails far from the cause, so say it here.
        raise SystemExit(
            f"no usable pairs from {labels_path}: needs rows with "
            "choice in (left, right) and pair metas carrying prompt_ids and "
            "cont_<side>_ids")
    return out


def logo_accuracy(diffs: np.ndarray, groups: list[str], l2: float) -> float:
    """Leave-one-prompt-out accuracy, one cold fit per prompt group.

    Slow — tens of minutes for 769 features and ~400 groups, scaling with
    d_model — but the cheap alternative was wrong (see below).
    """
    import time
    garr = np.array(groups)
    uniq = sorted(set(groups))
    n, d = diffs.shape
    if d >= n:
        print(f"[probe]   WARNING: {d} features >= {n} pairs; leave-one-out "
              f"accuracy is unreliable in this regime — get more labels",
              flush=True)
    # Every fold starts cold. A warm start from the all-data fit was tried
    # for speed and leaks: with more features than pairs the all-data fit
    # memorises every row, and 400 iterations from there leave the held-out
    # group's rows still fitted — it reported 0.993 held-out on 276 pairs.
    # The synthetic check that "verified" it had n >> d, where there is
    # nothing to memorise. Correctness over the 4.6x.
    # Folds are independent, so they run in parallel — the honest way to get
    # the speed back. numpy's matmul releases the GIL, so threads suffice and
    # the (n x d) matrix is shared rather than copied per worker.
    import os
    from concurrent.futures import ThreadPoolExecutor

    def fold(g):
        test = garr == g
        w = fit_bt(diffs[~test], l2=l2)
        return int(((diffs[test] @ w) > 0).sum())

    correct = 0
    t0 = time.time()
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(workers) as ex:
        for i, c in enumerate(ex.map(fold, uniq), 1):
            correct += c
            if i % 50 == 0 or i == len(uniq):
                el = time.time() - t0
                print(f"[probe]   cross-validation {i}/{len(uniq)} groups  "
                      f"{el:.0f}s elapsed, ~{(len(uniq) - i) * el / i:.0f}s left "
                      f"({workers} workers)", flush=True)
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
    import time
    t0 = time.time()
    layers = tuple(args.layers.split(",")) if args.layers else DEFAULT_LAYERS
    print(f"[probe] reading layers {layers}", flush=True)
    for i, (pw, cw, pl, cl, g) in enumerate(pairs, 1):
        fw = feature_vector(model, pw, cw, device, layers)
        fl = feature_vector(model, pl, cl, device, layers)
        if i % 200 == 0:
            el = time.time() - t0
            print(f"[probe] features {i}/{len(pairs)} pairs  {el:.0f}s elapsed, "
                  f"~{(len(pairs) - i) * el / i:.0f}s left", flush=True)
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
    # Every fit so far (113M, 202M) was best at l2=1 and fell monotonically
    # from there; 10000 collapses to w=0. The grid stays narrow because each
    # value costs one cold fit per prompt group.
    for l2 in (1.0, 3.0, 10.0):
        acc = logo_accuracy(diffs, groups, l2)
        print(f"[probe]   l2={l2:<9g} leave-one-prompt-out accuracy {acc:.3f}")
        if best is None or acc > best[1]:
            best = (l2, acc)
    l2, acc = best
    w = fit_bt(diffs, l2=l2)
    train_acc = float(((diffs @ w) > 0).mean())
    print(f"[probe] best l2={l2:g}: held-out {acc:.3f}, train {train_acc:.3f}")

    import hashlib
    h = hashlib.sha256()
    with open(args.checkpoint, "rb") as fh:            # 8 MB is plenty to identify
        h.update(fh.read(8 << 20))
    spec = {"kind": "probe", "checkpoint": str(Path(args.checkpoint).resolve()),
            "layers": list(layers),
            # corpus_v5 moves to a 598-token vocab and every musical id shifts;
            # a probe fitted on 590-vocab activations must never score a
            # 598-vocab model, and the hash guard alone would not say why
            "vocab_size": int(model.cfg.vocab_size),
            # the path is where it was fitted; the hash is what it was fitted
            # ON, and only the hash survives being copied to a GPU box
            "checkpoint_sha256_8mb": h.hexdigest(),
            "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
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
        self.layers = tuple(spec.get("layers", DEFAULT_LAYERS))
        self.features = [f"h{i}" for i in range(len(self.weights) - 1)] + ["mean_logprob"]
        if len(self.weights) != len(self.diff_std):
            raise ValueError("probe spec: weights and diff_std disagree in length")
        self.model = model
        self.device = device

    @classmethod
    def load(cls, path, model, device) -> "ProbeReward":
        return cls(json.loads(Path(path).read_text()), model, device)

    def score(self, tokenizer, cont_ids, prompt_ids=None) -> float | None:
        if prompt_ids is None:
            raise ValueError("ProbeReward needs prompt_ids (features are prompt-relative)")
        v = feature_vector(self.model, prompt_ids, cont_ids, self.device, self.layers)
        if v is None:
            return None
        if len(v) != len(self.weights):
            raise ValueError(f"probe spec expects {len(self.weights)} features for "
                             f"layers {self.layers}, model produced {len(v)}")
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
    f.add_argument("--layers", default=None,
                   help="comma-separated, e.g. block1 or block1,block3; default norm "
                        "(the final RMSNorm output). Sweep first; the best block is "
                        "usually a few up from the bottom, not the top.")
    a = p.parse_args()
    fit(a)


if __name__ == "__main__":
    main()
