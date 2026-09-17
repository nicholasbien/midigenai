"""The reward-model recipe for a new checkpoint, as three commands.

    python -m midigenai.probe_layers cache   --checkpoint CK --tokenizer TOK \
        --labels LABELS --out CACHE.npz                      # ~5 min on MPS
    python -m midigenai.probe_layers sweep   --cache CACHE.npz --labels LABELS
    python -m midigenai.probe_layers confirm --cache CACHE.npz --labels LABELS \
        --layers block1 --l2 10 --checkpoint CK --out SPEC.json

`cache` runs every decided pair through the model once and keeps the
mean-pooled continuation activations of EVERY block plus the final norm and
the mean log-prob, so all later fits are CPU-only.

`sweep` compares layers with grouped 5-fold CV (whole prompts move
together). It is a ranking tool: the fast numbers sit about a point off the
honest ones in either direction, so it picks and never reports.

`confirm` is the honest number: cold leave-one-prompt-out for one layer
combination, and it writes a ProbeReward spec grpo can load — with the
checkpoint hash, byte size, vocab and layers, so the guards can refuse a
mismatch.

Why the recipe exists at all: the probe reads best a few blocks up from the
embeddings, not at the final norm (113M: norm 0.765 -> block1 0.789; 202M:
norm 0.719 -> block3 0.765), the probe is a property of the checkpoint it
was fitted on, and cross-validation must be cold — a warm start leaked six
points once. Never centre the winner-minus-loser differences: with every
row a win, the mean difference IS the signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from midigenai.reward_align import fit_bt


def _labels(path: Path) -> dict[str, str]:
    return {r["pair_id"]: r["preferred"] for r in map(json.loads, open(path))
            if r.get("preferred") in ("a", "b")}


# ---------------------------------------------------------------- cache

def cache(a) -> None:
    import torch
    import torch.nn.functional as F
    from midigenai.grpo import _device, load_policy
    from midigenai.tokenizer import load_tokenizer

    dev = _device(a.device)
    model, cfg = load_policy(a.checkpoint, dev)
    model.eval()
    load_tokenizer(a.tokenizer)
    pairs_dir = a.labels.parent / "pairs"
    ids = sorted(_labels(a.labels))
    print(f"[cache] {len(ids)} decided pairs, {len(model.blocks)} blocks, d_model {cfg.d_model}", flush=True)

    grabbed: dict = {}
    hooks = [blk.register_forward_hook(
        lambda m, i, o, k=k: grabbed.__setitem__(k, o[0] if isinstance(o, tuple) else o))
        for k, blk in enumerate(model.blocks)]
    hooks.append(model.norm.register_forward_hook(lambda m, i, o: grabbed.__setitem__("norm", o)))

    def feats(prompt_ids, cont_ids):
        seq = torch.tensor([list(prompt_ids) + list(cont_ids)], dtype=torch.long, device=dev)
        if seq.shape[1] > cfg.max_seq_len:
            seq = seq[:, -cfg.max_seq_len:]
        n_p = len(prompt_ids)
        with torch.no_grad():
            logits, _ = model(seq)
        sl = slice(n_p - 1, -1) if seq.shape[1] > n_p else slice(-1, None)
        per = [grabbed[k][0][sl].float().mean(0).cpu().numpy() for k in range(len(model.blocks))]
        per.append(grabbed["norm"][0][sl].float().mean(0).cpu().numpy())
        logp = F.log_softmax(logits[0, :-1].float(), dim=-1).gather(
            -1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)[n_p - 1:]
        return np.stack(per), float(logp.mean())

    A, B, LPA, LPB, G, keep = [], [], [], [], [], []
    t0 = time.time()
    for i, pid in enumerate(ids, 1):
        m = json.loads((pairs_dir / f"{pid}.json").read_text())
        try:
            fa, la = feats(m["prompt_ids"], m["cont_a_ids"])
            fb, lb = feats(m["prompt_ids"], m["cont_b_ids"])
        except Exception:
            continue
        if not (np.isfinite(fa).all() and np.isfinite(fb).all()):
            continue
        A.append(fa); B.append(fb); LPA.append(la); LPB.append(lb); G.append(m["prompt_file"]); keep.append(pid)
        if i % 250 == 0:
            el = time.time() - t0
            print(f"[cache] {i}/{len(ids)}  {el:.0f}s  eta {(len(ids) - i) * el / i / 60:.0f} min", flush=True)
    for h in hooks:
        h.remove()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, pair_id=np.array(keep), group=np.array(G),
                        a=np.stack(A).astype(np.float32), b=np.stack(B).astype(np.float32),
                        logp_a=np.array(LPA), logp_b=np.array(LPB),
                        layer_names=np.array([f"block{k}" for k in range(len(model.blocks))] + ["norm"]),
                        checkpoint=str(a.checkpoint.resolve()), vocab_size=int(cfg.vocab_size),
                        d_model=int(cfg.d_model))
    print(f"[cache] wrote {a.out}: {len(keep)} pairs x {len(model.blocks) + 1} layers x {cfg.d_model}  "
          f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- shared

def _load(cache_path: Path, labels_path: Path):
    z = np.load(cache_path)
    pref = _labels(labels_path)
    keep = np.array([p in pref for p in z["pair_id"]])
    ids, G = z["pair_id"][keep], z["group"][keep]
    sign = np.array([1.0 if pref[p] == "a" else -1.0 for p in ids])[:, None]
    names = list(z["layer_names"])
    diffs = {n: (z["a"][keep][:, k, :] - z["b"][keep][:, k, :]) for k, n in enumerate(names)}
    LP = (z["logp_a"][keep] - z["logp_b"][keep])[:, None]
    return z, ids, G, sign, names, diffs, LP


def _X(diffs, LP, sign, layers):
    return np.hstack([diffs[l] for l in layers] + [LP]) * sign     # winner-minus-loser


# ---------------------------------------------------------------- sweep

def sweep(a) -> None:
    z, ids, G, sign, names, diffs, LP = _load(a.cache, a.labels)
    l2s = [float(x) for x in a.l2.split(",")]
    groups = sorted(set(G)); rng = np.random.default_rng(0); rng.shuffle(groups)
    fold_of = {g: i % a.folds for i, g in enumerate(groups)}
    fold = np.array([fold_of[g] for g in G])
    print(f"[sweep] {len(ids)} pairs, {len(groups)} prompt groups, {len(names)} layers, "
          f"grouped {a.folds}-fold, l2 in {l2s}", flush=True)

    def cv(X):
        accs = []
        for l2 in l2s:
            c = 0
            for f in range(a.folds):
                te = fold == f
                sd = X[~te].std(0); sd[sd == 0] = 1            # scale only, never centre
                c += int(((X[te] / sd) @ fit_bt(X[~te] / sd, l2=l2) > 0).sum())
            accs.append(c / len(X))
        return accs

    rows = []
    for n in names:
        accs = cv(_X(diffs, LP, sign, [n]))
        rows.append((n, accs))
        print(f"  {n:8s} " + "  ".join(f"l2={l2:g}: {x:.3f}" for l2, x in zip(l2s, accs)), flush=True)
    best = max(rows, key=lambda r: max(r[1]))
    kb = names.index(best[0])
    for combo in ([kb - 2, kb], [kb, kb + 2], [kb, len(names) - 1]):
        combo = [c for c in dict.fromkeys(combo) if 0 <= c < len(names)]
        if len(combo) < 2:
            continue
        accs = cv(_X(diffs, LP, sign, [names[c] for c in combo]))
        print(f"  concat {'+'.join(names[c] for c in combo):20s} "
              + "  ".join(f"l2={l2:g}: {x:.3f}" for l2, x in zip(l2s, accs)), flush=True)
    print(f"[sweep] best single layer: {best[0]} at {max(best[1]):.3f} — confirm it cold before quoting")


# ---------------------------------------------------------------- confirm

def confirm(a) -> None:
    z, ids, G, sign, names, diffs, LP = _load(a.cache, a.labels)
    layers = a.layers.split(",")
    X = _X(diffs, LP, sign, layers)
    sd = X.std(0); sd[sd == 0] = 1; X = X / sd
    groups = sorted(set(G)); garr = np.array(G)
    n, d = X.shape
    if d >= n:
        print(f"[confirm] WARNING: {d} features >= {n} pairs; leave-one-out is unreliable here", flush=True)
    print(f"[confirm] {n} pairs, {len(groups)} prompt groups, layers {layers} -> {d} features, l2={a.l2:g}", flush=True)
    t0 = time.time()

    def fold(g):
        te = garr == g
        return int(((X[te] @ fit_bt(X[~te], l2=a.l2)) > 0).sum())

    with ThreadPoolExecutor(8) as ex:
        correct = sum(ex.map(fold, groups))
    acc = correct / n
    w = fit_bt(X, l2=a.l2); train = float(((X @ w) > 0).mean())
    print(f"[confirm] layers={'+'.join(layers)} l2={a.l2:g}: held-out {acc:.3f}  train {train:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    if a.out:
        ck = Path(str(z["checkpoint"])) if "checkpoint" in z.files else a.checkpoint
        if ck is None or not Path(ck).exists():
            raise SystemExit("--checkpoint is required to write a spec (the cache does not name one)")
        h = hashlib.sha256()
        with open(ck, "rb") as fh:
            h.update(fh.read(8 << 20))
        spec = {"kind": "probe", "checkpoint": str(Path(ck).resolve()), "layers": layers, "l2": a.l2,
                "weights": w.tolist(), "diff_std": sd.tolist(),
                "heldout_accuracy": acc, "train_accuracy": train,
                "n_pairs": int(n), "n_prompt_groups": len(groups),
                "checkpoint_sha256_8mb": h.hexdigest(), "checkpoint_bytes": Path(ck).stat().st_size,
                "vocab_size": int(z["vocab_size"]) if "vocab_size" in z.files else None,
                "d_model": int(z["d_model"]) if "d_model" in z.files else None,
                "labels": str(a.labels), "cache": str(a.cache)}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(spec))
        print(f"[confirm] wrote {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--tokenizer", type=Path, required=True)
    c.add_argument("--labels", type=Path, required=True, help="labels.jsonl with pairs/ beside it")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--device", default="")
    s = sub.add_parser("sweep")
    s.add_argument("--cache", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--l2", default="1,10")
    s.add_argument("--folds", type=int, default=5)
    f = sub.add_parser("confirm")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--labels", type=Path, required=True)
    f.add_argument("--layers", required=True, help="e.g. block1 or block1,block3")
    f.add_argument("--l2", type=float, default=10.0)
    f.add_argument("--checkpoint", type=Path, default=None)
    f.add_argument("--out", type=Path, default=None, help="write a ProbeReward spec here")
    a = p.parse_args()
    {"cache": cache, "sweep": sweep, "confirm": confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
