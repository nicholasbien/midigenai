"""
Check whether a programmatic reward built from eval_v2 metrics agrees with
human preferences — the gate before using it as RL (GRPO) signal.

Fits a Bradley–Terry model over metric *differences*: P(A beats B) =
sigmoid(w · (f(A) - f(B))). Reports per-metric agreement with human votes,
leave-one-out accuracy of the fitted reward, and writes a reward spec JSON
(feature names + normalization + weights) that a GRPO loop can load.

Inputs (either or both):
    --labels evals/labeling/labels.jsonl    from v2.label_app (pairs/ alongside)
    --historical evals/preferences/preferences.csv   consolidated v1-era data

Run:
    python -m v2.reward_align --labels evals/labeling/labels.jsonl \\
        --historical evals/preferences/preferences.csv \\
        --out evals/reward_spec.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from v2.eval_v2 import compute_metrics

FEATURES = [
    "pitch_class_entropy", "scale_consistency", "polyphony_rate",
    "note_density_hz", "pitch_range", "repetition_rate", "ioi_entropy",
]


def load_label_app_pairs(labels_path: Path) -> list[tuple[Path, Path]]:
    """Return (winner_midi, loser_midi) tuples from label_app votes."""
    pairs_dir = labels_path.parent / "pairs"
    out = []
    with labels_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("preferred") not in ("a", "b"):
                continue  # tie / bad / skip carry no pairwise signal here
            win = rec["preferred"]
            lose = "b" if win == "a" else "a"
            pid = rec["pair_id"]
            w, l = pairs_dir / f"{pid}_{win}.mid", pairs_dir / f"{pid}_{lose}.mid"
            if w.exists() and l.exists():
                out.append((w, l))
    return out


def load_historical_pairs(csv_path: Path) -> list[tuple[Path, Path]]:
    base = csv_path.parent
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            w, l = base / row["file_preferred"], base / row["file_rejected"]
            if w.exists() and l.exists():
                out.append((w, l))
    return out


def feature_vector(midi_path: Path) -> np.ndarray | None:
    try:
        m = compute_metrics(midi_path)
    except Exception:
        return None
    if m["n_notes"] == 0:
        return None
    return np.array([float(m[k]) for k in FEATURES])


def build_diffs(pairs: list[tuple[Path, Path]]) -> np.ndarray:
    """Rows of f(winner) - f(loser), caching per-file metrics."""
    cache: dict[Path, np.ndarray | None] = {}

    def feats(p: Path):
        if p not in cache:
            cache[p] = feature_vector(p)
        return cache[p]

    diffs = []
    for w, l in pairs:
        fw, fl = feats(w), feats(l)
        if fw is not None and fl is not None:
            diffs.append(fw - fl)
    return np.array(diffs)


def fit_bt(diffs: np.ndarray, l2: float = 1.0, iters: int = 2000,
           lr: float = 0.1) -> np.ndarray:
    """Bradley–Terry MLE by gradient descent; every row is a win (y=1)."""
    n, d = diffs.shape
    w = np.zeros(d)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-diffs @ w))
        grad = diffs.T @ (1.0 - p) / n - l2 * w / n
        w += lr * grad
    return w


def loo_accuracy(diffs: np.ndarray, l2: float) -> float:
    n = len(diffs)
    correct = 0
    for i in range(n):
        w = fit_bt(np.delete(diffs, i, axis=0), l2=l2)
        correct += (diffs[i] @ w) > 0
    return correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, default=None,
                   help="labels.jsonl from v2.label_app")
    p.add_argument("--historical", type=Path, default=None,
                   help="consolidated preferences.csv")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("evals/reward_spec.json"))
    args = p.parse_args()

    pairs = []
    if args.labels and args.labels.exists():
        got = load_label_app_pairs(args.labels)
        print(f"[align] {len(got)} pairs from {args.labels}")
        pairs += got
    if args.historical and args.historical.exists():
        got = load_historical_pairs(args.historical)
        print(f"[align] {len(got)} pairs from {args.historical}")
        pairs += got
    if not pairs:
        raise SystemExit("no preference pairs found")

    diffs = build_diffs(pairs)
    print(f"[align] {len(diffs)} usable pairs (both sides parse, non-empty)")
    if len(diffs) < 10:
        print("[align] WARNING: very few pairs — treat every number below "
              "as anecdote, not signal")

    # normalize by the std of diffs so weights are comparable across features
    std = diffs.std(axis=0)
    std[std == 0] = 1.0
    norm_diffs = diffs / std

    print(f"\nper-metric agreement with human votes "
          f"(0.5 = chance; N excludes zero-diff pairs):")
    for i, name in enumerate(FEATURES):
        nz = diffs[:, i] != 0
        n = int(nz.sum())
        agree = float((diffs[nz, i] > 0).mean()) if n else float("nan")
        print(f"  {name:22s}  {agree:5.2f}  (N={n})")

    w = fit_bt(norm_diffs, l2=args.l2)
    train_acc = float(((norm_diffs @ w) > 0).mean())
    loo = loo_accuracy(norm_diffs, l2=args.l2)
    print(f"\nfitted Bradley–Terry reward:")
    for name, wi in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {name:22s}  {wi:+.3f}")
    print(f"\ntrain accuracy {train_acc:.2f}   leave-one-out accuracy {loo:.2f}")
    verdict = ("USABLE as RL reward" if loo >= 0.65 else
               "NOT yet reliable — collect more labels before GRPO")
    print(f"verdict: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "features": FEATURES,
        "diff_std": std.tolist(),
        "weights": w.tolist(),
        "n_pairs": len(diffs),
        "loo_accuracy": loo,
    }, indent=2))
    print(f"[align] wrote {args.out}")


if __name__ == "__main__":
    main()
