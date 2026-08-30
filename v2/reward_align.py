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

# Degradation drift: second half of the continuation minus first half, computed
# from the generated token ids (label_app pairs only — needs the pair's meta
# JSON). Captures "starts good, wanders off": rising repetition, collapsing or
# exploding density, drifting tonality.
DRIFT_FEATURES = ["repetition_drift", "density_drift", "pce_drift"]


def drift_vector(cont_ids: list[int], tokenizer) -> np.ndarray | None:
    from v2.eval_v2 import (note_density_hz, pitch_class_entropy,
                            repetition_rate)
    if len(cont_ids) < 32:
        return None
    half = len(cont_ids) // 2
    try:
        s1 = tokenizer.decode(cont_ids[:half])
        s2 = tokenizer.decode(cont_ids[half:])
        return np.array([
            repetition_rate(s2) - repetition_rate(s1),
            note_density_hz(s2) - note_density_hz(s1),
            pitch_class_entropy(s2) - pitch_class_entropy(s1),
        ])
    except Exception:
        return None


def load_label_app_pairs(labels_path: Path):
    """(winner, loser, group) tuples + per-pair vote lists from label_app.

    `group` is the seed prompt, so cross-validation can split by prompt and
    never test on a prompt it trained on. Repeat votes on the same pair_id
    feed the self-consistency estimate; the pair's majority vote is used for
    fitting (split votes drop the pair).
    """
    pairs_dir = labels_path.parent / "pairs"
    votes: dict[str, list[str]] = {}
    with labels_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("preferred") in ("a", "b"):
                votes.setdefault(rec["pair_id"], []).append(rec["preferred"])

    out = []
    for pid, vs in votes.items():
        n_a = vs.count("a")
        if n_a * 2 == len(vs):
            continue  # split votes -> no signal
        win = "a" if n_a * 2 > len(vs) else "b"
        lose = "b" if win == "a" else "a"
        w, l = pairs_dir / f"{pid}_{win}.mid", pairs_dir / f"{pid}_{lose}.mid"
        if not (w.exists() and l.exists()):
            continue
        meta_path = pairs_dir / f"{pid}.json"
        group = pid
        if meta_path.exists():
            group = json.loads(meta_path.read_text()).get("prompt_file", pid)
        else:
            meta_path = None
        out.append((w, l, group, meta_path, win))
    return out, votes


def load_historical_pairs(csv_path: Path):
    base = csv_path.parent
    votes: dict[str, list[str]] = {}
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
            votes.setdefault(row["pair_id"], []).append(row["file_preferred"])
    out = []
    for row in rows:
        w, l = base / row["file_preferred"], base / row["file_rejected"]
        if w.exists() and l.exists():
            out.append((w, l, f"hist:{row['pair_id']}", None, None))
    return out, votes


def self_consistency(votes: dict[str, list[str]]) -> tuple[float, int]:
    """Majority-agreement rate over pairs voted on more than once."""
    agree = total = 0
    for vs in votes.values():
        if len(vs) < 2:
            continue
        top = max(vs.count(v) for v in set(vs))
        agree += top
        total += len(vs)
    return (agree / total if total else float("nan")), total


def feature_vector(midi_path: Path) -> np.ndarray | None:
    try:
        m = compute_metrics(midi_path)
    except Exception:
        return None
    if m["n_notes"] == 0:
        return None
    return np.array([float(m[k]) for k in FEATURES])


def build_diffs(pairs) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Rows of f(winner) - f(loser) + group ids. Drift features are appended only
    when every pair carries a meta JSON (pure label_app data) — mixing them
    with zero-filled historical rows would bias the fit.
    """
    use_drift = all(meta is not None for *_, meta, _win in pairs)
    tokenizer = None
    if use_drift:
        from v2.tokenizer_v2 import build_tokenizer
        tokenizer = build_tokenizer()

    cache: dict[Path, np.ndarray | None] = {}

    def feats(p: Path):
        if p not in cache:
            cache[p] = feature_vector(p)
        return cache[p]

    diffs, groups = [], []
    for w, l, g, meta_path, win in pairs:
        fw, fl = feats(w), feats(l)
        if fw is None or fl is None:
            continue
        row = fw - fl
        if use_drift:
            meta = json.loads(meta_path.read_text())
            lose = "b" if win == "a" else "a"
            dw = drift_vector(meta[f"cont_{win}_ids"], tokenizer)
            dl = drift_vector(meta[f"cont_{lose}_ids"], tokenizer)
            if dw is None or dl is None:
                continue
            row = np.concatenate([row, dw - dl])
        diffs.append(row)
        groups.append(g)
    names = FEATURES + (DRIFT_FEATURES if use_drift else [])
    return np.array(diffs), groups, names


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


def logo_accuracy(diffs: np.ndarray, groups: list[str], l2: float) -> tuple[float, np.ndarray]:
    """
    Leave-one-group-out accuracy, where a group is one seed prompt: every pair
    sharing the held-out prompt is excluded from the fit, so the score never
    benefits from having seen that prompt's continuation distribution.
    Returns (accuracy, held-out win probabilities per pair).
    """
    uniq = sorted(set(groups))
    garr = np.array(groups)
    probs = np.zeros(len(diffs))
    for g in uniq:
        test = garr == g
        w = fit_bt(diffs[~test], l2=l2)
        probs[test] = 1.0 / (1.0 + np.exp(-(diffs[test] @ w)))
    return float((probs > 0.5).mean()), probs


def calibration_table(probs: np.ndarray, n_bins: int = 4) -> list[tuple[float, float, int]]:
    """
    (mean predicted, empirical rate, N) per bin. Rows are all oriented
    winner-first, so we symmetrize: each pair contributes (p, hit=1) and
    (1-p, hit=0). A calibrated reward's bins sit near the diagonal.
    """
    p_all = np.concatenate([probs, 1.0 - probs])
    y_all = np.concatenate([np.ones(len(probs)), np.zeros(len(probs))])
    edges = np.quantile(p_all, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p_all >= lo) & ((p_all < hi) if i < n_bins - 1 else (p_all <= hi))
        if m.sum():
            rows.append((float(p_all[m].mean()),
                         float(y_all[m].mean()), int(m.sum())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, default=None,
                   help="labels.jsonl from v2.label_app")
    p.add_argument("--historical", type=Path, default=None,
                   help="consolidated preferences.csv")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("evals/reward_spec.json"))
    args = p.parse_args()

    pairs, all_votes = [], {}
    if args.labels and args.labels.exists():
        got, votes = load_label_app_pairs(args.labels)
        print(f"[align] {len(got)} pairs from {args.labels}")
        pairs += got
        all_votes.update(votes)
    if args.historical and args.historical.exists():
        got, votes = load_historical_pairs(args.historical)
        print(f"[align] {len(got)} pairs from {args.historical}")
        pairs += got
        all_votes.update({f"hist:{k}": v for k, v in votes.items()})
    if not pairs:
        raise SystemExit("no preference pairs found")

    consistency, n_repeat_votes = self_consistency(all_votes)
    if n_repeat_votes:
        print(f"[align] labeler self-consistency: {consistency:.2f} over "
              f"{n_repeat_votes} repeat votes — this is the accuracy ceiling "
              f"for any reward fit on these labels")

    diffs, groups, feat_names = build_diffs(pairs)
    print(f"[align] {len(diffs)} usable pairs (both sides parse, non-empty), "
          f"{len(set(groups))} prompt groups, features: {len(feat_names)}")
    if len(diffs) < 10:
        print("[align] WARNING: very few pairs — treat every number below "
              "as anecdote, not signal")

    # normalize by the std of diffs so weights are comparable across features
    std = diffs.std(axis=0)
    std[std == 0] = 1.0
    norm_diffs = diffs / std

    print(f"\nper-metric agreement with human votes "
          f"(0.5 = chance; N excludes zero-diff pairs):")
    for i, name in enumerate(feat_names):
        nz = diffs[:, i] != 0
        n = int(nz.sum())
        agree = float((diffs[nz, i] > 0).mean()) if n else float("nan")
        print(f"  {name:22s}  {agree:5.2f}  (N={n})")

    w = fit_bt(norm_diffs, l2=args.l2)
    train_acc = float(((norm_diffs @ w) > 0).mean())
    logo, heldout_probs = logo_accuracy(norm_diffs, groups, l2=args.l2)
    print(f"\nfitted Bradley–Terry reward:")
    for name, wi in sorted(zip(feat_names, w), key=lambda t: -abs(t[1])):
        print(f"  {name:22s}  {wi:+.3f}")
    print(f"\ntrain accuracy {train_acc:.2f}   "
          f"held-out (leave-one-prompt-out) accuracy {logo:.2f}")

    print(f"\ncalibration (held-out; predicted vs empirical, diagonal = calibrated):")
    for pred, emp, n in calibration_table(heldout_probs):
        print(f"  predicted {pred:.2f}  empirical {emp:.2f}  (N={n})")

    ceiling = consistency if n_repeat_votes else None
    if ceiling is not None and not np.isnan(ceiling):
        gap = ceiling - logo
        print(f"\nreward vs labeler ceiling: {logo:.2f} vs {ceiling:.2f} "
              f"(gap {gap:+.2f})")
    verdict = ("USABLE as RL reward" if logo >= 0.65 else
               "NOT yet reliable — collect more labels before GRPO")
    print(f"verdict: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "features": feat_names,
        "diff_std": std.tolist(),
        "weights": w.tolist(),
        "n_pairs": len(diffs),
        "n_prompt_groups": len(set(groups)),
        "heldout_accuracy": logo,
        "self_consistency": ceiling,
        "calibration": calibration_table(heldout_probs),
    }, indent=2))
    print(f"[align] wrote {args.out}")


if __name__ == "__main__":
    main()
