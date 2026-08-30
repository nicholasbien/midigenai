"""
Quantitative evaluation for v2.

Operates on output MIDI files (model-agnostic) so v1 vs v2 comparisons are
fair regardless of tokenization. Metrics chosen to surface common failure modes:

- pitch_class_entropy   higher = more pitches used. Mode-collapse → low.
- scale_consistency     fraction of notes in detected key's scale (Krumhansl-Schmuckler).
                        Lower = atonal/random.
- polyphony_rate        avg simultaneous notes. 1.0 = monophonic; 3-5 = chordal.
- note_density_hz       notes/sec. Sanity check that the model emits content.
- pitch_range           MIDI pitch spread.
- repetition_rate       % of length-4 pitch n-grams that repeat. High = stuck loop.
- ioi_entropy           entropy of inter-onset intervals (binned 1/16th).
                        Lower = more grid-aligned.

Usage:
    python -m midigenai.eval dir_a dir_b --label-a v1 --label-b v2 --out evals/pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from symusic import Score


# Krumhansl-Kessler key profiles.
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def _all_notes(score):
    for track in score.tracks:
        if track.is_drum:
            continue
        for n in track.notes:
            yield n


def pitch_class_histogram(score):
    counts = Counter()
    for n in _all_notes(score):
        counts[n.pitch % 12] += 1
    total = sum(counts.values())
    if total == 0:
        return [0.0] * 12
    return [counts.get(pc, 0) / total for pc in range(12)]


def pitch_class_entropy(score):
    hist = pitch_class_histogram(score)
    return -sum(p * math.log2(p) for p in hist if p > 0)


def _correlate(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def detect_key(score):
    hist = pitch_class_histogram(score)
    if sum(hist) == 0:
        return 0, "major"
    best = (-2.0, 0, "major")  # corr in [-1,1]
    for tonic in range(12):
        major_profile = [KS_MAJOR[(i - tonic) % 12] for i in range(12)]
        minor_profile = [KS_MINOR[(i - tonic) % 12] for i in range(12)]
        c_maj = _correlate(hist, major_profile)
        c_min = _correlate(hist, minor_profile)
        if c_maj > best[0]:
            best = (c_maj, tonic, "major")
        if c_min > best[0]:
            best = (c_min, tonic, "minor")
    _, tonic, mode = best
    return tonic, mode


def scale_consistency(score):
    notes = list(_all_notes(score))
    if not notes:
        return 0.0
    tonic, mode = detect_key(score)
    scale_pcs = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    in_scale_pcs = {(tonic + i) % 12 for i in scale_pcs}
    in_scale = sum(1 for n in notes if (n.pitch % 12) in in_scale_pcs)
    return in_scale / len(notes)


def polyphony_rate(score):
    """Avg simultaneous notes at each onset point. O(N^2) but fine for our N."""
    notes = sorted(_all_notes(score), key=lambda n: n.start)
    if not notes:
        return 0.0
    counts = []
    for i, n in enumerate(notes):
        active = 1
        # walk back while previous notes still ringing
        for j in range(i - 1, -1, -1):
            other = notes[j]
            if other.start + other.duration > n.start:
                active += 1
            # notes are sorted by start; if their start is far enough back that
            # even the longest prior note ended before n.start, we could break.
            # Skipping that optimization — N is small for our samples.
        counts.append(active)
    return mean(counts) if counts else 0.0


def note_density_hz(score):
    notes = list(_all_notes(score))
    if not notes:
        return 0.0
    tpq = score.ticks_per_quarter
    bpm = float(score.tempos[0].qpm) if len(score.tempos) > 0 else 120.0
    seconds_per_tick = 60.0 / (bpm * tpq)
    duration = max(n.start + n.duration for n in notes) * seconds_per_tick
    return len(notes) / max(duration, 1e-6)


def pitch_range(score):
    pitches = [n.pitch for n in _all_notes(score)]
    if not pitches:
        return 0
    return max(pitches) - min(pitches)


def repetition_rate(score, n=4):
    notes = sorted(_all_notes(score), key=lambda n: n.start)
    pitches = [x.pitch for x in notes]
    if len(pitches) < n:
        return 0.0
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def ioi_entropy(score):
    notes = sorted(_all_notes(score), key=lambda n: n.start)
    if len(notes) < 2:
        return 0.0
    tpq = score.ticks_per_quarter
    sixteenth = tpq / 4
    iois = []
    for i in range(1, len(notes)):
        d = notes[i].start - notes[i - 1].start
        if d > 0:
            iois.append(round(d / sixteenth))
    if not iois:
        return 0.0
    counts = Counter(iois)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def compute_metrics(midi_path):
    score = Score(str(midi_path))
    return {
        "n_notes": sum(1 for _ in _all_notes(score)),
        "pitch_class_entropy": pitch_class_entropy(score),
        "scale_consistency": scale_consistency(score),
        "polyphony_rate": polyphony_rate(score),
        "note_density_hz": note_density_hz(score),
        "pitch_range": pitch_range(score),
        "repetition_rate": repetition_rate(score, n=4),
        "ioi_entropy": ioi_entropy(score),
    }


def aggregate(per_file):
    if not per_file:
        return {}
    summary = {}
    for k in per_file[0]:
        if k == "file":
            continue
        vals = [m[k] for m in per_file if isinstance(m[k], (int, float))]
        if vals:
            summary[k] = {
                "mean": mean(vals),
                "median": median(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
            }
    return summary


def evaluate(midi_dir):
    files = sorted(Path(midi_dir).glob("*.mid"))
    per_file = []
    for f in files:
        try:
            per_file.append({"file": f.name, **compute_metrics(f)})
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
    return per_file, aggregate(per_file)


def print_table(agg_a, agg_b, label_a, label_b):
    keys = list(agg_a)
    width = max(25, max(len(k) for k in keys))
    header = f"{'metric':<{width}}  {label_a+' mean':>14s}  {label_b+' mean':>14s}  {'Δ (B-A)':>12s}"
    print(header)
    print("-" * len(header))
    for k in keys:
        ma = agg_a[k]["mean"]
        mb = agg_b.get(k, {}).get("mean", 0.0)
        d = mb - ma
        print(f"{k:<{width}}  {ma:14.4f}  {mb:14.4f}  {d:+12.4f}")


def compare(dir_a, dir_b, label_a, label_b):
    pf_a, agg_a = evaluate(dir_a)
    pf_b, agg_b = evaluate(dir_b)
    print(f"=== {label_a}: {len(pf_a)} files | {label_b}: {len(pf_b)} files ===")
    print_table(agg_a, agg_b, label_a, label_b)
    return {
        label_a: {"per_file": pf_a, "agg": agg_a},
        label_b: {"per_file": pf_b, "agg": agg_b},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = compare(args.dir_a, args.dir_b, args.label_a, args.label_b)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.out}")
