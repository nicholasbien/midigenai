"""
Drop or trim degenerate transcriptions before they reach training.

A transcriber that loses the thread emits one pitch hammered at a fixed
spacing. The tempting filter is 4-gram repetition, but that would be a
mistake here: the median transcribed electronic clip scores 0.62 and most of
those are legitimate loops - repetition is the genre, not the failure. The
failure looks different: a tiny pitch alphabet AND a single onset spacing
repeated throughout.

Measured on 983 FMA transcriptions: 86% clean, 6% degenerate part-way (those
are trimmed at the seam, keeping 25s on average), 8% degenerate from the
start (dropped).

    python -m midigenai.data.transcript_filter --in <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

WINDOW_SECONDS = 5.0
MIN_KEEP_WINDOWS = 3          # shorter than this and there is nothing worth keeping


def _windows(score, secs: float = WINDOW_SECONDS):
    tpq = max(score.ticks_per_quarter, 1)
    bpm = score.tempos[0].qpm if len(score.tempos) else 120.0
    per = collections.defaultdict(list)
    for track in score.tracks:
        for n in track.notes:
            per[int(n.start / tpq * 60 / bpm // secs)].append((n.start, n.pitch))
    return per


# Two presets. "strict" is the original: a 5 s window is degenerate if one
# pitch takes >60% of notes, or >45% with a near-constant onset spacing, or
# the window uses <=2 pitches. "loose" is for transcribed audio, where a
# kick+hat pattern or a held drone is legitimately two pitches on a grid --
# the user's read on the transcriptions was "sometimes they degenerate in a
# cool way", so the loose preset only flags the truly collapsed windows and
# trims rather than drops.
PRESETS = {
    "strict": dict(top=0.60, top_rep=0.45, same=0.90, min_pitches=3, min_keep_windows=3),
    "loose":  dict(top=0.75, top_rep=0.60, same=0.95, min_pitches=2, min_keep_windows=2),
}
_P = dict(PRESETS["strict"])


def set_preset(name: str) -> None:
    _P.update(PRESETS[name])


def window_is_degenerate(notes) -> bool:
    """One pitch dominating, or one onset spacing repeated with a tiny alphabet."""
    if len(notes) < 8:
        return False
    pitches = [p for _, p in notes]
    top = max(pitches.count(x) for x in set(pitches)) / len(pitches)
    onsets = sorted({t for t, _ in notes})
    gaps = collections.Counter(onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1))
    same = max(gaps.values()) / sum(gaps.values()) if gaps else 1.0
    return (top > _P["top"] or (top > _P["top_rep"] and same > _P["same"])
            or len(set(pitches)) < _P["min_pitches"])


def clean_score(score):
    """(score, verdict): 'clean', 'trimmed at N s', or None to drop."""
    per = _windows(score)
    if not per:
        return None, "empty"
    keys = sorted(per)
    flags = [window_is_degenerate(per[k]) for k in keys]
    if not any(flags):
        return score, "clean"
    first = flags.index(True)
    if first < _P["min_keep_windows"]:
        return None, "degenerate"
    cut_s = first * WINDOW_SECONDS
    tpq = max(score.ticks_per_quarter, 1)
    bpm = score.tempos[0].qpm if len(score.tempos) else 120.0
    cut_tick = int(cut_s * bpm / 60 * tpq)
    out = score.copy()
    for track in out.tracks:
        track.notes = [n for n in track.notes if n.start < cut_tick]
    return out, f"trimmed at {cut_s:.0f}s"


def main() -> None:
    from symusic import Score
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="strict")
    args = p.parse_args()
    set_preset(args.preset)
    args.dst.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    for f in sorted(args.src.rglob("*.mid")):
        try:
            sc = Score(str(f))
        except Exception:
            counts["unreadable"] += 1
            continue
        out, verdict = clean_score(sc)
        counts[verdict.split(" at ")[0]] += 1
        if out is not None and sum(len(t.notes) for t in out.tracks) >= 32:
            out.dump_midi(args.dst / f.name)
        elif out is not None:
            counts["too short after trim"] += 1
    print(f"[filter] {dict(counts)} -> {len(list(args.dst.glob('*.mid')))} files kept")


if __name__ == "__main__":
    main()
