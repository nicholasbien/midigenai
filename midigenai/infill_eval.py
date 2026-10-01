"""How good is infill? A report over one pairgen infill set.

Two kinds of evidence, kept apart:

  * **Preferences** (human votes from the labeling hub, or `llm_judge label`
    output). On a `--vs-original` set side b is the source's own bars, so the
    model's win rate against it is the headline: 50% means a listener cannot
    tell the fill from what was really there; far below means they can.
    Reported with a Wilson interval and split by seed source and gap length.
  * **Label-free checks** on every pair, model fill vs original bars (on a
    vs-original set) or fill vs fill: empty fills, fills that came back short
    of their bars, note density and pitch-class fit against the kept bars,
    and how often the model reproduced the original exactly (loops invite
    it; it is not a defect, but it is not a test of invention either).

    python -m midigenai.infill_eval --set evals/labeling_infill_v5rl_vs_orig
    python -m midigenai.infill_eval --set ... --labels evals/reward/judge_infill_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _split(path: Path, gap_beats) -> tuple[list, list, float]:
    """(fill notes, kept notes, kept bars) of one side file; notes as
    (start_beat, pitch, is_drum)."""
    from symusic import Score
    sc = Score(str(path))
    tpq = max(sc.ticks_per_quarter, 1)
    g0, g1 = gap_beats
    fill, kept = [], []
    end = 0.0
    for t in sc.tracks:
        for n in t.notes:
            b = n.start / tpq
            end = max(end, (n.start + n.duration) / tpq)
            (fill if g0 <= b < g1 else kept).append((b, int(n.pitch), bool(t.is_drum)))
    kept_bars = max(1e-9, (max(end, g1) - (g1 - g0)) / 4.0)
    return fill, kept, kept_bars


def _pc_cosine(a, b) -> float:
    ha, hb = [0] * 12, [0] * 12
    for _, p, d in a:
        if not d:
            ha[p % 12] += 1
    for _, p, d in b:
        if not d:
            hb[p % 12] += 1
    na, nb = math.sqrt(sum(x * x for x in ha)), math.sqrt(sum(x * x for x in hb))
    if not na or not nb:
        return float("nan")
    return sum(x * y for x, y in zip(ha, hb)) / (na * nb)


def side_metrics(pairs_dir: Path, pid: str, side: str, meta: dict) -> dict:
    gb = meta["gap_beats"]
    fill, kept, kept_bars = _split(pairs_dir / f"{pid}_{side}.mid", gb)
    gap_bars = (gb[1] - gb[0]) / 4.0
    kept_density = len(kept) / kept_bars
    return {
        "notes": len(fill),
        "empty": not fill,
        "bars_ok": bool(meta.get(f"bars_ok_{side}", True)),
        # fill notes per bar over kept notes per bar: 1.0 = as busy as its surroundings
        "density_ratio": (len(fill) / gap_bars) / kept_density if kept_density else float("nan"),
        "pc_fit": _pc_cosine(fill, kept),
        "_fill": sorted((round(b, 3), p) for b, p, _ in fill),
    }


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _source(name: str) -> str:
    return "ableton" if name.startswith("ableton") else "fma" if name.startswith("transcribed") else "other"


def load_labels(paths: list[Path]) -> dict[str, list[str]]:
    """pair_id -> list of preferred sides ('a' / 'b' / 'tie' / other non-votes)."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pref = r.get("preferred") or r.get("choice")
            if pref in ("a", "b", "tie"):
                out[r["pair_id"]].append(pref)
    return out


def report(set_dir: Path, label_paths: list[Path] | None = None) -> dict:
    pairs_dir = set_dir / "pairs" if (set_dir / "pairs").is_dir() else set_dir
    metas = {}
    for mp in sorted(pairs_dir.glob("*.json")):
        m = json.loads(mp.read_text())
        if m.get("mode") == "infill":
            metas[m["pair_id"]] = m
    if not metas:
        raise SystemExit(f"no infill pairs in {pairs_dir}")
    vs_orig = any(m.get("vs_original") for m in metas.values())

    rows = []
    for pid, m in metas.items():
        a = side_metrics(pairs_dir, pid, "a", m)
        b = side_metrics(pairs_dir, pid, "b", m)
        rows.append({"pid": pid, "meta": m, "a": a, "b": b,
                     "identical": a["_fill"] == b["_fill"]})

    # side a is always a model fill; on a vs-original set b is the reference
    model_sides = [r["a"] for r in rows] + ([] if vs_orig else [r["b"] for r in rows])
    out = {"set": set_dir.name, "pairs": len(rows), "vs_original": vs_orig,
           "model": {
               "empty_rate": _mean([s["empty"] for s in model_sides]),
               "short_rate": _mean([not s["bars_ok"] for s in model_sides]),
               "density_ratio": _mean([s["density_ratio"] for s in model_sides]),
               "pc_fit": _mean([s["pc_fit"] for s in model_sides]),
           }}
    if vs_orig:
        out["original"] = {
            "density_ratio": _mean([r["b"]["density_ratio"] for r in rows]),
            "pc_fit": _mean([r["b"]["pc_fit"] for r in rows]),
        }
        out["identical_to_original"] = _mean([r["identical"] for r in rows])

    labels = load_labels(label_paths or [set_dir / "labels.jsonl"])
    if labels:
        wins = defaultdict(lambda: [0, 0, 0])      # key -> [model wins, decided, ties]
        for r in rows:
            for pref in labels.get(r["pid"], []):
                m = r["meta"]
                for key in ("all", f"source={_source(m['prompt_file'])}",
                            f"gap={m['source_bars']} bar{'s' if m['source_bars'] > 1 else ''}"):
                    w = wins[key]
                    if pref == "tie":
                        w[2] += 1
                    else:
                        w[1] += 1
                        w[0] += pref == "a"
        name = "model_win_rate_vs_original" if vs_orig else "side_a_rate"
        out["preferences"] = {
            k: {name: (w[0] / w[1]) if w[1] else None, "decided": w[1], "ties": w[2],
                "ci95": wilson(w[0], w[1])}
            for k, w in sorted(wins.items())}
    return out


def print_report(r: dict) -> None:
    print(f"[infill] {r['set']}: {r['pairs']} pairs"
          f"{' (side b = original bars)' if r['vs_original'] else ''}")
    m = r["model"]
    print(f"  model fills   empty {m['empty_rate']:.1%}  short of bars {m['short_rate']:.1%}  "
          f"density vs kept bars {m['density_ratio']:.2f}  pitch-class fit {m['pc_fit']:.3f}")
    if r["vs_original"]:
        o = r["original"]
        print(f"  original bars density vs kept bars {o['density_ratio']:.2f}  "
              f"pitch-class fit {o['pc_fit']:.3f}")
        print(f"  model fill identical to the original: {r['identical_to_original']:.1%}")
    prefs = r.get("preferences")
    if not prefs:
        print("  no labels yet")
        return
    for k, v in prefs.items():
        rate = v.get("model_win_rate_vs_original", v.get("side_a_rate"))
        lo, hi = v["ci95"]
        what = "model wins vs original" if r["vs_original"] else "side a preferred"
        rs = f"{rate:.1%} [{lo:.0%}-{hi:.0%}]" if rate is not None else "-"
        print(f"  {k:<16} {what} {rs}  ({v['decided']} decided, {v['ties']} ties)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True,
                    help="a pairgen --mode infill output dir (with pairs/)")
    ap.add_argument("--labels", type=Path, nargs="*", default=None,
                    help="labels.jsonl files (hub votes and/or llm_judge label output); "
                         "default: <set>/labels.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    r = report(a.set, a.labels)
    print_report(r)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
