"""
Materialize a large held-out prompt set for labeling / eval.

Prompts must be files the model never trained on. `build_dataset` splits
train/val by hashing the file path, so the same hash identifies genuine
held-out files here — no separate bookkeeping, and the pool is as big as the
val split (~0.5% of the corpus, thousands of files).

    python -m midigenai.make_prompt_set --out evals/prompts_heldout --n 400

Stratified across sources so piano transcriptions can't dominate, and
filtered to files long and dense enough to make a usable prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_MANIFEST = Path.home() / "midigenai_data" / "manifest_plus_ggm_dedup.jsonl"
SOURCES = ("lakh", "lamd", "aria", "gigamidi", "maestro", "pop909", "giantmidi")


def source_of(path: str) -> str | None:
    for s in SOURCES:
        if f"/raw/{s}/" in path:
            return s
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--min-notes", type=int, default=120)
    p.add_argument("--min-seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from midigenai.data.build_dataset import VAL_FRACTION, split_by_path

    by_source: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("n_notes", 0) < args.min_notes:
                continue
            if e.get("duration_seconds", 0) < args.min_seconds:
                continue
            if split_by_path(e["path"], VAL_FRACTION) != "val":
                continue
            src = source_of(e["path"])
            if src:
                by_source[src].append(e)
    print("[prompts] held-out pool:", {s: len(v) for s, v in sorted(by_source.items())})

    rng = random.Random(args.seed)
    live = {s: v for s, v in by_source.items() if v}
    for v in live.values():
        rng.shuffle(v)
    picked: list[tuple[str, dict]] = []
    i = 0
    while len(picked) < args.n and live:
        for s in sorted(live):
            if i < len(live[s]):
                picked.append((s, live[s][i]))
                if len(picked) >= args.n:
                    break
        i += 1
        if all(i >= len(v) for v in live.values()):
            break

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for src, e in picked:
        stem = Path(e["path"]).stem[:24]
        dst = args.out / f"val_{src}_{stem}.mid"
        if dst.exists():
            continue
        try:
            shutil.copy(e["path"], dst)
            written += 1
        except OSError:
            continue
    print(f"[prompts] wrote {written} new prompts to {args.out} "
          f"({len(list(args.out.glob('*.mid')))} total)")


if __name__ == "__main__":
    main()
