"""
Sample candidate prompts for a held-out eval set.

Reads a clean.py manifest, filters to entries that landed in the build_dataset.py
val split (using the same path-hash function so the choice is deterministic),
stratified-samples across source datasets, and copies selected MIDIs into a
candidate folder for hand-curation.

Workflow:
    1. python -m midigenai.data.sample_eval_set --manifest data/manifest_full.jsonl \\
           --out evals/eval_set_candidates --per-source 20
    2. Listen to each candidate, delete the bad ones (broken/awkward starts).
    3. mv what survives → evals/eval_set/  and commit.
    4. Use evals/eval_set/ as the prompt set for future eval runs.

The val split is the same `split_by_path()` from build_dataset.py — so files
selected here are the same files held out during training (guaranteed never seen).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from midigenai.data.browse_corpus import detect_source


# Must match build_dataset.py's split. Keep these constants in sync.
VAL_FRACTION = 0.005


def split_by_path(path: str, val_fraction: float = VAL_FRACTION) -> str:
    h = int(hashlib.sha1(path.encode()).hexdigest(), 16)
    return "val" if (h % 1_000_000) / 1_000_000 < val_fraction else "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="JSONL manifest from clean.py")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for candidate MIDIs")
    p.add_argument("--per-source", type=int, default=20,
                   help="how many candidates per source dataset")
    p.add_argument("--min-notes", type=int, default=64,
                   help="skip prompts with fewer than this many notes")
    p.add_argument("--max-notes", type=int, default=2000,
                   help="skip overlong prompts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    val_by_source: dict[str, list[dict]] = {}
    with args.manifest.open() as f:
        for line in f:
            e = json.loads(line)
            if split_by_path(e["path"]) != "val":
                continue
            n = e.get("n_notes", 0)
            if n < args.min_notes or n > args.max_notes:
                continue
            src = detect_source(e["path"])
            val_by_source.setdefault(src, []).append(e)

    rng = random.Random(args.seed)
    chosen = []
    for src, entries in sorted(val_by_source.items()):
        rng.shuffle(entries)
        picked = entries[: args.per_source]
        chosen.extend((src, e) for e in picked)
        print(f"  {src:10s}: {len(picked):>3d} of {len(entries):>5d} val candidates")

    print(f"\ncopying {len(chosen)} files to {args.out}")
    for src, e in chosen:
        srcpath = Path(e["path"])
        if not srcpath.exists():
            print(f"  [skip] missing on disk: {srcpath}")
            continue
        # Prefix with source so the eval set is browsable per-dataset.
        dst = args.out / f"{src}__{srcpath.stem}.mid"
        shutil.copy2(srcpath, dst)

    print(f"\nNext: listen to {args.out}/*.mid, delete bad ones, then "
          f"`mv {args.out} evals/eval_set` (or rename).")


if __name__ == "__main__":
    main()
