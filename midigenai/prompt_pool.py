"""One weighted prompt pool for the RL loop.

With a single judge (fit_only, luna) and a single reward across every seed
source, GRPO can sample prompts from one pool instead of three runs. This
builds that pool as a directory of symlinks, weighted by source, so pairgen
and grpo take it as an ordinary --prompts directory and the weighting is a
number in a command rather than a fact someone remembers.

    python -m midigenai.prompt_pool --out evals/prompts_pool_v5 -n 1200 \\
        --source ableton=evals/prompts_ableton:0.5 \\
        --source fma=~/midigenai-v4/evals/prompts_fma:0.25 \\
        --source val=~/midigenai-v4/evals/prompts_heldout:0.25

Every source here is held out of training by construction: Ableton is
excluded from the corpus entirely, the FMA seeds are enforced by
--exclude-ids, and prompts_heldout is the val split. FMA seeds the training
filter called degenerate or empty are skipped when heldout.json is present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _files(src: Path) -> list[Path]:
    files = sorted(src.glob("*.mid"))
    meta = src / "heldout.json"
    if meta.exists():
        h = json.loads(meta.read_text())
        rows = h if isinstance(h, list) else h.get("files", list(h.values()))
        bad = {r.get("filename") or r.get("file") for r in rows
               if r.get("verdict") in ("degenerate", "empty")}
        bad_ids = {str(r.get("id")) for r in rows if r.get("verdict") in ("degenerate", "empty")}
        files = [f for f in files if f.name not in bad and not any(i and i in f.name for i in bad_ids)]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=1200, help="total prompts in the pool")
    ap.add_argument("--source", action="append", required=True,
                    help="name=dir:weight, repeatable; weights are normalised")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    srcs = []
    for spec in a.source:
        name, rest = spec.split("=", 1)
        d, w = rest.rsplit(":", 1)
        srcs.append((name, Path(os.path.expanduser(d)), float(w)))
    total_w = sum(w for _, _, w in srcs)
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("*.mid"):
        old.unlink()
    made = {}
    for name, d, w in srcs:
        files = _files(d)
        want = round(a.n * w / total_w)
        pick = rng.sample(files, min(want, len(files)))
        for f in pick:
            (a.out / f"pool_{name}_{f.name}").symlink_to(f.resolve())
        made[name] = (len(pick), len(files))
    (a.out / "POOL.json").write_text(json.dumps(
        {"n": a.n, "sources": {n: {"dir": str(d), "weight": w, "picked": made[n][0], "available": made[n][1]}
                                 for n, d, w in srcs}, "seed": a.seed}, indent=1))
    print(f"[pool] {a.out}: " + ", ".join(f"{n} {made[n][0]}/{made[n][1]} (w={w:g})" for n, _, w in srcs))


if __name__ == "__main__":
    main()
