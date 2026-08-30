"""
Score corpus files by per-token NLL under a trained checkpoint — the
positive-quality signal the heuristic filters can't provide.

Low NLL  = predictable, idiomatic music (the model has seen the pattern).
High NLL = surprising: corruption, junk quantization, atonal noise — or
genuinely novel music. The explorer's "quality extremes (scored)" view plays
both tails so a human decides where the junk threshold is; the scores then
feed filtering (drop the top tail) or mixture reweighting.

Run (scores 500 random manifest entries with the hub model):
    python -m midigenai.score_corpus \\
        --manifest ~/midigenai_data/manifest_all_dedup.jsonl \\
        --out ~/midigenai_data/scores_pilot.jsonl --sample 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def score_file(gen, path: Path, max_tokens: int) -> dict | None:
    from symusic import Score

    from midigenai.tokenizer import normalize_drums
    try:
        score = Score(str(path))
        normalize_drums(score, path.name)
        ids = gen.tokenizer(score).ids[:max_tokens]
    except Exception:
        return None
    if len(ids) < 32:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=gen.device)
    with torch.no_grad():
        logits, _ = gen.model(x)
        nll = F.cross_entropy(
            logits[0, :-1].float(), x[0, 1:], reduction="mean").item()
    return {"path": str(path), "n_tokens": len(ids), "nll": round(nll, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--hub-version", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.checkpoint:
        from midigenai.generate import Generator
        gen = Generator(args.checkpoint, args.tokenizer, backend="torch")
    else:
        from midigenai.hub import load_from_hub
        gen = load_from_hub(version=args.hub_version, backend="torch")

    entries = [json.loads(line) for line in args.manifest.open() if line.strip()]
    rng = random.Random(args.seed)
    picked = rng.sample(entries, min(args.sample, len(entries)))

    done = 0
    with args.out.open("w") as f:
        for e in picked:
            row = score_file(gen, Path(e["path"]), args.max_tokens)
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"[score] {done}/{len(picked)}")
    print(f"[score] wrote {done} scores -> {args.out}")


if __name__ == "__main__":
    main()
