"""
Checkpoint evaluation harness: turn any checkpoint into a one-command
scorecard of GENERATION BEHAVIOR — the things val loss can't see.

For each held-out prompt it generates continuations and measures:
- termination: did generation stop via EOS or hit the token cap?
  (models trained before EOS-in-docs can't ever stop on their own)
- length: tokens/notes/seconds actually generated
- degradation drift: 2nd-half minus 1st-half repetition/density/entropy
  ("starts good then wanders")
- prompt coherence: pitch-class histogram correlation between prompt and
  continuation (does it stay in key / related material?)
- the eval_v2 musical battery on the continuation only

Score one checkpoint:
    python -m midigenai.eval_checkpoint --checkpoint runs/pilot_baseline/ckpt_final.pt \\
        --tokenizer ~/midigenai_data/corpus_pilot/tokenizer.json \\
        --prompts evals/prompts --out evals/scorecards/pilot_baseline.json

Compare scorecards:
    python -m midigenai.eval_checkpoint --compare evals/scorecards/*.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

METRICS = [
    "eos_rate", "gen_tokens", "gen_notes", "gen_seconds",
    "repetition_rate", "note_density_hz", "pitch_class_entropy",
    "scale_consistency", "polyphony_rate", "ioi_entropy",
    "repetition_drift", "density_drift", "pce_drift",
    "prompt_coherence",
]
# direction hints for the compare table: +1 higher is better, -1 lower, 0 neutral
DIRECTION = {"eos_rate": +1, "repetition_rate": -1, "repetition_drift": -1,
             "scale_consistency": +1, "prompt_coherence": +1}


def evaluate_checkpoint(checkpoint: str, tokenizer: str | None, prompts_dir: Path,
                        n_prompts: int, gens_per_prompt: int, prompt_tokens: int,
                        max_new_tokens: int, temperature: float, top_k: int,
                        seed: int) -> dict:
    import numpy as np

    from midigenai.eval import (ioi_entropy, note_density_hz,
                                pitch_class_entropy, pitch_class_histogram,
                                polyphony_rate, repetition_rate,
                                scale_consistency, _correlate)
    from midigenai.generate import Generator
    from midigenai.tokenizer import normalize_drums

    gen = Generator(checkpoint, tokenizer)
    from symusic import Score

    files = sorted(prompts_dir.glob("*.mid"))
    rng = random.Random(seed)
    picked = rng.sample(files, min(n_prompts, len(files)))

    def cont_score(prompt_ids, new_ids):
        """Continuation-only Score: decode with prompt context, cut, re-zero."""
        full = gen.tokenizer.decode(list(prompt_ids) + list(new_ids))
        cut = gen.tokenizer.decode(list(prompt_ids)).end()
        for t in full.tracks:
            kept = [n for n in t.notes if n.start >= cut]
            for n in kept:
                n.start -= cut
            t.notes = kept
        return full

    rows = []
    for f in picked:
        score = Score(str(f))
        normalize_drums(score, f.name)
        ids = gen.tokenizer(score).ids
        if len(ids) < 16:
            continue
        if len(ids) > prompt_tokens:
            ids = ids[:prompt_tokens]
        prompt_hist = pitch_class_histogram(gen.tokenizer.decode(list(ids)))
        for g in range(gens_per_prompt):
            new_ids = list(gen.generate_ids(ids, max_new_tokens=max_new_tokens,
                                            temperature=temperature, top_k=top_k))
            stopped = len(new_ids) < max_new_tokens  # generate halts on EOS
            try:
                cont = cont_score(ids, new_ids)
            except Exception:
                continue
            n_notes = sum(len(t.notes) for t in cont.tracks)
            tpq = max(cont.ticks_per_quarter, 1)
            seconds = cont.end() / tpq * 0.5  # 120bpm equivalent
            row = {
                "prompt": f.name, "gen": g,
                "eos_rate": 1.0 if stopped else 0.0,
                "gen_tokens": len(new_ids), "gen_notes": n_notes,
                "gen_seconds": round(seconds, 1),
            }
            if n_notes >= 4:
                row.update({
                    "repetition_rate": repetition_rate(cont),
                    "note_density_hz": note_density_hz(cont),
                    "pitch_class_entropy": pitch_class_entropy(cont),
                    "scale_consistency": scale_consistency(cont),
                    "polyphony_rate": polyphony_rate(cont),
                    "ioi_entropy": ioi_entropy(cont),
                    "prompt_coherence": _correlate(
                        prompt_hist, pitch_class_histogram(cont)),
                })
                half = len(new_ids) // 2
                if half >= 16:
                    try:
                        s1 = gen.tokenizer.decode(list(new_ids[:half]))
                        s2 = gen.tokenizer.decode(list(new_ids[half:]))
                        row.update({
                            "repetition_drift": repetition_rate(s2) - repetition_rate(s1),
                            "density_drift": note_density_hz(s2) - note_density_hz(s1),
                            "pce_drift": pitch_class_entropy(s2) - pitch_class_entropy(s1),
                        })
                    except Exception:
                        pass
            rows.append(row)

    agg = {}
    for m in METRICS:
        vals = [r[m] for r in rows if m in r and isinstance(r[m], (int, float))]
        if vals:
            agg[m] = {"mean": round(st.mean(vals), 4),
                      "median": round(st.median(vals), 4)}
    return {
        "checkpoint": str(checkpoint),
        "n_generations": len(rows),
        "params": {"prompt_tokens": prompt_tokens, "max_new_tokens": max_new_tokens,
                   "temperature": temperature, "top_k": top_k, "seed": seed},
        "aggregate": agg,
        "rows": rows,
    }


def compare(paths: list[Path]) -> None:
    cards = []
    for p in paths:
        d = json.loads(p.read_text())
        cards.append((p.stem, d["aggregate"]))
    name_w = max(len(n) for n, _ in cards) + 2
    print(f"{'metric':22s}" + "".join(f"{n:>{max(len(n)+2, 12)}s}" for n, _ in cards))
    for m in METRICS:
        cells = []
        for n, agg in cards:
            v = agg.get(m, {}).get("mean")
            cells.append(f"{v:>{max(len(n)+2, 12)}.3f}" if v is not None
                         else f"{'—':>{max(len(n)+2, 12)}s}")
        arrow = {1: " (higher+)", -1: " (lower+)"}.get(DIRECTION.get(m, 0), "")
        print(f"{m + arrow:22s}" + "".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", nargs="*", type=Path, default=None,
                   help="scorecard JSONs to compare side by side")
    p.add_argument("--checkpoint")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompts", type=Path, default=Path("evals/prompts"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--gens-per-prompt", type=int, default=2)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint required (or --compare)")
    card = evaluate_checkpoint(args.checkpoint, args.tokenizer, args.prompts,
                               args.n_prompts, args.gens_per_prompt,
                               args.prompt_tokens, args.max_new_tokens,
                               args.temperature, args.top_k, args.seed)
    out = args.out or Path(f"evals/scorecards/{Path(args.checkpoint).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print(f"[eval] {card['n_generations']} generations -> {out}")
    for m, v in card["aggregate"].items():
        print(f"  {m:22s} mean {v['mean']:>8.3f}   median {v['median']:>8.3f}")


if __name__ == "__main__":
    main()
