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
    # v4 structure metrics (meaningful for any tokenizer; v4 should move them)
    "downbeat_alignment", "downbeat_delta", "bar_offset_beats",
    # accompaniment mode only
    "accomp_bars_ok", "accomp_pc_overlap", "accomp_notes",
]
# direction hints for the compare table: +1 higher is better, -1 lower, 0 neutral
DIRECTION = {"eos_rate": +1, "repetition_rate": -1, "repetition_drift": -1,
             "scale_consistency": +1, "prompt_coherence": +1,
             "downbeat_alignment": +1, "downbeat_delta": +1, "bar_offset_beats": -1,
             "accomp_bars_ok": +1, "accomp_pc_overlap": +1}


def evaluate_checkpoint(checkpoint: str, tokenizer: str | None, prompts_dir: Path,
                        n_prompts: int, gens_per_prompt: int, prompt_tokens: int,
                        max_new_tokens: int, temperature: float, top_k: int,
                        seed: int, mode: str = "continue", bars: int = 8) -> dict:
    """`mode`: "continue" (default) or "accompany" (v4 only: the prompt file's
    largest track is the condition over `bars` bars, the model writes the
    rest, and the generated parts are scored against the real other parts
    and the condition)."""
    import numpy as np

    from midigenai.eval import (ioi_entropy, note_density_hz,
                                pitch_class_entropy, pitch_class_histogram,
                                polyphony_rate, repetition_rate,
                                scale_consistency, _correlate,
                                downbeat_alignment, phrase_start_offset_beats,
                                pitch_class_overlap)
    from midigenai.generate import Generator
    from midigenai.tokenizer import normalize_drums

    gen = Generator(checkpoint, tokenizer)
    from symusic import Score
    if mode == "accompany":
        return _evaluate_accompany(gen, prompts_dir, n_prompts, gens_per_prompt,
                                   bars, temperature, top_k, seed, checkpoint)

    files = sorted(prompts_dir.glob("*.mid"))
    rng = random.Random(seed)
    picked = rng.sample(files, min(n_prompts, len(files)))

    def cont_score(prompt_ids, new_ids):
        """Continuation-only Score: decode with prompt context, cut, re-zero.
        Also returns the un-rezeroed full decode and the cut tick, so bar-grid
        metrics can be computed on the shared grid."""
        full = gen.tokenizer.decode(list(prompt_ids) + list(new_ids))
        cut = gen.tokenizer.decode(list(prompt_ids)).end()
        grid = gen.tokenizer.decode(list(prompt_ids) + list(new_ids))
        for t in full.tracks:
            kept = [n for n in t.notes if n.start >= cut]
            for n in kept:
                n.start -= cut
            t.notes = kept
        return full, grid, cut

    rows = []
    for f in picked:
        score = Score(str(f))
        normalize_drums(score, f.name)
        ids = gen.tokenizer(score).ids
        if len(ids) < 16:
            continue
        if len(ids) > prompt_tokens:
            ids = ids[:prompt_tokens]
        # v4: the prompt gets the header the builder would have given this file
        header = gen.make_header(f) if gen.v4 else []
        prompt = [*header, *ids] if header else ids
        prompt_score = gen.tokenizer.decode(list(ids))
        prompt_hist = pitch_class_histogram(prompt_score)
        prompt_align = downbeat_alignment(prompt_score)
        for g in range(gens_per_prompt):
            new_ids = list(gen.generate_ids(prompt, max_new_tokens=max_new_tokens,
                                            temperature=temperature, top_k=top_k))
            stopped = len(new_ids) < max_new_tokens  # generate halts on EOS
            try:
                cont, grid, cut = cont_score(prompt, new_ids)
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
                    "downbeat_alignment": downbeat_alignment(grid, cut),
                    "bar_offset_beats": phrase_start_offset_beats(grid, cut),
                })
                if row["downbeat_alignment"] == row["downbeat_alignment"] and prompt_align == prompt_align:
                    row["downbeat_delta"] = row["downbeat_alignment"] - prompt_align
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
        vals = [r[m] for r in rows if m in r and isinstance(r[m], (int, float))
                and r[m] == r[m]]
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


def _evaluate_accompany(gen, prompts_dir, n_prompts, gens_per_prompt, bars,
                        temperature, top_k, seed, checkpoint) -> dict:
    """v4 accompaniment scorecard. For each multi-track prompt file: take a
    `bars`-bar window, condition on its busiest track, generate the rest,
    and measure bar-count adherence, pitch-class overlap with the condition
    (vs the real accompaniment's overlap as a reference) and the usual
    battery on the generated parts."""
    from symusic import Score

    from midigenai.data.v4_docs import _subscore, _window, bar_edges, trim_leading
    from midigenai.eval import (ioi_entropy, note_density_hz, pitch_class_entropy,
                                pitch_class_overlap, polyphony_rate, repetition_rate,
                                scale_consistency)
    from midigenai.tokenizer import normalize_drums

    files = sorted(prompts_dir.glob("*.mid"))
    rng = random.Random(seed)
    rng.shuffle(files)
    rows = []
    for f in files:
        if len(rows) >= n_prompts * gens_per_prompt:
            break
        score = Score(str(f))
        normalize_drums(score, f.name)
        score = trim_leading(score)
        live = [i for i, t in enumerate(score.tracks) if len(t.notes) >= 16]
        if len(live) < 2:
            continue
        edges = bar_edges(score)
        if len(edges) - 1 < bars:
            continue
        b0 = rng.randrange(0, len(edges) - 1 - bars + 1)
        win = _window(score, edges[b0], edges[b0 + bars])
        cond_i = max(live, key=lambda i: len(score.tracks[i].notes))
        cond = _subscore(win, [cond_i])
        real = _subscore(win, [i for i in range(len(win.tracks)) if i != cond_i])
        if sum(len(t.notes) for t in cond.tracks) < 4 or sum(len(t.notes) for t in real.tracks) < 4:
            continue
        cond_ids = gen.tokenizer(cond).ids
        header = gen.make_header(None)  # instruments from the whole window:
        from midigenai.attributes import header_for_score
        header = gen.sp.header_ids_for(gen.tokenizer, header_for_score(win))
        ref_overlap = pitch_class_overlap(real, cond, tpq=win.tpq)
        for g in range(gens_per_prompt):
            new_ids = list(gen.accompany(cond_ids, bars, header=header,
                                         temperature=temperature, top_k=top_k))
            try:
                out = gen.tokenizer.decode(new_ids)
            except Exception:
                continue
            n_notes = sum(len(t.notes) for t in out.tracks)
            got_bars = gen.count_bars(new_ids)
            row = {"prompt": f.name, "gen": g, "gen_tokens": len(new_ids),
                   "accomp_notes": n_notes,
                   "accomp_bars_ok": 1.0 if got_bars == bars else 0.0,
                   "eos_rate": 1.0 if (new_ids and new_ids[-1] == gen.eos_id) else 0.0}
            if n_notes >= 4:
                ov = pitch_class_overlap(out, cond, tpq=out.ticks_per_quarter,
                                         bar_ticks=out.ticks_per_quarter * 4)
                row.update({
                    "accomp_pc_overlap": ov,
                    "accomp_pc_overlap_ref": ref_overlap,
                    "repetition_rate": repetition_rate(out),
                    "note_density_hz": note_density_hz(out),
                    "pitch_class_entropy": pitch_class_entropy(out),
                    "scale_consistency": scale_consistency(out),
                    "polyphony_rate": polyphony_rate(out),
                    "ioi_entropy": ioi_entropy(out),
                })
            rows.append(row)
    agg = {}
    for m in METRICS + ["accomp_pc_overlap_ref"]:
        vals = [r[m] for r in rows if m in r and isinstance(r[m], (int, float))
                and r[m] == r[m]]
        if vals:
            agg[m] = {"mean": round(st.mean(vals), 4), "median": round(st.median(vals), 4)}
    return {"checkpoint": str(checkpoint), "mode": "accompany", "bars": bars,
            "n_generations": len(rows), "aggregate": agg, "rows": rows}


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
    p.add_argument("--mode", choices=["continue", "accompany"], default="continue",
                   help="accompany: v4 accompaniment scorecard (see _evaluate_accompany)")
    p.add_argument("--bars", type=int, default=8, help="accompany: window length")
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint required (or --compare)")
    card = evaluate_checkpoint(args.checkpoint, args.tokenizer, args.prompts,
                               args.n_prompts, args.gens_per_prompt,
                               args.prompt_tokens, args.max_new_tokens,
                               args.temperature, args.top_k, args.seed,
                               mode=args.mode, bars=args.bars)
    suffix = "" if args.mode == "continue" else f"_{args.mode}"
    out = args.out or Path(f"evals/scorecards/{Path(args.checkpoint).parent.name}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print(f"[eval] {card['n_generations']} generations -> {out}")
    for m, v in card["aggregate"].items():
        print(f"  {m:22s} mean {v['mean']:>8.3f}   median {v['median']:>8.3f}")


if __name__ == "__main__":
    main()
