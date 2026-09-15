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
- control adherence (`--mode control`, v4): ask for each Density / Poly /
  Range bucket in turn and measure which bucket came back
- the eval_v2 musical battery on the continuation only

Score one checkpoint:
    python -m midigenai.eval_checkpoint --checkpoint runs/pilot_baseline/ckpt_final.pt \\
        --tokenizer ~/midigenai_data/corpus_pilot/tokenizer.json \\
        --prompts evals/prompts --out evals/scorecards/pilot_baseline.json

Compare scorecards:
    python -m midigenai.eval_checkpoint --compare evals/scorecards/*.json

`--prompt-set evals/prompt_sets/heldout_v1.json` checks the prompt directory
against a frozen set (see make_prompt_set) before generating, and records
which prompts produced the numbers. Two scorecards are comparable exactly
when their `prompt_set.set_id` and `prompt_set.selected` agree; without it,
a checkpoint scored after someone re-ran `make_prompt_set build` is being
compared on different music.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from midigenai.make_prompt_set import file_sha256, load_manifest, verify_prompts

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
    # control mode only
    "control_accuracy", "control_adjacent", "control_accuracy_off_prompt",
    "control_effect",
]
# direction hints for the compare table: +1 higher is better, -1 lower, 0 neutral
DIRECTION = {"eos_rate": +1, "repetition_rate": -1, "repetition_drift": -1,
             "scale_consistency": +1, "prompt_coherence": +1,
             "downbeat_alignment": +1, "downbeat_delta": +1, "bar_offset_beats": -1,
             "accomp_bars_ok": +1, "accomp_pc_overlap": +1,
             "control_accuracy": +1, "control_adjacent": +1,
             "control_accuracy_off_prompt": +1, "control_effect": +1}


def _prompt_pool(prompts_dir: Path, prompt_set: dict | None) -> list[tuple[str, Path]]:
    """(sha256, path) for the prompt files, in a deterministic order.

    Without a frozen set the order is by filename, as before. With one, the
    directory is verified against the manifest and the order is by content
    hash, so renaming a file (or rebuilding the set on another machine, where
    `build` writes the same music under a different stem) cannot change which
    prompts the seed picks."""
    files = sorted(prompts_dir.glob("*.mid"))
    if prompt_set is None:
        return [("", f) for f in files]

    res = verify_prompts(prompts_dir, prompt_set)
    if res["missing"]:
        raise SystemExit(
            f"[eval] {prompts_dir} is missing {len(res['missing'])} of "
            f"{res['n_expected']} prompts in set {res['name']} "
            f"({res['set_id']}). Scoring a different set of prompts is not a "
            f"comparison. Rebuild with make_prompt_set build, or drop "
            f"--prompt-set to score whatever is on disk.")
    want = {f["sha256"] for f in prompt_set["files"]}
    pool = [(file_sha256(f), f) for f in files]
    return sorted((h, f) for h, f in pool if h in want)


def _stamp_prompt_set(card: dict, prompt_set: dict | None, rows: list[dict]) -> None:
    """Record which frozen set the numbers came from, and which of its
    members actually produced rows (files can be skipped: too few tokens,
    single-track in accompany mode, a decode failure)."""
    if prompt_set is None:
        return
    used = sorted({r["prompt_sha"] for r in rows if r.get("prompt_sha")})
    card["prompt_set"] = {
        "name": prompt_set.get("name"),
        "set_id": prompt_set.get("set_id"),
        "n_files": prompt_set.get("n"),
        "n_used": len(used),
        "selected": used,
    }


def evaluate_checkpoint(checkpoint: str, tokenizer: str | None, prompts_dir: Path,
                        n_prompts: int, gens_per_prompt: int, prompt_tokens: int,
                        max_new_tokens: int, temperature: float, top_k: int,
                        seed: int, mode: str = "continue", bars: int = 8,
                        pad_to_bar: bool = False,
                        prompt_set: dict | None = None) -> dict:
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
                                   bars, temperature, top_k, seed, checkpoint,
                                   prompt_set=prompt_set)
    if mode == "control":
        return _evaluate_control(gen, prompts_dir, n_prompts, gens_per_prompt,
                                 prompt_tokens, max_new_tokens, temperature,
                                 top_k, seed, checkpoint, prompt_set=prompt_set)

    pool = _prompt_pool(prompts_dir, prompt_set)
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n_prompts, len(pool)))

    def cont_score(prompt_ids, new_ids):
        return _continuation_score(gen, prompt_ids, new_ids)

    rows = []
    for sha, f in picked:
        score = Score(str(f))
        normalize_drums(score, f.name)
        ids = gen.tokenizer(score).ids
        if len(ids) < 16:
            continue
        if len(ids) > prompt_tokens:
            ids = _cut_prompt(gen, ids, prompt_tokens)
        # v4: the prompt gets the header the builder would have given this file
        header = gen.make_header(f) if gen.v4 else []
        if pad_to_bar and gen.v4:
            # end the prompt on a bar line so the answer should start on "1"
            ids = gen.close_bar(ids)
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
                "prompt": f.name, "prompt_sha": sha[:12], "gen": g,
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
    card = {
        "checkpoint": str(checkpoint),
        "n_generations": len(rows),
        "params": {"prompt_tokens": prompt_tokens, "max_new_tokens": max_new_tokens,
                   "temperature": temperature, "top_k": top_k, "seed": seed},
        "aggregate": agg,
        "rows": rows,
    }
    _stamp_prompt_set(card, prompt_set, rows)
    return card


def _cut_prompt(gen, ids: list[int], n: int) -> list[int]:
    """Truncate a prompt at a note boundary (before a Position / TimeShift /
    Bar token), never mid-note: a prompt ending in `Pitch_60` makes the
    model's first token a Velocity, which says nothing about the music."""
    vocab = gen.tokenizer.vocab
    boundary = {tid for name, tid in vocab.items()
                if name.startswith(("Position_", "TimeShift_", "Bar_", "Rest_"))}
    for i in range(min(n, len(ids)) - 1, 8, -1):
        if ids[i] in boundary:
            return ids[:i]
    return ids[:n]


def _continuation_score(gen, prompt_ids, new_ids):
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


def _rank_corr(xs: list[float], ys: list[float]) -> float:
    """Spearman: Pearson on tie-averaged ranks. Buckets are ordinal and there
    are three or four of them, so ties are the normal case, not an edge one."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def _evaluate_control(gen, prompts_dir, n_prompts, gens_per_prompt, prompt_tokens,
                      max_new_tokens, temperature, top_k, seed, checkpoint,
                      prompt_set: dict | None = None) -> dict:
    """v4 control-adherence scorecard: ask for a bucket, measure what arrives.

    For each prompt and each Density / Poly / Range bucket, the prompt's own
    header is rebuilt with that one family overridden and everything else
    left describing the file — the distribution training saw. The
    continuation is then bucketed by the same function the dataset builder
    used, so "asked for Density_3, got Density_3" is the same statement on
    both sides.

    Exact accuracy alone is gameable by inertia: a model that ignores the
    header still scores well whenever the requested bucket happens to match
    the prompt's own. So the card also reports accuracy on the requests that
    *differ* from the prompt's bucket, and the low-to-high effect (mean
    realized bucket when asking for the lowest vs the highest), which is zero
    for a model that is not listening.
    """
    from symusic import Score

    from midigenai.attributes import FAMILY_SIZES, realized_buckets
    from midigenai.tokenizer import normalize_drums

    if not gen.v4:
        raise SystemExit("[eval] --mode control needs a v4 (header) checkpoint")

    pool = _prompt_pool(prompts_dir, prompt_set)
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n_prompts, len(pool)))

    rows = []
    for sha, f in picked:
        score = Score(str(f))
        normalize_drums(score, f.name)
        ids = gen.tokenizer(score).ids
        if len(ids) < 16:
            continue
        if len(ids) > prompt_tokens:
            ids = _cut_prompt(gen, ids, prompt_tokens)
        prompt_score = gen.tokenizer.decode(list(ids))
        prompt_buckets = realized_buckets(prompt_score)
        for family, n_buckets in FAMILY_SIZES.items():
            for requested in range(n_buckets):
                header = gen.make_header(f, **{family: requested})
                for g in range(gens_per_prompt):
                    new_ids = list(gen.generate_ids(
                        [*header, *ids], max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k))
                    try:
                        cont, _grid, _cut = _continuation_score(gen, ids, new_ids)
                    except Exception:
                        continue
                    n_notes = sum(len(t.notes) for t in cont.tracks)
                    row = {"prompt": f.name, "prompt_sha": sha[:12], "gen": g,
                           "family": family, "requested": requested,
                           "prompt_bucket": prompt_buckets.get(family),
                           "gen_tokens": len(new_ids), "gen_notes": n_notes}
                    if n_notes >= 4:
                        got = realized_buckets(cont).get(family)
                        if got is not None:
                            row["realized"] = got
                            row["hit"] = 1.0 if got == requested else 0.0
                            row["adjacent"] = 1.0 if abs(got - requested) <= 1 else 0.0
                    rows.append(row)

    families = {}
    for family, n_buckets in FAMILY_SIZES.items():
        scored = [r for r in rows if r["family"] == family and "realized" in r]
        if not scored:
            continue
        confusion = [[0] * n_buckets for _ in range(n_buckets)]
        for r in scored:
            confusion[r["requested"]][r["realized"]] += 1
        by_request = [[r["realized"] for r in scored if r["requested"] == b]
                      for b in range(n_buckets)]
        means = [round(st.mean(v), 3) if v else None for v in by_request]
        off = [r for r in scored
               if r["prompt_bucket"] is not None and r["requested"] != r["prompt_bucket"]]
        lo, hi = means[0], means[-1]
        families[family] = {
            "n": len(scored),
            "empty": sum(1 for r in rows
                         if r["family"] == family and "realized" not in r),
            "accuracy": round(st.mean([r["hit"] for r in scored]), 4),
            "adjacent": round(st.mean([r["adjacent"] for r in scored]), 4),
            "accuracy_off_prompt": (round(st.mean([r["hit"] for r in off]), 4)
                                    if off else None),
            "n_off_prompt": len(off),
            "spearman": round(_rank_corr([r["requested"] for r in scored],
                                         [r["realized"] for r in scored]), 4),
            "effect_low_to_high": (round(hi - lo, 3)
                                   if lo is not None and hi is not None else None),
            "mean_realized_by_request": means,
            "confusion": confusion,
        }

    def _pooled(key):
        vals = [fam[key] for fam in families.values() if fam.get(key) is not None]
        return {"mean": round(st.mean(vals), 4),
                "median": round(st.median(vals), 4)} if vals else None

    agg = {}
    for card_key, fam_key in (("control_accuracy", "accuracy"),
                              ("control_adjacent", "adjacent"),
                              ("control_accuracy_off_prompt", "accuracy_off_prompt"),
                              ("control_effect", "effect_low_to_high")):
        v = _pooled(fam_key)
        if v:
            agg[card_key] = v
    for m in ("gen_tokens", "gen_notes"):
        vals = [r[m] for r in rows if m in r]
        if vals:
            agg[m] = {"mean": round(st.mean(vals), 4),
                      "median": round(st.median(vals), 4)}

    card = {"checkpoint": str(checkpoint), "mode": "control",
            "n_generations": len(rows),
            "params": {"prompt_tokens": prompt_tokens,
                       "max_new_tokens": max_new_tokens,
                       "temperature": temperature, "top_k": top_k, "seed": seed},
            "families": families, "aggregate": agg, "rows": rows}
    _stamp_prompt_set(card, prompt_set, rows)
    return card


def _evaluate_accompany(gen, prompts_dir, n_prompts, gens_per_prompt, bars,
                        temperature, top_k, seed, checkpoint,
                        prompt_set: dict | None = None) -> dict:
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

    pool = _prompt_pool(prompts_dir, prompt_set)
    rng = random.Random(seed)
    rng.shuffle(pool)
    rows = []
    for sha, f in pool:
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
            row = {"prompt": f.name, "prompt_sha": sha[:12], "gen": g,
                   "gen_tokens": len(new_ids),
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
    card = {"checkpoint": str(checkpoint), "mode": "accompany", "bars": bars,
            "n_generations": len(rows), "aggregate": agg, "rows": rows}
    _stamp_prompt_set(card, prompt_set, rows)
    return card


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
    p.add_argument("--mode", choices=["continue", "accompany", "control"],
                   default="continue",
                   help="accompany: v4 accompaniment scorecard; control: v4 "
                        "control-adherence scorecard (10 buckets x "
                        "--gens-per-prompt generations per prompt, so use a "
                        "smaller --n-prompts)")
    p.add_argument("--bars", type=int, default=8, help="accompany: window length")
    p.add_argument("--pad-to-bar", action="store_true",
                   help="continue (v4): pad the prompt to its bar line first")
    p.add_argument("--prompt-set", type=Path, default=None,
                   help="frozen prompt-set manifest (make_prompt_set freeze): "
                        "verify the prompt dir against it and stamp the "
                        "scorecard with what was scored")
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint required (or --compare)")
    prompt_set = load_manifest(args.prompt_set) if args.prompt_set else None
    card = evaluate_checkpoint(args.checkpoint, args.tokenizer, args.prompts,
                               args.n_prompts, args.gens_per_prompt,
                               args.prompt_tokens, args.max_new_tokens,
                               args.temperature, args.top_k, args.seed,
                               mode=args.mode, bars=args.bars,
                               pad_to_bar=args.pad_to_bar,
                               prompt_set=prompt_set)
    suffix = "" if args.mode == "continue" else f"_{args.mode}"
    if args.pad_to_bar:
        suffix += "_padbar"
    out = args.out or Path(f"evals/scorecards/{Path(args.checkpoint).parent.name}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-prompt rows name the files they were generated from, and those are
    # site uploads and corpus excerpts -- third-party music whose filenames
    # alone name artists and tracks. They stay local (gitignored) while the
    # aggregate, which is the thing worth reviewing, is committable.
    rows = card.pop("rows", None)
    if rows is not None:
        card["n_rows"] = len(rows)
        rows_path = out.with_suffix(".rows.json")
        rows_path.write_text(json.dumps(rows, indent=2))
    out.write_text(json.dumps(card, indent=2) + "\n")
    print(f"[eval] {card['n_generations']} generations -> {out}")
    if "prompt_set" in card:
        ps = card["prompt_set"]
        print(f"[eval] prompt set {ps['name']} ({ps['set_id']}): "
              f"{ps['n_used']}/{ps['n_files']} prompts scored")
    if rows is not None:
        print(f"[eval] per-prompt rows (local only) -> {rows_path}")
    for m, v in card["aggregate"].items():
        print(f"  {m:22s} mean {v['mean']:>8.3f}   median {v['median']:>8.3f}")
    for family, fam in card.get("families", {}).items():
        off = fam["accuracy_off_prompt"]
        print(f"  {family}: exact {fam['accuracy']:.2f}  within-1 "
              f"{fam['adjacent']:.2f}  off-prompt "
              f"{off if off is None else f'{off:.2f}'} (n={fam['n_off_prompt']})  "
              f"low->high {fam['effect_low_to_high']}  "
              f"realized by request {fam['mean_realized_by_request']}")


if __name__ == "__main__":
    main()
