"""Headless A/B pair generation — the same construction `label_app` serves,
without a human in the loop.

For RLAIF the pairs must be *on-policy*: both sides sampled from the very
checkpoint GRPO will optimise, at the temperature it will be sampled at. A
reward model fit on v3-vs-v4 pairs learns to tell two checkpoints apart,
which is not the question GRPO asks — GRPO asks which of eight samples from
*this* policy is better.

The pair layout matches `label_app` exactly (`<id>_prompt.mid`, `<id>_a.mid`,
`<id>_b.mid`, `<id>.json`) so `llm_judge` and `reward_align` read these pairs
with no special case.
"""

from __future__ import annotations

import datetime
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PairConfig:
    mode: str = "continue"         # "continue" | "accompany"
    bars: int = 16                 # accompany: window length (training default)
    single_target_frac: float = 0.6   # v4_docs.DocBuilder default
    prompt_tokens: int = 256
    min_prompt_bars: float = 2.0   # a window covering less is not a phrase
    max_new_tokens: int = 256
    temperature: float = 1.1
    top_k: int = 50
    max_cont_seconds: float = 8.0
    min_cont_notes: int = 8       # below this the take is silence, not a sample
    v4_close_bar: bool = False
    model_label: str = "v4"
    extra: dict = field(default_factory=dict)


def _accomp_prompt(gen, header, cond_ids, bars) -> list[int]:
    from midigenai.sequence_format import accompaniment_prompt
    return accompaniment_prompt(gen.sp, list(header), gen.pad_to_bars(list(cond_ids), bars))


def _dumps_midi(score) -> bytes:
    """symusic writes to a path; newer builds also dump to bytes."""
    try:
        return bytes(score.dumps_midi())
    except Exception:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp = f.name
        try:
            score.dump_midi(tmp)
            return Path(tmp).read_bytes()
        finally:
            os.unlink(tmp)


_INV_CACHE: dict[int, dict[int, str]] = {}


def _slice_with_program(gen, ids: list[int], start: int, length: int) -> list[int]:
    """Window of `length` tokens, with the instrument sounding at `start`
    restored in front of it.

    `Program_*` tokens are stateful: everything after one keeps that program
    until the next, so a window cut after a `Program_-1` decodes a drum kit
    as piano. Same rule as `label_app._slice_with_program`.
    """
    inv = _INV_CACHE.get(id(gen.tokenizer))
    if inv is None:
        inv = {v: k for k, v in gen.tokenizer.vocab.items()}
        _INV_CACHE[id(gen.tokenizer)] = inv
    prefix: list[int] = []
    for j in range(start - 1, -1, -1):
        if inv.get(ids[j], "").startswith("Program_"):
            prefix = [ids[j]]
            break
    return prefix + list(ids[start:start + length])


def _bar_aligned_start(gen, ids: list[int], length: int, rng: random.Random) -> int:
    """A window start that lands on a Bar token when the tokenizer has them
    (v4), so the prompt opens on a downbeat; a plain random offset otherwise."""
    bar_id = getattr(gen, "bar_id", None)
    limit = len(ids) - length
    if bar_id is None:
        return rng.randrange(0, limit)
    starts = [i for i, t in enumerate(ids) if t == bar_id and i <= limit]
    return rng.choice(starts) if starts else rng.randrange(0, limit)


def _bars_spanned(gen, ids: list[int]) -> float:
    """How much music a token window covers, in bars."""
    try:
        sc = gen.tokenizer.decode(list(ids))
    except Exception:
        return 0.0
    notes = [n for t in sc.tracks for n in t.notes]
    if not notes:
        return 0.0
    tpq = max(sc.ticks_per_quarter, 1)
    return (max(n.start + n.duration for n in notes) - min(n.start for n in notes)) / tpq / 4


def make_pair(gen, prompt_file: Path, cfg: PairConfig,
              rng: random.Random) -> dict | None:
    """One pair: a prompt slice and two independent continuations of it.

    Returns None when the prompt or either take is unusable, so a caller can
    simply skip it — a silent take is not a preference, it is a defect, and
    feeding it to the judge spends a call to learn nothing.
    """
    from symusic import Score, Tempo
    from midigenai.tokenizer import normalize_drums

    try:
        prompt_score = Score(str(prompt_file))
    except Exception:
        return None
    normalize_drums(prompt_score, prompt_file.name)
    prompt_ids = gen.tokenizer(prompt_score).ids
    if len(prompt_ids) < 32:
        return None
    if len(prompt_ids) > cfg.prompt_tokens:
        # A window cut at a random token starts mid-bar, mid-phrase, and in
        # dense material covers almost nothing: whole-arrangement Ableton
        # exports (median 8,293 notes) gave 256-token prompts of 0.2-1.8
        # bars, and the labeler flagged them as bad prompts 10 times in 49
        # votes. Start on a bar line, and refuse a window that spans less
        # than `min_prompt_bars` of music.
        start = _bar_aligned_start(gen, prompt_ids, cfg.prompt_tokens, rng)
        prompt_ids = _slice_with_program(gen, prompt_ids, start, cfg.prompt_tokens)
        if _bars_spanned(gen, prompt_ids) < cfg.min_prompt_bars:
            return None

    tempo = gen.detect_tempo(prompt_file)
    pair_id = f"{datetime.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"

    prompt_out = gen.tokenizer.decode(list(prompt_ids))
    prompt_out.tempos = [Tempo(time=0, qpm=tempo)]
    prompt_bytes = _dumps_midi(prompt_out)

    # v4 reads its own attribute header and re-tokenises the written prompt,
    # exactly as production prompts it
    if getattr(gen, "v4", False):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(prompt_bytes)
            tmp = f.name
        try:
            ids = gen.tokenizer(Score(tmp)).ids
            if cfg.v4_close_bar:
                ids = gen.close_bar(ids)
            ids = [*gen.make_header(Path(tmp)), *ids]
        finally:
            os.unlink(tmp)
    else:
        ids = list(prompt_ids)

    sides = {}
    for name in ("a", "b"):
        seed = rng.randrange(1 << 30)
        new_ids = list(gen.generate_ids(
            list(ids), max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, top_k=cfg.top_k, seed=seed))
        if not new_ids:
            return None
        full = gen.tokenizer.decode(list(ids) + new_ids)
        tpq = max(full.ticks_per_quarter, 1)
        cut_tick = gen.tokenizer.decode(list(ids)).end()
        cap = cut_tick + int(cfg.max_cont_seconds * tempo / 60.0 * tpq)
        cont = full.copy()
        kept = 0
        for track, ct in zip(full.tracks, cont.tracks):
            ct.notes = [n for n in track.notes if cut_tick <= n.start < cap]
            for n in ct.notes:
                n.start -= cut_tick
            kept += len(ct.notes)
        if kept < cfg.min_cont_notes:
            return None
        cont.tempos = [Tempo(time=0, qpm=tempo)]
        sides[name] = {"bytes": _dumps_midi(cont), "seed": seed, "n_new": len(new_ids),
                       "n_notes": kept, "ids": new_ids}

    meta = {
        "pair_id": pair_id,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt_file": prompt_file.name,
        "model_a": cfg.model_label, "model_b": cfg.model_label,
        "cross_model": False,
        "on_policy": True,
        "tempo_bpm": tempo,
        "temperature": cfg.temperature, "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens,
        "seed_a": sides["a"]["seed"], "seed_b": sides["b"]["seed"],
        "n_notes_a": sides["a"]["n_notes"], "n_notes_b": sides["b"]["n_notes"],
        # what the model was actually prompted with (header included), and
        # what it sampled. reward_probe reads both — its features are a
        # forward pass over prompt + continuation — and a meta without
        # prompt_ids made it KeyError on the first 202M pair.
        "prompt_ids": list(ids),
        "cont_a_ids": sides["a"]["ids"], "cont_b_ids": sides["b"]["ids"],
        **cfg.extra,
    }
    return {"pair_id": pair_id, "meta": meta,
            "prompt": prompt_bytes,
            "a": sides["a"]["bytes"], "b": sides["b"]["bytes"]}


_DRUM_IDS: dict[int, list[int]] = {}


def _drum_token_ids(tokenizer) -> list[int]:
    """Every token that can only produce percussion: PitchDrum_* and the drum
    program (Program_-1). 63 ids on the v4 tokenizer."""
    key = id(tokenizer)
    if key not in _DRUM_IDS:
        _DRUM_IDS[key] = [i for name, i in tokenizer.vocab.items()
                          if name.startswith("PitchDrum_") or name == "Program_-1"]
    return _DRUM_IDS[key]


def sample_accompany_window(score, bars: int, rng: random.Random,
                            single_target_frac: float = 0.6, name: str = ""):
    """The training sampler for accompaniment windows, shared by pair
    generation and RL so both draw the same distribution. Returns
    (cond, tgt, header_src, kind) as Scores, or None."""
    from midigenai.data.v4_docs import (MIN_SEGMENT_NOTES, _n_notes, _subscore,
                                        _tracks_in_window, _window, bar_edges,
                                        split_hands, trim_leading)
    from midigenai.tokenizer import normalize_drums
    normalize_drums(score, name)
    score = trim_leading(score)
    edges = bar_edges(score)
    n_bars = len(edges) - 1
    if n_bars < bars:
        return None
    b0 = rng.randrange(0, n_bars - bars + 1)
    s_tick, e_tick = edges[b0], edges[b0 + bars]
    win = _window(score, s_tick, e_tick)
    live = _tracks_in_window(score, s_tick, e_tick, MIN_SEGMENT_NOTES)
    if len(live) >= 2:
        rng.shuffle(live)
        n_cond = 1 if len(live) == 2 or rng.random() < 0.7 else 2
        cond_idx, rest = live[:n_cond], live[n_cond:]
        tgt_idx = [rng.choice(rest)] if rng.random() < single_target_frac else rest
        cond, tgt = _subscore(win, cond_idx), _subscore(win, tgt_idx)
        header_src, kind = _subscore(win, cond_idx + tgt_idx), "tracks"
    else:
        solo = [i for i, t in enumerate(score.tracks) if not t.is_drum and len(t.notes)]
        if len(solo) != 1 or any(t.is_drum and len(t.notes) for t in score.tracks):
            return None
        hands = split_hands(win)
        if hands is None:
            return None
        low, high = hands
        cond, tgt = (low, high) if rng.random() < 0.5 else (high, low)
        header_src, kind = win, "hands"
    if _n_notes(cond) < MIN_SEGMENT_NOTES or _n_notes(tgt) < MIN_SEGMENT_NOTES:
        return None
    return cond, tgt, header_src, kind


def make_accompany_pair(gen, prompt_file: Path, cfg: PairConfig,
                        rng: random.Random) -> dict | None:
    """One accompaniment pair, sampled the way TRAINING samples them.

    This deliberately mirrors `v4_docs.DocBuilder`'s accompaniment windows
    rather than doing something reasonable-looking of its own:

      * the condition is a RANDOM shuffle of the live tracks, not the busiest
        one. Picking the busiest made 48% of conditions drum tracks, because
        hi-hats win on note count — a distribution the model is not trained
        on and a question ("what accompanies this hi-hat") with no harmonic
        grounds to answer it.
      * 1 condition track 70% of the time, 2 the rest (always 1 when only two
        tracks are live).
      * the target is ONE random remaining track 60% of the time, otherwise
        all of them.
      * the header names condition + target only, so it reads as "add a bass"
        rather than "add everything this file had".
      * solo keyboard files go through a left/right hand split, which is how
        a piano-only corpus contributes accompaniment data at all.

    Labelling a different distribution from the one trained on would measure
    a model nobody built, and would hide sampling bugs instead of exposing
    them.
    """
    from symusic import Score, Tempo
    from midigenai.attributes import header_for_score

    if not getattr(gen, "v4", False):
        return None
    try:
        score = Score(str(prompt_file))
    except Exception:
        return None
    sampled = sample_accompany_window(score, cfg.bars, rng, cfg.single_target_frac,
                                      prompt_file.name)
    if sampled is None:
        return None
    cond, tgt, header_src, kind = sampled
    cond_ids = gen.tokenizer(cond).ids
    # Unless a kit was asked for, keep one out. With the header naming
    # condition + target the v4 model still puts an uninvited drum kit in
    # 30-45% of the notes of a pitched target — its prior for "more than
    # one instrument" is "there are drums" — so an "add a bass" pair is
    # really "add a bass and a kit", and a vote on it is partly a vote about
    # the kit. Banning the drum tokens removed that and roughly doubled the
    # requested family's share (bass 21% -> 36% of notes). Production's
    # /api/accompany applies the same ban, which keeps these pairs on-policy.
    ban = [i for i in (gen.sp.sep, gen.sp.mask) if i is not None]
    target_has_drums = any(t.is_drum for t in tgt.tracks)
    cond_has_drums = any(t.is_drum for t in cond.tracks)
    if not target_has_drums and not cond_has_drums:
        ban += _drum_token_ids(gen.tokenizer)
    have = gen.count_bars(cond_ids)
    bars = max(cfg.bars, have)
    if have > cfg.bars * 2:
        return None
    header = gen.sp.header_ids_for(gen.tokenizer, header_for_score(header_src))
    tempo = gen.detect_tempo(prompt_file)
    pair_id = f"{datetime.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"

    cond_score = gen.tokenizer.decode(list(cond_ids))
    cond_score.tempos = [Tempo(time=0, qpm=tempo)]
    cond_end = max((n.start + n.duration
                    for t in cond_score.tracks for n in t.notes), default=0)

    sides = {}
    for name in ("a", "b"):
        seed = rng.randrange(1 << 30)
        new_ids = list(gen.accompany(cond_ids, bars, header=header,
                                     temperature=cfg.temperature,
                                     top_k=cfg.top_k, seed=seed, ban_ids=ban))
        if not new_ids:
            return None
        try:
            acc = gen.tokenizer.decode(new_ids)
        except Exception:
            return None
        # clamp to the span being accompanied: a tail playing alone after the
        # condition stops is not accompaniment, and it biases a listening test
        for t in acc.tracks:
            kept = [n for n in t.notes if n.start < cond_end]
            for n in kept:
                n.duration = min(n.duration, max(1, cond_end - n.start))
            t.notes = kept
        n_notes = sum(len(t.notes) for t in acc.tracks)
        if n_notes < cfg.min_cont_notes:
            return None
        mix = cond_score.copy()
        for t in acc.tracks:
            mix.tracks.append(t)
        mix.tempos = [Tempo(time=0, qpm=tempo)]
        sides[name] = {"bytes": _dumps_midi(mix), "seed": seed, "ids": new_ids,
                       "n_notes": n_notes,
                       "bars_ok": gen.count_bars(new_ids) == bars}

    meta = {
        "pair_id": pair_id,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt_file": prompt_file.name,
        "mode": "accompany", "bars": bars, "bars_requested": cfg.bars,
        "sampling": kind, "drums_banned": not target_has_drums and not cond_has_drums,
        "n_cond_tracks": len(cond.tracks), "n_target_tracks": len(tgt.tracks),
        "cond_is_drums": any(t.is_drum for t in cond.tracks),
        "model_a": cfg.model_label, "model_b": cfg.model_label,
        "cross_model": False, "on_policy": True,
        "tempo_bpm": tempo,
        "temperature": cfg.temperature, "top_k": cfg.top_k,
        # exactly what the model was prompted with — BOS, Task_accomp, header,
        # the condition padded to the window, SEP — so a probe reward sees the
        # same layout when it is fitted on these pairs and when GRPO scores
        # samples from this same prompt
        "prompt_ids": list(_accomp_prompt(gen, header, cond_ids, bars)),
        "cond_ids": cond_ids,
        "cont_a_ids": sides["a"]["ids"], "cont_b_ids": sides["b"]["ids"],
        "seed_a": sides["a"]["seed"], "seed_b": sides["b"]["seed"],
        "n_notes_a": sides["a"]["n_notes"], "n_notes_b": sides["b"]["n_notes"],
        "bars_ok_a": sides["a"]["bars_ok"], "bars_ok_b": sides["b"]["bars_ok"],
        **cfg.extra,
    }
    return {"pair_id": pair_id, "meta": meta,
            "prompt": _dumps_midi(cond_score),
            "a": sides["a"]["bytes"], "b": sides["b"]["bytes"]}


def write_pair(pair: dict, pairs_dir: Path) -> None:
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pid = pair["pair_id"]
    (pairs_dir / f"{pid}_prompt.mid").write_bytes(pair["prompt"])
    (pairs_dir / f"{pid}_a.mid").write_bytes(pair["a"])
    (pairs_dir / f"{pid}_b.mid").write_bytes(pair["b"])
    (pairs_dir / f"{pid}.json").write_text(json.dumps(pair["meta"]))


def generate_pairs(gen, prompt_files: list[Path], n: int, cfg: PairConfig,
                   seed: int = 0, on_pair=None) -> list[dict]:
    """`n` pairs, cycling the prompt list; unusable prompts are skipped."""
    rng = random.Random(seed)
    order = list(prompt_files)
    rng.shuffle(order)
    out: list[dict] = []
    i = attempts = 0
    while len(out) < n and attempts < n * 4:
        attempts += 1
        pf = order[i % len(order)]
        i += 1
        pair = (make_accompany_pair(gen, pf, cfg, rng) if cfg.mode == "accompany"
                else make_pair(gen, pf, cfg, rng))
        if pair is None:
            continue
        out.append(pair)
        if on_pair:
            on_pair(pair, len(out))
    return out


def main() -> None:
    import argparse
    import time

    p = argparse.ArgumentParser(description="Generate on-policy A/B pairs.")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="pairs/ is created here")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--version", default="v4")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--min-prompt-bars", type=float, default=2.0,
                   help="skip windows that span less music than this; dense sources "
                        "need a bigger --prompt-tokens to clear it")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None,
                   help="model name written to each pair's meta; defaults to the "
                        "hub version or the checkpoint's stem, which is ckpt_final "
                        "for every run and says nothing about which one")
    p.add_argument("--mode", choices=("continue", "accompany"), default="continue")
    p.add_argument("--bars", type=int, default=16,
                   help="accompany: window length (training uses 16)")
    a = p.parse_args()

    if a.checkpoint:
        from midigenai.generate import Generator
        gen = Generator(checkpoint_path=a.checkpoint, tokenizer_path=a.tokenizer)
        label_name = a.checkpoint.stem
    else:
        from midigenai.hub import load_from_hub
        gen = load_from_hub(version=a.version)
        label_name = a.version

    prompts = sorted(Path(a.prompts).glob("*.mid"))
    if not prompts:
        raise SystemExit(f"no .mid files in {a.prompts}")
    pairs_dir = Path(a.out) / "pairs"
    have = len(list(pairs_dir.glob("*_prompt.mid"))) if pairs_dir.exists() else 0
    todo = max(0, a.n - have)
    print(f"[pairs] {len(prompts)} prompts, {have} pairs already on disk, "
          f"generating {todo} more into {pairs_dir}")
    if not todo:
        return

    cfg = PairConfig(mode=a.mode, bars=a.bars, prompt_tokens=a.prompt_tokens,
                     min_prompt_bars=a.min_prompt_bars,
                     max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                     top_k=a.top_k, model_label=a.label or label_name)
    t0 = time.time()

    def on_pair(pair, i):
        write_pair(pair, pairs_dir)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"[pairs] {i}/{todo}  {el/i:.2f}s/pair  "
                  f"eta {(todo - i) * el / i / 60:.0f} min", flush=True)

    made = generate_pairs(gen, prompts, todo, cfg, seed=a.seed + have, on_pair=on_pair)
    el = time.time() - t0
    n = len(made)
    print(f"[pairs] done: {n} pairs in {el/60:.1f} min "
          f"({el/max(n,1):.2f}s/pair)")
    if n < todo:
        # Accompaniment needs a prompt with at least two live tracks, and a
        # solo-piano corpus has none — reporting the request rather than the
        # result hid that the first run produced one pair out of three.
        print(f"[pairs] WARNING: asked for {todo}, got {n}. Prompts are "
              f"skipped when they are too short, or (accompany mode) have "
              f"fewer than two tracks with notes.")


if __name__ == "__main__":
    main()
