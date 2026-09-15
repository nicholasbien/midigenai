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
    prompt_tokens: int = 256
    max_new_tokens: int = 256
    temperature: float = 1.1
    top_k: int = 50
    max_cont_seconds: float = 8.0
    min_cont_notes: int = 8       # below this the take is silence, not a sample
    v4_close_bar: bool = False
    model_label: str = "v4"
    extra: dict = field(default_factory=dict)


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
        start = rng.randrange(0, len(prompt_ids) - cfg.prompt_tokens)
        prompt_ids = _slice_with_program(gen, prompt_ids, start, cfg.prompt_tokens)

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
                       "n_notes": kept}

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
        **cfg.extra,
    }
    return {"pair_id": pair_id, "meta": meta,
            "prompt": prompt_bytes,
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
        pair = make_pair(gen, pf, cfg, rng)
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
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
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

    cfg = PairConfig(prompt_tokens=a.prompt_tokens, max_new_tokens=a.max_new_tokens,
                     temperature=a.temperature, top_k=a.top_k, model_label=label_name)
    t0 = time.time()

    def on_pair(pair, i):
        write_pair(pair, pairs_dir)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"[pairs] {i}/{todo}  {el/i:.2f}s/pair  "
                  f"eta {(todo - i) * el / i / 60:.0f} min", flush=True)

    generate_pairs(gen, prompts, todo, cfg, seed=a.seed + have, on_pair=on_pair)
    el = time.time() - t0
    print(f"[pairs] done: {todo} pairs in {el/60:.1f} min ({el/todo:.2f}s/pair)")


if __name__ == "__main__":
    main()
