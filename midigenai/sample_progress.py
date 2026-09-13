"""
Training-progress listening page: the same prompts continued by every
checkpoint of a run (and reference models), in one self-contained HTML file
with inline MIDI players. Re-run as checkpoints land; it only generates what
is missing.

    python -m midigenai.sample_progress --run runs/v4_full \\
        --tokenizer ~/midigenai_data/corpus_full_v4/tokenizer.json \\
        --ref v3 --prompts evals/prompts --n-prompts 6 --bars 8 \\
        --out evals/progress_v4

Layout: evals/progress_v4/<model>/<prompt>.mid (continuation only, prompt
tempo re-applied), evals/progress_v4/prompt/<prompt>.mid, index.html.
v4 checkpoints are prompted the production way (attribute header, prompt
closed to a bar line, stop after `--bars` bars); MIDILike references get the
same prompt and a matching token budget.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path


def _prompt_score(path: Path, tok, prompt_tokens: int, rng: random.Random):
    from symusic import Score

    from midigenai.tokenizer import normalize_drums
    sc = Score(str(path))
    normalize_drums(sc, path.name)
    ids = tok(sc).ids
    if len(ids) > prompt_tokens:
        start = rng.randrange(0, len(ids) - prompt_tokens)
        ids = ids[start:start + prompt_tokens]
    return tok.decode(list(ids))


def _continue(gen, prompt_path: Path, bars: int, max_new_tokens: int,
              temperature: float, top_k: int, seed: int):
    from symusic import Score, Tempo
    tempo = gen.detect_tempo(prompt_path)
    ids = gen.tokenizer(Score(str(prompt_path))).ids
    kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)
    if gen.v4:
        ids = [*gen.make_header(prompt_path), *gen.close_bar(ids)]
        kw["stop_after_bars"] = bars
    cut = gen.tokenizer.decode(list(ids)).end()
    new = list(gen.generate_ids(ids, **kw))
    full = gen.tokenizer.decode(list(ids) + new)
    for t in full.tracks:
        kept = [n for n in t.notes if n.start >= cut]
        for n in kept:
            n.start -= cut
        t.notes = kept
    full.tempos = [Tempo(time=0, qpm=tempo)]
    return full, len(new)


def build_page(out: Path, models: list[str], prompts: list[str], notes: dict) -> None:
    def data_uri(p: Path) -> str:
        return "data:audio/midi;base64," + base64.b64encode(p.read_bytes()).decode()
    rows = []
    for pr in prompts:
        cells = [f'<td class="name">{pr}<br><midi-player src="{data_uri(out / "prompt" / (pr + ".mid"))}" sound-font></midi-player></td>']
        for m in models:
            f = out / m / (pr + ".mid")
            if f.exists():
                cells.append(f'<td><midi-player src="{data_uri(f)}" sound-font></midi-player>'
                             f'<div class="meta">{notes.get(m, {}).get(pr, "")}</div></td>')
            else:
                cells.append('<td class="missing">—</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>midigenai — v4 training progress</title>
<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
<style>body{{font-family:-apple-system,system-ui,sans-serif;margin:20px;color:#222}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:6px;vertical-align:top;min-width:240px}}th{{background:#f4f4f4;font-size:13px}}
td.name{{font-size:12px;color:#555;min-width:200px}}midi-player{{width:230px;display:block}}.meta{{font-size:11px;color:#888}}
td.missing{{color:#bbb;text-align:center}}h1{{font-size:18px}}.sub{{color:#666;font-size:13px}}</style></head><body>
<h1>v4 training progress — same prompts, every checkpoint</h1>
<div class="sub">Each cell is the continuation only (prompt on the left, played separately). v4 checkpoints get the prompt's header and a closed bar and stop after N bars; reference models get the same prompt with a matching token budget. Columns left→right = earlier→later.</div>
<table><tr><th>prompt</th>{"".join(f"<th>{m}</th>" for m in models)}</tr>{"".join(rows)}</table></body></html>"""
    (out / "index.html").write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="dir with ckpt_*.pt of the run")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json for the run's checkpoints")
    p.add_argument("--ref", action="append", default=[], help="hub version(s) as reference columns, e.g. v3")
    p.add_argument("--prompts", type=Path, default=Path("evals/prompts"))
    p.add_argument("--n-prompts", type=int, default=6)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("evals/progress_v4"))
    args = p.parse_args()

    from midigenai.generate import Generator
    from midigenai.hub import load_from_hub
    from midigenai.tokenizer import load_tokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prompt").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(args.prompts.glob("*.mid"))
    picked = rng.sample(files, min(args.n_prompts, len(files)))
    prompts = [f.stem for f in picked]
    # fixed prompt excerpts, cut with the run's tokenizer once and reused
    tok = load_tokenizer(args.tokenizer)
    for f in picked:
        dst = args.out / "prompt" / f"{f.stem}.mid"
        if not dst.exists():
            _prompt_score(f, tok, args.prompt_tokens, random.Random(f.stem)).dump_midi(dst)

    notes_path = args.out / "notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    ckpts = sorted(args.run.glob("ckpt_*.pt"), key=lambda x: x.stem)
    columns = [(v, None) for v in args.ref] + [(c.stem, c) for c in ckpts]
    for name, ckpt in columns:
        d = args.out / name
        d.mkdir(exist_ok=True)
        todo = [pr for pr in prompts if not (d / f"{pr}.mid").exists()]
        if not todo:
            continue
        gen = load_from_hub(version=name) if ckpt is None else Generator(ckpt, args.tokenizer)
        for pr in todo:
            sc, n = _continue(gen, args.out / "prompt" / f"{pr}.mid", args.bars,
                              args.max_new_tokens, args.temperature, args.top_k, args.seed)
            sc.dump_midi(d / f"{pr}.mid")
            notes.setdefault(name, {})[pr] = f"{n} tokens, {sum(len(t.notes) for t in sc.tracks)} notes"
        print(f"[progress] {name}: {len(todo)} continuations")
        notes_path.write_text(json.dumps(notes, indent=1))
    build_page(args.out, [n for n, _ in columns], prompts, notes)
    print(f"[progress] page: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
