"""web/midigen-prompt.mjs builds prompts in a page: it must give the ids Python gives. Random scores go through
MidiTok (encode, decode), symusic (clip, bar lines) and Generator (postprocess, padding, infill prompts) here and
through the JS module under node, and the results must match exactly. Skipped without node."""
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from symusic import ControlChange, Note, Score, Tempo, TimeSignature, Track

from midigenai.attributes import content_tokens, header_for_score, instrument_tokens, sort_header, with_instruments
from midigenai.data.v4_docs import _window, bar_edges
from midigenai.generate import Generator, fit_to_context
from midigenai.infill_span import InfillError, plan_score, prompt_segments, splice
from midigenai.tokenizer import build_tokenizer, normalize_drums, save_tokenizer

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "tests" / "web" / "prompt-driver.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def tokenizer_params(scheme: str) -> tuple[dict, Path]:
    d = Path(tempfile.mkdtemp())
    save_tokenizer(build_tokenizer(scheme=scheme), d / "tokenizer.json")
    return json.loads((d / "tokenizer.json").read_text()), d / "tokenizer.json"


@pytest.fixture(scope="module", params=["v4", "v4-24"])
def tok(request):
    params, path = tokenizer_params(request.param)
    g = Generator(checkpoint_path=None, tokenizer_path=path, backend="remote", decode=lambda spec: [])
    return params, g


def node(cases: list[dict]) -> list:
    out = subprocess.run(["node", str(DRIVER)], input=json.dumps(cases), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def score_json(sc: Score) -> dict:
    return {
        "tpq": sc.ticks_per_quarter,
        "tracks": [{"name": t.name, "program": int(t.program), "is_drum": bool(t.is_drum),
                    "notes": [{"time": n.time, "duration": n.duration, "pitch": n.pitch, "velocity": n.velocity} for n in t.notes],
                    "controls": [{"time": c.time, "number": c.number, "value": c.value} for c in t.controls]}
                   for t in sc.tracks],
        "time_signatures": [{"time": x.time, "numerator": x.numerator, "denominator": x.denominator} for x in sc.time_signatures],
        "tempos": [{"time": x.time, "qpm": x.qpm} for x in sc.tempos],
    }


def notes_of(sc: dict) -> list:
    """What decoding is compared on: each track's program, kit flag and notes, in order, and the meter changes."""
    return [[[t["program"], t["is_drum"], [[n["time"], n["duration"], n["pitch"], n["velocity"]] for n in t["notes"]]]
             for t in sc["tracks"]], [[x["time"], x["numerator"], x["denominator"]] for x in sc["time_signatures"]]]


def random_score(rng: random.Random, tpq: int = 480, meter: tuple | None = (4, 4), bars: int | None = None) -> Score:
    sc = Score(tpq)
    if meter:
        sc.time_signatures.append(TimeSignature(0, *meter))
    sc.tempos.append(Tempo(0, 120))
    bars = bars or rng.randint(1, 10)
    bar = tpq * 4 * (meter[0] / meter[1] if meter else 1)
    grids = [tpq, tpq // 2, tpq // 4, tpq // 3, tpq // 8, 1, tpq // 6, 7]
    programs = [0, 33, 81, 89, 4, 48, 38] + [rng.randrange(128) for _ in range(2)]
    for _ in range(rng.randint(1, 5)):
        drum = rng.random() < 0.3
        prog = 0 if drum else rng.choice(programs)
        t = Track(name=rng.choice(["drums" if drum else f"prog{prog}", "keys", "Kick", "tom fill", "lead", ""]), program=prog, is_drum=drum)
        grid = max(1, rng.choice(grids))
        for _ in range(rng.randint(0, 40)):
            start = int(rng.randrange(int(bars * bar)) // grid * grid + (rng.randrange(grid) if rng.random() < 0.1 else 0))
            dur = rng.choice([grid, 2 * grid, tpq, tpq * 4, 1, rng.randrange(1, tpq * 8)])
            pitch = rng.randrange(20, 100) if drum else rng.randrange(0, 128)
            t.notes.append(Note(start, dur, pitch, rng.randrange(1, 128)))
        if rng.random() < 0.3:
            t.notes.sort()
        if rng.random() < 0.2:
            t.controls.append(ControlChange(int(bars * bar) - tpq // 2, 7, 100))
        sc.tracks.append(t)
    return sc


def test_vocab_matches(tok):
    params, g = tok
    assert node([{"op": "vocab", "tokenizer": params}])[0]["ok"] == [t for t, _ in sorted(g.tokenizer.vocab.items(), key=lambda kv: kv[1])]


def test_encode_matches_miditok(tok):
    params, g = tok
    rng = random.Random(7)
    scores = []
    for i in range(300):
        meter = rng.choice([(4, 4), (4, 4), None, (3, 4), (6, 8), (7, 8), (2, 2)])
        scores.append(random_score(rng, tpq=rng.choice([480, 480, 220, 96, 8]), meter=meter))
    want = [list(g.tokenizer(sc).ids) for sc in scores]
    got = node([{"op": "encode", "tokenizer": params, "score": score_json(sc)} for sc in scores])
    for i, (w, r) in enumerate(zip(want, got)):
        assert r.get("ok") == w, f"score {i}: {r.get('error') or 'ids differ'}"


def test_decode_matches_miditok(tok):
    params, g = tok
    rng = random.Random(11)
    voc = g.tokenizer.vocab
    names = list(voc)
    seqs = []
    for i in range(200):
        ids = list(g.tokenizer(random_score(rng, meter=rng.choice([(4, 4), (3, 4), (6, 8)]))).ids)
        # what a model writes: tokens in odd places, meter changes, a sequence cut mid-note
        for _ in range(rng.randint(0, 6)):
            pos = rng.randrange(len(ids) + 1)
            ids.insert(pos, voc[rng.choice(names[:5] + [n for n in names if n.startswith(("TimeSig_", "Program_", "Bar_", "Position_", "Pitch"))])])
        if rng.random() < 0.3 and ids:
            ids = ids[:rng.randrange(1, len(ids) + 1)]
        seqs.append(ids)
    got = node([{"op": "decode", "tokenizer": params, "ids": ids} for ids in seqs])
    for i, (ids, r) in enumerate(zip(seqs, got)):
        assert "ok" in r, f"seq {i}: {r.get('error')}"
        assert notes_of(r["ok"]) == notes_of(score_json(g.tokenizer.decode(ids))), f"seq {i} decodes differently"


def test_postprocess_and_padding_match(tok):
    params, g = tok
    rng = random.Random(3)
    special = [g.bar_id, g.bar_id, g.bar_id, *sorted(g.timesig_ids)[:3], g.eos_id, g.sp.sep, g.bos_id]
    note = [v for k, v in g.tokenizer.vocab.items() if k.startswith(("Pitch_", "Position_", "Velocity_"))]
    cases, want = [], []
    for _ in range(400):
        prompt = [g.bos_id] + [rng.choice(special[:6] + note) for _ in range(rng.randint(0, 12))]
        raw = [rng.choice(special + note * 1) for _ in range(rng.randint(0, 30))]
        stop = rng.choice([None, 0, 1, 2, 4])
        trim = rng.random() < 0.6
        cases.append({"op": "postprocess", "tokenizer": params, "prompt": prompt, "raw": raw, "stop": stop, "trim": trim})
        want.append(list(g.postprocess(prompt, raw, stop_after_bars=stop, trim_leading_bars=trim)))
        cases.append({"op": "ends_on_bar_line", "tokenizer": params, "ids": prompt})
        want.append(g.ends_on_bar_line(prompt))
        n = g.count_bars(prompt) + rng.randint(0, 3)
        cases.append({"op": "pad", "tokenizer": params, "ids": prompt, "bars": n})
        want.append(g.pad_to_bars(prompt, n))
        ln, mx, sq = rng.randint(0, 900), rng.randint(0, 900), rng.choice([0, 64, 512, 1024])
        cases.append({"op": "fit", "ids": list(range(ln)), "max_new": mx, "max_seq": sq})
        f = fit_to_context(list(range(ln)), mx, sq)
        want.append([f[0], f[1]])
    for i, (w, r) in enumerate(zip(want, node(cases))):
        assert r.get("ok") == w, f"case {i} ({cases[i]['op']}): {r.get('error') or r.get('ok')} != {w}"


def test_bars_windows_and_headers_match():
    rng = random.Random(5)
    cases, want = [], []
    for _ in range(150):
        sc = random_score(rng, meter=rng.choice([(4, 4), None, (3, 4)]))
        cases.append({"op": "bar_edges", "score": score_json(sc)})
        edges = bar_edges(sc)
        want.append(edges)
        a = rng.randrange(len(edges) - 1)
        b = rng.randrange(a + 1, len(edges))
        cases.append({"op": "window", "score": score_json(sc), "start": edges[a], "end": edges[b]})
        want.append(notes_of(score_json(_window(sc, edges[a], edges[b]))))
        tempo = rng.choice([None, 60, 90, 128.5, 174, 200])
        cases.append({"op": "header", "score": score_json(sc), "tempo": tempo})
        want.append(header_for_score(sc, tempo=tempo) if tempo else instrument_tokens(sc) + content_tokens(sc))
        hint = rng.choice(["upload.mid", "drums.mid", "my beat.mid"])
        cases.append({"op": "normalize", "score": score_json(sc), "hint": hint})
        c = sc.copy(); normalize_drums(c, hint)
        want.append([bool(t.is_drum) for t in c.tracks])
    fams = ["Piano", "Bass", "Drums", "SynthPad", "Strings", "Brass", "Reed", "Organ"]
    for _ in range(50):
        f = [rng.choice(fams) for _ in range(rng.randint(0, 8))]
        names = rng.sample(["Tempo_3", "Genre_jazz", "Density_1", "Inst_Piano", "Poly_2", "Range_0", "Task_accomp"], rng.randint(0, 5))
        cases.append({"op": "with_instruments", "names": names, "families": f})
        want.append(with_instruments(names, f))
        cases.append({"op": "sort_header", "names": names})
        want.append(sort_header(names))
    got = node(cases)
    for i, (w, r) in enumerate(zip(want, got)):
        if cases[i]["op"] == "window":
            assert notes_of(r["ok"]) == w, f"case {i}: window differs"
        else:
            assert r.get("ok") == w, f"case {i} ({cases[i]['op']}): {r.get('error') or r.get('ok')} != {w}"


def test_infill_prompts_and_splice_match(tok):
    params, g = tok
    rng = random.Random(9)
    cases, want = [], []
    while len(cases) < 120:
        sc = random_score(rng, meter=(4, 4), bars=rng.randint(3, 20))
        normalize_drums(sc, "upload.mid")
        n = len(bar_edges(sc)) - 1
        bars = rng.randint(1, 4)
        start = rng.randint(0, max(0, n - bars))
        tempo = rng.choice([90, 120, 140])
        answer = list(g.tokenizer(random_score(rng, bars=bars)).ids)
        try:
            p = plan_score(sc, start, bars)
        except InfillError:
            cases.append({"op": "infill", "tokenizer": params, "score": score_json(sc), "start": start, "bars": bars, "tempo": tempo})
            want.append("InfillError")
            continue
        prefix, suffix = prompt_segments(g, p)
        header = g.sp.header_ids_for(g.tokenizer, header_for_score(p.window(), tempo=tempo))
        from midigenai.sequence_format import infill_prompt
        cases.append({"op": "infill", "tokenizer": params, "score": score_json(sc), "start": start, "bars": bars,
                      "tempo": tempo, "answer": answer})
        want.append({"prefix": prefix, "suffix": suffix, "header": header, "ctx": [p.ctx_start, p.ctx_end],
                     "prompt": infill_prompt(g.sp, header, prefix, suffix),
                     "spliced": notes_of(score_json(splice(p, g.tokenizer.decode(answer))))})
    for i, (w, r) in enumerate(zip(want, node(cases))):
        if w == "InfillError":
            assert "InfillError" in (r.get("error") or ""), f"case {i}: expected an InfillError, got {r}"
            continue
        assert "ok" in r, f"case {i}: {r.get('error')}"
        got = dict(r["ok"])
        got["spliced"] = notes_of(got["spliced"])
        assert got == w, f"case {i}: infill differs ({[k for k in w if w[k] != got.get(k)]})"
