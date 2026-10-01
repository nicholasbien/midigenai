"""Span infill as served: "keep these bars, redo bars i..j".

Pins the span limits (the trained shape), the context window, the splice
(everything outside the span comes back untouched), and the end-to-end path
on a tiny random model. The Flask route is checked for its 400s with Modal
stubbed out.
"""
import io
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai import infill_span
from midigenai.infill_span import InfillError, context_window, plan, splice

TPQ = 480
BAR = 4 * TPQ


def _song(n_bars=8, tracks=((0, False, 60), (33, False, 40))):
    """One note per beat on every track, pitch = base + bar index, so a
    note's bar is readable from its pitch."""
    s = Score(TPQ)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    for program, drum, base in tracks:
        t = Track(name=f"p{program}", program=program, is_drum=drum)
        for b in range(n_bars):
            for beat in range(4):
                t.notes.append(Note(b * BAR + beat * TPQ, TPQ // 2, base + b, 80))
        s.tracks.append(t)
    return s


def _bytes(score) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        score.dump_midi(f.name)
        return Path(f.name).read_bytes()


@pytest.mark.parametrize("n,start,bars", [
    (8, 1, 1), (8, 3, 2), (8, 5, 2), (8, 1, 4), (40, 1, 2), (40, 20, 4), (40, 37, 2), (4, 1, 2),
])
def test_context_window_holds_the_span_with_a_bar_either_side(n, start, bars):
    b0, b1 = context_window(n, start, bars)
    assert 0 <= b0 < start and start + bars < b1 <= n
    assert b1 - b0 == min(infill_span.CONTEXT_BARS, n)


@pytest.mark.parametrize("start,bars", [(0, 2), (6, 2), (7, 1), (3, 0), (1, 5), (-1, 1)])
def test_untrained_spans_are_refused_by_name(start, bars):
    with pytest.raises(InfillError):
        plan(_bytes(_song(8)), start, bars)


def test_plan_keeps_file_bar_numbers():
    """Leading silence is not trimmed: bar numbers are the file's own."""
    s = _song(8)
    for t in s.tracks:
        for n in t.notes:
            n.time += BAR           # one empty bar up front
    p = plan(_bytes(s), 3, 2)
    assert p.span_ticks == (3 * BAR, 5 * BAR)
    assert p.prefix_bars == 3 - p.ctx_start and p.suffix_bars == p.ctx_end - 5


def test_splice_replaces_only_the_span():
    s = _song(8)
    p = plan(_bytes(s), 3, 2)
    ans = Score(16)                 # the tokenizer's rate, not the file's
    ans.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    t.notes.append(Note(0, 8, 100, 90))            # bar 0 of the span
    t.notes.append(Note(4 * 16, 8, 101, 90))       # bar 1 of the span
    t.notes.append(Note(8 * 16, 8, 102, 90))       # past the span: dropped
    ans.tracks.append(t)
    drums = Track(program=0, is_drum=True)
    drums.notes.append(Note(0, 4, 36, 90))
    ans.tracks.append(drums)

    out = splice(p, ans)
    lo, hi = p.span_ticks
    piano = out.tracks[0]
    assert sorted(n.pitch for n in piano.notes if lo <= n.time < hi) == [100, 101]
    assert [n.time for n in piano.notes if n.pitch in (100, 101)] == [lo, lo + BAR]
    # outside the span: every original note, on both tracks, unchanged
    for orig, new in zip(s.tracks, out.tracks):
        before = [(n.time, n.pitch, n.duration) for n in orig.notes if not lo <= n.time < hi]
        after = [(n.time, n.pitch, n.duration) for n in new.notes
                 if not lo <= n.time < hi and n.pitch < 100]
        assert before == after
    assert not any(lo <= n.time < hi for n in out.tracks[1].notes)   # bass span cleared
    assert out.tracks[-1].is_drum and out.tracks[-1].name == "infill"
    assert all(n.time + n.duration <= hi for n in piano.notes if n.pitch >= 100)


def test_splice_cuts_a_note_ringing_into_the_span():
    s = _song(8, tracks=((0, False, 60),))
    s.tracks[0].notes.append(Note(3 * BAR - TPQ, 3 * TPQ, 90, 80))   # rings into bar 3
    p = plan(_bytes(s), 3, 2)
    out = splice(p, Score(16))
    held = [n for n in out.tracks[0].notes if n.pitch == 90]
    assert held and held[0].time + held[0].duration == 3 * BAR


@pytest.fixture(scope="module")
def gen():
    from midigenai.generate import Generator
    from midigenai.model import ModelConfig, MusicTransformer
    from midigenai.tokenizer import build_tokenizer, save_tokenizer
    tok = build_tokenizer(scheme="v4")
    d = Path(tempfile.mkdtemp())
    save_tokenizer(tok, d / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=1024)
    torch.manual_seed(0)
    torch.save({"model": MusicTransformer(cfg).state_dict(), "model_config": asdict(cfg)},
               d / "ckpt.pt")
    return Generator(checkpoint_path=d / "ckpt.pt", tokenizer_path=d / "tokenizer.json",
                     device=torch.device("cpu"))


def test_prompt_segments_have_exact_bar_counts(gen):
    p = plan(_bytes(_song(8)), 3, 2)
    prefix, suffix = infill_span.prompt_segments(gen, p)
    assert gen.count_bars(prefix) == p.prefix_bars
    assert gen.count_bars(suffix) == p.suffix_bars


def test_run_end_to_end_keeps_everything_outside_the_span(gen):
    s = _song(8)
    p, outs, counts = infill_span.run(gen, _bytes(s), 3, 2, n_samples=2,
                                      max_new_tokens=200, top_k=None)
    assert len(outs) == 2 and len(counts) == 2
    lo, hi = p.span_ticks
    for out in outs:
        for orig, new in zip(s.tracks, out.tracks):
            keep = lambda tr: [(n.time, n.pitch) for n in tr.notes if not lo <= n.time < hi]
            assert keep(orig) == keep(new)


def test_infill_keeps_a_leading_rest(gen, monkeypatch):
    """A gap that opens with an empty bar must stay a bar late: infill does
    not trim leading bars the way continuation does."""
    bar, ts = gen.bar_id, gen.tokenizer.vocab["TimeSig_4/4"]
    pos = next(v for k, v in gen.tokenizer.vocab.items() if k.startswith("Position_"))
    raw = [bar, ts, bar, ts, pos, pos, bar]
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter(raw))
    out = list(gen.infill([bar, ts], [bar, ts], 2))
    assert out[:4] == [bar, ts, bar, ts]


# ---------- the Flask route ---------- #

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import midigenai.web_server as ws
    monkeypatch.setattr(ws, "GENERATED_FOLDER", str(tmp_path))
    calls = []

    class _Remote:
        def remote(self, midi_bytes, **kw):
            calls.append(kw)
            m = _bytes(_song(8))
            return {"start": kw["start"], "bars": kw["bars"], "bars_available": 8,
                    "context": [0, 8], "span_seconds": [6.0, 10.0],
                    "generated_notes": [3, 4], "midis": [m, m]}

    class _Gen:
        infill_batch = _Remote()

    monkeypatch.setattr(ws, "_generator", lambda version: _Gen())
    ws.app.config["TESTING"] = True
    return ws.app.test_client(), calls


def _post(c, query, data=None):
    return c.post(f"/api/infill?{query}", data={
        "midiFile": (io.BytesIO(data or _bytes(_song(8))), "song.mid")},
        content_type="multipart/form-data")


def test_route_happy_path(client):
    c, calls = client
    r = _post(c, "start=3&bars=2")
    assert r.status_code == 200, r.json
    assert r.json["start"] == 3 and r.json["bars"] == 2 and r.json["midiUrl2"]
    assert calls[0]["start"] == 3 and calls[0]["n_samples"] == 2


@pytest.mark.parametrize("query,needle", [
    ("bars=2", "start="),
    ("start=0&bars=2", "at least one kept bar"),
    ("start=3&bars=9", "1-4 bars"),
    ("start=3&model=v3", "infill needs one of"),
])
def test_route_refuses_before_reaching_modal(client, query, needle):
    c, calls = client
    r = _post(c, query)
    assert r.status_code == 400 and needle in r.json["error"]
    assert not calls
