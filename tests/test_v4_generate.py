"""Generator v4 helpers on a tiny random checkpoint: header building,
bar padding, bar-count stop, split_bars, accompany/infill prompt plumbing."""
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai.generate import Generator
from midigenai.sequence_format import accompaniment_prompt
from midigenai.model import ModelConfig, MusicTransformer
from midigenai.tokenizer import build_tokenizer, save_tokenizer


@pytest.fixture(scope="module")
def gen():
    tok = build_tokenizer(scheme="v4")
    d = Path(tempfile.mkdtemp())
    save_tokenizer(tok, d / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=512)
    torch.manual_seed(0)
    model = MusicTransformer(cfg)
    torch.save({"model": model.state_dict(), "model_config": asdict(cfg)}, d / "ckpt.pt")
    return Generator(d / "ckpt.pt", d / "tokenizer.json", backend="torch",
                     dtype=torch.float32, inference_seq_len=512)


def _phrase(n_bars=2, extra_beats=0):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    for b in range(n_bars):
        for beat in range(4):
            t.notes.append(Note(b * 1920 + beat * 480, 240, 60 + beat, 80))
    for beat in range(extra_beats):
        t.notes.append(Note(n_bars * 1920 + beat * 480, 240, 67, 80))
    s.tracks.append(t)
    return s


def test_make_header(gen):
    inv = {v: k for k, v in gen.tokenizer.vocab.items()}
    h = gen.make_header(instruments=["Piano", "Bass"], density=1, source="lakh")
    assert [inv[t] for t in h] == ["Inst_Piano", "Inst_Bass", "Density_1", "Source_lakh"]
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        _phrase().dump_midi(f.name)
        names = [inv[t] for t in gen.make_header(f.name, instruments=["Drums"])]
    assert "Inst_Drums" in names and "Inst_Piano" not in names and any(n.startswith("Density_") for n in names)


def test_pad_to_bars_and_bar_line(gen):
    ids = gen.tokenizer(_phrase(2, extra_beats=1)).ids     # 2 bars + 1 beat
    assert gen.count_bars(ids) == 3
    assert not gen.ends_on_bar_line(ids)
    closed = gen.close_bar(ids)
    assert gen.count_bars(closed) == 4 and gen.ends_on_bar_line(closed)   # the Bar that opens bar 4
    assert gen.close_bar(closed) == closed
    padded = gen.pad_to_bars(ids, 4)
    assert gen.count_bars(padded) == 4 and gen.ends_on_bar_line(padded)
    with pytest.raises(ValueError):
        gen.pad_to_bars(ids, 2)


def test_stop_after_bars_filters_stream(gen, monkeypatch):
    """The Bar that would open bar N+1 ends generation; SEP ends it silently."""
    bar, ts = gen.bar_id, gen.tokenizer.vocab["TimeSig_4/4"]
    pos, pitch = gen.tokenizer.vocab["Position_0"], gen.tokenizer.vocab["Pitch_60"]
    fake = [bar, ts, pos, pitch, bar, ts, pos, pitch, bar, ts, pos, pitch]
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter(fake))
    out = list(gen.generate_ids([gen.bos_id, pos, pitch], stop_after_bars=2,
                                max_new_tokens=50, trim_leading_bars=False))
    assert out == fake[:8]
    # prompt ending on a bar line: that bar line is bar 1 of the answer
    out = list(gen.generate_ids([gen.bos_id, pos, pitch, bar, ts], stop_after_bars=2,
                                max_new_tokens=50, trim_leading_bars=False))
    assert out == fake[:4]
    # trim_leading_bars (default): empty bars before the first note are
    # dropped. A prompt that ends mid-bar keeps ONE Bar, which closes it --
    # without it the first generated Position would date to the prompt's own
    # bar (see test_trim_leading_bars_keeps_the_prompts_bar_line).
    out = list(gen.generate_ids([gen.bos_id, pos, pitch], stop_after_bars=2,
                                max_new_tokens=50))
    assert out == fake[:8]
    assert gen.count_bars(out) == 2
    # a prompt already on a bar line has no open bar to close, so every
    # leading empty bar is dropped; that bar line is bar 1 of the answer
    out = list(gen.generate_ids([gen.bos_id, pos, pitch, bar, ts], stop_after_bars=2,
                                max_new_tokens=50))
    assert out == fake[2:8]
    assert gen.count_bars(out) == 1   # plus the prompt's own bar line
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter([pos, pitch, gen.sp.sep, pitch]))
    assert list(gen.generate_ids([gen.bos_id, pos], max_new_tokens=50)) == [pos, pitch]


def test_ban_ids_never_sampled(gen):
    """Random model, no top-k: without a ban SEP/MASK/BOS appear; with the
    default v4 ban they never do."""
    ids = gen.tokenizer(_phrase(2)).ids
    banned = {gen.sp.sep, gen.sp.mask, gen.bos_id}
    out = list(gen._generate_raw([gen.bos_id, *ids], 300, 1.0, None, 0, 0, None))
    assert any(t in banned for t in out)             # sanity: the random model does emit them
    out = list(gen.generate_ids(ids, max_new_tokens=300, top_k=None, seed=0))
    assert not any(t in banned for t in out)
    p = accompaniment_prompt(gen.sp, [], ids)
    assert p[1] == gen.sp.task_accomp and p[-1] == gen.sp.sep


def test_split_bars(gen):
    ids = gen.tokenizer(_phrase(6)).ids
    prefix, suffix = gen.split_bars(ids, 2, 2)
    assert gen.count_bars(prefix) == 2 and gen.count_bars(suffix) == 2
    assert suffix[0] == gen.bar_id


def test_accompany_and_infill_run(gen):
    """End to end on the random model: prompt layout is right and output is a
    decodable segment of at most `bars` bars."""
    ids = gen.tokenizer(_phrase(4)).ids
    header = gen.make_header(instruments=["Piano", "Bass"])
    out = list(gen.accompany(ids, 4, header=header, max_new_tokens=200, top_k=None))
    assert gen.count_bars(out) <= 4
    gen.tokenizer.decode(out)                       # self-contained segment decodes
    prefix, suffix = gen.split_bars(ids, 1, 2)
    out = list(gen.infill(prefix, suffix, 2, header=header, max_new_tokens=200, top_k=None))
    assert gen.count_bars(out) <= 2
    # non-v4 guard
    legacy = Generator.__new__(Generator)
    legacy.v4 = False
    with pytest.raises(RuntimeError):
        Generator._require_v4(legacy, "accompany")


def test_trim_leading_bars_keeps_the_prompts_bar_line(gen, monkeypatch):
    """Position is bar-relative, so a continuation that opens with `Bar` needs
    that Bar kept when the prompt stops mid-bar: dropping it decodes the first
    generated note into the bar the prompt is already part-way through, i.e.
    before the prompt ends. Checked on decoded times, not token ids."""
    bar, ts = gen.bar_id, gen.tokenizer.vocab["TimeSig_4/4"]
    v = gen.tokenizer.vocab

    def note(position, pitch):
        return [v[f"Position_{position}"], v["Program_0"], v[f"Pitch_{pitch}"],
                v["Velocity_79"], v["Duration_1.0.8"]]

    # prompt: notes on beats 0 and 2 of a 4/4 bar -> ends mid-bar
    prompt = [gen.bos_id, bar, ts, *note(0, 60), *note(16, 62)]
    fake = [bar, ts, *note(0, 67)]
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter(fake))
    new_ids = list(gen.generate_ids(prompt, max_new_tokens=50))

    score = gen.tokenizer.decode(prompt + new_ids)
    tpq = max(score.ticks_per_quarter, 1)
    beats = sorted(n.time / tpq for tr in score.tracks for n in tr.notes)
    assert beats == [0.0, 2.0, 4.0]   # the generated note lands on the next bar

    # a prompt already on a bar line has no open bar to close: the empty bar
    # the model offers is dropped and the note lands on the bar line we gave it
    on_line = prompt + [bar, ts]
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter(fake))
    score = gen.tokenizer.decode(on_line + list(gen.generate_ids(on_line, max_new_tokens=50)))
    tpq = max(score.ticks_per_quarter, 1)
    beats = sorted(n.time / tpq for tr in score.tracks for n in tr.notes)
    assert beats == [0.0, 2.0, 4.0]
