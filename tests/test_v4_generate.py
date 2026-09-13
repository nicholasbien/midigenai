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
    out = list(gen.generate_ids([gen.bos_id, pos, pitch], stop_after_bars=2, max_new_tokens=50))
    assert out == fake[:8]
    # prompt ending on a bar line: that bar line is bar 1 of the answer
    out = list(gen.generate_ids([gen.bos_id, pos, pitch, bar, ts], stop_after_bars=2, max_new_tokens=50))
    assert out == fake[:4]
    monkeypatch.setattr(gen, "_generate_raw", lambda *a, **k: iter([pos, pitch, gen.sp.sep, pitch]))
    assert list(gen.generate_ids([gen.bos_id, pos], max_new_tokens=50)) == [pos, pitch]


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
