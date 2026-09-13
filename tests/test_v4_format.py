"""v4 tokenizer + document formats: Bar/Position REMI scheme, attribute
header, accompaniment / span-infill documents, header re-injection and
dropout in the training stream, REMI-aware pitch-shift augmentation."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from symusic import Note, Score, TimeSignature, Track

from midigenai.attributes import header_for_score, is_header_token
from midigenai.data.augment import TokenAugmenter
from midigenai.data.v4_docs import SKIP_TIMESIG, DocBuilder
from midigenai.sequence_format import (
    Specials, accompaniment_prompt, count_bars, drop_header_families,
    infill_prompt, split_header,
)
from midigenai.tokenizer import build_tokenizer, is_v4, load_tokenizer, save_tokenizer
from midigenai.train import ShardedTokenStream


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer(scheme="v4")


@pytest.fixture(scope="module")
def sp(tok):
    return Specials.from_tokenizer(tok)


def _song(n_bars=24, ts=(4, 4), tpq=480, drums=True) -> Score:
    s = Score(tpq)
    s.time_signatures.append(TimeSignature(0, *ts))
    bar = tpq * 4 * ts[0] // ts[1]
    piano, bass, dr = Track(program=0), Track(program=33), Track(program=0, is_drum=True)
    for b in range(n_bars):
        for beat in range(ts[0]):
            t = b * bar + beat * tpq
            piano.notes.append(Note(t, tpq // 2, 60 + (beat * 2) % 12, 80))
            piano.notes.append(Note(t, tpq // 2, 64 + (beat * 2) % 12, 70))
            dr.notes.append(Note(t, tpq // 4, 36 if beat % 2 == 0 else 38, 100))
        bass.notes.append(Note(b * bar, bar, 36 + b % 5, 90))
    s.tracks.append(piano)
    s.tracks.append(bass)
    if drums:
        s.tracks.append(dr)
    return s


def _write(score: Score, name="song.mid") -> str:
    d = tempfile.mkdtemp()
    p = str(Path(d) / name)
    score.dump_midi(p)
    return p


def test_vocab_and_roundtrip(tok, sp):
    assert is_v4(tok)
    assert sp.mask is not None and sp.bar is not None and len(sp.header_ids) > 40
    d = tempfile.mkdtemp()
    save_tokenizer(tok, Path(d) / "tokenizer.json")
    tok2 = load_tokenizer(Path(d) / "tokenizer.json")
    assert type(tok2).__name__ == "REMI" and len(tok2.vocab) == len(tok.vocab)


def test_header_tokens_are_ignored_on_decode(tok, sp):
    ids = tok(_song(4)).ids
    names = header_for_score(_song(4), source="lakh")
    assert {"Inst_Piano", "Inst_Bass", "Inst_Drums", "Source_lakh"} <= set(names)
    assert all(is_header_token(n) for n in names)
    with_header = [sp.bos, *sp.header_ids_for(tok, names), *ids, sp.eos]

    def notes(sc):
        return sorted((n.time, n.pitch, n.duration) for t in sc.tracks for n in t.notes)
    assert notes(tok.decode(with_header)) == notes(tok.decode(ids))


def test_docbuilder_shapes(tok, sp):
    b = DocBuilder(tok, accomp_windows=3, infill_windows=3, window_bars=8,
                   context_bars=16, max_span_bars=4)
    docs = b.build(_write(_song(40)))
    assert isinstance(docs, dict)
    assert docs["continuation"] and docs["accompaniment"] and docs["infill"]
    for kind, kdocs in docs.items():
        for doc in kdocs:
            assert doc[0] == sp.bos
            # continuation docs end with EOS; segment targets do not (the
            # next document's BOS terminates them at inference)
            assert (doc[-1] == sp.eos) == (kind == "continuation")
            header, rest = split_header(sp, doc[1:] if kind != "continuation" else doc[1:-1])
            assert header and not any(t in sp.header_ids for t in rest)
    for doc in docs["accompaniment"]:
        _, rest = split_header(sp, doc[1:])
        i = rest.index(sp.sep)
        assert count_bars(sp, rest[:i]) == count_bars(sp, rest[i + 1:]) == 8
        assert sp.mask not in rest
    for doc in docs["infill"]:
        _, rest = split_header(sp, doc[1:])
        m, i = rest.index(sp.mask), rest.index(sp.sep)
        span = count_bars(sp, rest[i + 1:])
        assert 1 <= span <= 4
        assert count_bars(sp, rest[:m]) + span + count_bars(sp, rest[m + 1:i]) == 16


def test_short_target_is_padded_to_window(tok, sp):
    """A part that stops before the window end must still carry N bars."""
    s = _song(16)
    s.tracks[1].notes = s.tracks[1].notes[:6]      # bass stops after bar 6
    b = DocBuilder(tok, accomp_windows=6, infill_windows=0, window_bars=16)
    docs = b.build(_write(s))
    assert docs["accompaniment"]
    for doc in docs["accompaniment"]:
        _, rest = split_header(sp, doc[1:])
        i = rest.index(sp.sep)
        assert count_bars(sp, rest[:i]) == count_bars(sp, rest[i + 1:]) == 16


def test_single_track_file_has_no_accompaniment(tok):
    s = Score(480)
    s.tracks.append(_song(20).tracks[0])
    docs = DocBuilder(tok).build(_write(s))
    assert docs["continuation"] and not docs["accompaniment"] and docs["infill"]


def test_unsupported_time_signature_is_skipped(tok):
    assert DocBuilder(tok).build(_write(_song(8, ts=(13, 16)))) == SKIP_TIMESIG
    assert isinstance(DocBuilder(tok).build(_write(_song(8, ts=(5, 4)))), dict)


def test_prompts_end_where_generation_starts(sp):
    h, cond = [7, 8], [50, 51, 52]
    p = accompaniment_prompt(sp, h, cond)
    assert p[0] == sp.bos and p[-1] == sp.sep and p[1:3] == h
    q = infill_prompt(sp, h, [50], [51])
    assert q[-1] == sp.sep and q[q.index(sp.mask) + 1] == 51


def test_augmenter_shifts_pitch_only(tok, sp):
    aug = TokenAugmenter(tok, max_pitch_shift=2, max_velocity_jitter=0)
    ids = np.asarray(tok(_song(4)).ids, dtype=np.int64)
    inv = {v: k for k, v in tok.vocab.items()}
    out = aug.pitch_tables[2][ids]
    for a, b in zip(ids, out):
        na, nb = inv[int(a)], inv[int(b)]
        if na.startswith("Pitch_"):
            assert int(nb.split("_")[1]) == int(na.split("_")[1]) + 2
        else:
            assert na == nb                      # PitchDrum_, Bar, Position, ...


def _shard_dir(tok, sp) -> Path:
    b = DocBuilder(tok, accomp_windows=0, infill_windows=0, track_views=0)
    docs = [b.build(_write(_song(n), f"s{n}.mid"))["continuation"][0] for n in (30, 40, 50)]
    d = Path(tempfile.mkdtemp())
    (d / "shards").mkdir()
    np.save(d / "shards" / "train_00000.npy",
            np.concatenate([np.asarray(x, dtype=np.uint16) for x in docs]))
    return d


def test_stream_reinjects_header(tok, sp):
    d = _shard_dir(tok, sp)
    block = 128
    rng = np.random.default_rng(0)
    full = ShardedTokenStream([d / "shards" / "train_00000.npy"], block,
                              specials=sp, header_dropout=0.0, header_drop_all=0.0)
    x, _ = full.sample_batch(16, rng)
    for row in x.numpy():
        assert row[0] == sp.bos
        header, rest = split_header(sp, row[1:])
        assert len(header) >= 4                   # Inst x3 + Density/Poly/Range
        assert len(row) == block
    none = ShardedTokenStream([d / "shards" / "train_00000.npy"], block,
                              specials=sp, header_dropout=1.0, header_drop_all=0.0)
    x, y = none.sample_batch(8, rng)
    # the augmenter indexes remap tables with the window: must stay integer
    raw = none._window_v4(none.shards[0], none.all_starts[0], 5, rng)
    assert raw.dtype == np.int64 and len(raw) == block + 1
    from midigenai.data.augment import TokenAugmenter
    TokenAugmenter(tok, 6, 1)(raw, rng)
    for row, tgt in zip(x.numpy(), y.numpy()):
        assert row[0] == sp.bos and not any(t in sp.header_ids for t in row)
        assert (row[1:] == tgt[:-1]).all()        # x/y still shifted by one


def test_drop_header_families(sp, tok):
    names = ["Inst_Piano", "Inst_Bass", "Density_1", "Poly_0", "Source_lakh"]
    header = sp.header_ids_for(tok, names)
    rng = np.random.default_rng(1)
    assert drop_header_families(sp, header, rng, 0.0, 0.0) == header
    assert drop_header_families(sp, header, rng, 0.0, 1.0) == []
    kept = drop_header_families(sp, header, rng, 1.0, 0.0)
    assert kept == []
