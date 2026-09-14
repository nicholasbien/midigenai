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
            # task token opens segment docs, never a continuation
            expect = {"accompaniment": sp.task_accomp, "infill": sp.task_infill}.get(kind)
            assert (doc[1] == expect) if expect else (doc[1] not in (sp.task_accomp, sp.task_infill))
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


def test_single_track_file_accompaniment_only_via_hand_split(tok):
    """A single track has no other part to predict - unless it is a keyboard
    part, where the two hands are a real pair (see split_hands)."""
    s = Score(480)
    s.tracks.append(_song(20).tracks[0])
    path = _write(s)
    assert not DocBuilder(tok, hand_split_windows=0).build(path)["accompaniment"]
    docs = DocBuilder(tok).build(path)
    assert docs["continuation"] and docs["infill"]


def test_unsupported_time_signature_is_skipped(tok):
    assert DocBuilder(tok).build(_write(_song(8, ts=(13, 16)))) == SKIP_TIMESIG
    assert isinstance(DocBuilder(tok).build(_write(_song(8, ts=(5, 4)))), dict)


def test_prompts_end_where_generation_starts(sp):
    h, cond = [7, 8], [50, 51, 52]
    p = accompaniment_prompt(sp, h, cond)
    assert p[0] == sp.bos and p[1] == sp.task_accomp and p[-1] == sp.sep and p[2:4] == h
    q = infill_prompt(sp, h, [50], [51])
    assert q[1] == sp.task_infill and q[-1] == sp.sep and q[q.index(sp.mask) + 1] == 51


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
    # the task token survives every dropout setting
    task_header = [sp.task_accomp, *header]
    assert drop_header_families(sp, task_header, rng, 1.0, 0.0) == [sp.task_accomp]
    assert drop_header_families(sp, task_header, rng, 0.0, 1.0) == [sp.task_accomp]


def test_looks_like_drums():
    """Content-based drum promotion: strict enough to leave a bass line alone."""
    from midigenai.tokenizer import looks_like_drums, normalize_drums

    kit = Track(program=0)
    for b in range(8):
        for beat in range(4):
            t = b * 1920 + beat * 480
            kit.notes.append(Note(t, 60, 36 if beat % 2 == 0 else 38, 100))
            kit.notes.append(Note(t, 60, 42, 80))
    assert looks_like_drums(kit)

    bass = Track(program=33)
    for i in range(40):
        bass.notes.append(Note(i * 480, 400, 36 + (i * 7) % 19, 90))
    assert not looks_like_drums(bass)          # too many distinct pitches

    ostinato = Track(program=33)               # 4-note bass riff in the drum range
    for i in range(40):
        ostinato.notes.append(Note(i * 480, 400, [36, 41, 46, 51][i % 4], 90))
    assert not looks_like_drums(ostinato)      # kick+hat but no snare

    riff = Track(program=0)                    # few pitches, but not a kit
    for i in range(40):
        riff.notes.append(Note(i * 240, 200, [40, 45, 47][i % 3], 90))
    assert not looks_like_drums(riff)          # no kick, no hat

    assert not looks_like_drums(Track(program=0))   # empty

    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    s.tracks.append(kit)
    s.tracks.append(bass)
    assert normalize_drums(s, "untitled.mid") == 1
    assert s.tracks[0].is_drum and not s.tracks[1].is_drum


def test_prompt_window_keeps_program_state():
    """A prompt window cut after the file's Program token must carry it, or a
    drum kit decodes as piano (real bug, val_gigamidi_5aa9c6cc..., 2026-09-14)."""
    from types import SimpleNamespace

    from midigenai.label_app import PairFactory
    tok = build_tokenizer()                       # MIDILike, as v3 uses
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    kit = Track(program=0, is_drum=True)
    for b in range(24):
        for beat in range(4):
            kit.notes.append(Note(b * 1920 + beat * 480, 60, 36 if beat % 2 == 0 else 38, 100))
    s.tracks.append(kit)
    ids = tok(s).ids
    inv = {v: k for k, v in tok.vocab.items()}
    assert inv[ids[0]] == "Program_-1"            # drums declared once, up front

    f = PairFactory.__new__(PairFactory)
    f.gen_a = SimpleNamespace(tokenizer=tok)
    window = f._slice_with_program(ids, 40, 64)
    assert inv[window[0]] == "Program_-1"
    assert tok.decode(window).tracks[0].is_drum
    # without the fix the same window decodes as a pitched track
    assert not tok.decode(ids[40:104]).tracks[0].is_drum


def test_name_says_drums_needs_boundaries():
    """"909" inside a numeric file id is not a TR-909 (real bug: Aria piano
    transcription val_aria_909098_0.mid was served and trained as drums)."""
    from midigenai.tokenizer import name_says_drums, normalize_drums

    assert name_says_drums("Drums")
    assert name_says_drums("acoustic snare")
    assert name_says_drums("HiHat")
    assert name_says_drums("TR-909")
    assert name_says_drums("909_kit")
    assert not name_says_drums("val_aria_909098_0.mid")
    assert not name_says_drums("785909_0.mid")
    assert not name_says_drums("9d4b20cadb6f4a8909b2a73bad2c0014.mid")
    assert not name_says_drums("What a Wonderful World")   # 'hat' inside a word
    assert not name_says_drums("Tomorrow Never Knows")     # 'tom' inside a word

    s = Score(480)
    piano = Track(program=0)
    for i in range(40):
        piano.notes.append(Note(i * 240, 200, 60 + (i * 3) % 24, 80))
    s.tracks.append(piano)
    assert normalize_drums(s, "val_aria_909098_0.mid") == 0
    assert not s.tracks[0].is_drum


def test_hand_split_makes_solo_piano_contribute_accompaniment(tok, sp):
    """Aria is a third of the corpus and single-track, so it produced no
    accompaniment documents at all; the hands are a real pair, both ways."""
    from midigenai.data.v4_docs import split_hands

    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    piano = Track(program=0)
    for b in range(20):
        # left hand sustains under the bar, right hand plays over it — the
        # shape split_hands looks for (see _held_under)
        piano.notes.append(Note(b * 1920, 1900, 40 + (b % 5), 70))
        for beat in range(4):
            piano.notes.append(Note(b * 1920 + beat * 480, 220,
                                    72 + (beat * 2) % 7, 90))
    s.tracks.append(piano)

    low, high = split_hands(s)
    assert max(n.pitch for t in low.tracks for n in t.notes) < \
           min(n.pitch for t in high.tracks for n in t.notes)

    path = _write(s, "solo.mid")
    docs = DocBuilder(tok, accomp_windows=4, infill_windows=0, track_views=0,
                      window_bars=8, hand_split_windows=2).build(path)
    assert docs["accompaniment"], "solo piano should now yield accompaniment docs"
    for doc in docs["accompaniment"]:
        _, rest = split_header(sp, doc[1:])
        i = rest.index(sp.sep)
        assert count_bars(sp, rest[:i]) == count_bars(sp, rest[i + 1:]) == 8
    # both directions present: the two conditions differ
    conds = {tuple(split_header(sp, d[1:])[1][:split_header(sp, d[1:])[1].index(sp.sep)])
             for d in docs["accompaniment"]}
    assert len(conds) >= 2

    # off by default for multi-track files, and disabled by 0
    assert not DocBuilder(tok, accomp_windows=0, infill_windows=0, track_views=0,
                          window_bars=8, hand_split_windows=0).build(path)["accompaniment"]


def test_pitch_class_overlap_survives_different_tick_rates():
    """A decoded generation runs at 16 ticks/quarter, a condition from a file
    at 480; bucketing both with one bar length compared bar 1 against bar 30."""
    from midigenai.eval import pitch_class_overlap

    def piece(tpq, pitches):
        s = Score(tpq)
        s.time_signatures.append(TimeSignature(0, 4, 4))
        t = Track(program=0)
        for bar in range(4):
            for beat in range(4):
                t.notes.append(Note(bar * tpq * 4 + beat * tpq, tpq // 2,
                                    pitches[bar % len(pitches)], 80))
        s.tracks.append(t)
        return s

    same_pitches = [60, 62, 64, 65]
    a, b = piece(16, same_pitches), piece(480, same_pitches)
    assert pitch_class_overlap(a, b) == 1.0          # identical music, two clocks
    assert pitch_class_overlap(piece(480, same_pitches), b) == 1.0
    c = piece(16, [61, 63, 66, 68])                   # disjoint pitch classes
    assert pitch_class_overlap(c, b) == 0.0


def test_hand_split_rejects_a_single_melodic_line():
    """Slicing a melody at a pitch gives two half-melodies, not two hands."""
    from midigenai.data.v4_docs import split_hands

    def window(build):
        s = Score(480)
        s.time_signatures.append(TimeSignature(0, 4, 4))
        t = Track(program=0)
        build(t)
        s.tracks.append(t)
        return s

    def melody(t):                      # one line, notes never overlap
        for i in range(48):
            t.notes.append(Note(i * 240, 220, 48 + (i * 5) % 30, 80))

    def two_hands(t):                   # sustained left hand under a melody
        for bar in range(12):
            t.notes.append(Note(bar * 1920, 1900, 43 + bar % 5, 70))
            for beat in range(4):
                t.notes.append(Note(bar * 1920 + beat * 480, 220, 72 + beat, 90))

    assert split_hands(window(melody)) is None
    assert split_hands(window(two_hands)) is not None


def test_single_target_accompaniment(tok, sp):
    """"Add a bass" is the shape the jam asks for, so most accompaniment
    documents should target one track and name it in the header."""
    from midigenai.attributes import is_header_token
    inv = {v: k for k, v in tok.vocab.items()}

    s = _song(40)                       # piano + bass + drums
    path = _write(s, "trio.mid")
    docs = DocBuilder(tok, accomp_windows=12, infill_windows=0, track_views=0,
                      window_bars=8, single_target_frac=1.0).build(path)
    assert docs["accompaniment"]
    for doc in docs["accompaniment"]:
        header, rest = split_header(sp, doc[1:])
        names = [inv[t] for t in header if is_header_token(inv[t])]
        insts = [n for n in names if n.startswith("Inst_")]
        # condition (1-2 tracks) + exactly one target: at most three families
        assert 1 <= len(insts) <= 3, names
        i = rest.index(sp.sep)
        assert count_bars(sp, rest[:i]) == count_bars(sp, rest[i + 1:]) == 8

    everything = DocBuilder(tok, accomp_windows=12, infill_windows=0, track_views=0,
                            window_bars=8, single_target_frac=0.0).build(path)
    assert everything["accompaniment"]
