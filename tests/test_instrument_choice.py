"""Picking the accompaniment's instrument.

The header names what the finished document contains, so "add a bass" is
written as the condition's family plus Inst_Bass. These tests pin the name
resolution and the header rewrite; the model call itself is covered by
test_v4_generate.py.
"""
import pytest

from midigenai.attributes import (
    DRUMS, HEADER_PREFIXES, INSTRUMENT_FAMILIES, MAX_INST_TOKENS,
    header_for_score, is_auto_instrument, resolve_family, sort_header,
    with_instruments,
)
from midigenai.modal_serve import accompaniment_header


@pytest.mark.parametrize("name,family", [
    ("bass", "Bass"),
    ("Bass", "Bass"),
    ("drums", "Drums"),
    ("Drum Kit", "Drums"),
    ("piano", "Piano"),
    ("electric piano", "Piano"),   # GM program 4 is still the Piano family
    ("synth lead", "SynthLead"),
    ("SynthLead", "SynthLead"),
    ("synth_pad", "SynthPad"),
    ("sax", "Reed"),
    ("808", "Bass"),
])
def test_names_resolve_to_families(name, family):
    assert resolve_family(name) == family


@pytest.mark.parametrize("junk", ["kazoo", "v4", "Inst_Bass", "theremin"])
def test_unknown_names_are_not_guessed(junk):
    """A typo has to be rejected, not silently turned into some other part."""
    assert resolve_family(junk) is None
    assert not is_auto_instrument(junk)


@pytest.mark.parametrize("auto", ["", "  ", "auto", "Any", "original", "none"])
def test_auto_is_told_apart_from_a_typo(auto):
    assert is_auto_instrument(auto)


def test_every_family_resolves_to_itself():
    for f in INSTRUMENT_FAMILIES + [DRUMS]:
        assert resolve_family(f) == f


def test_with_instruments_replaces_the_inst_list():
    names = ["Inst_Piano", "Inst_Strings", "Density_2", "Poly_1", "Range_1"]
    out = with_instruments(names, ["Piano", "Bass"])
    assert [n for n in out if n.startswith("Inst_")] == ["Inst_Piano", "Inst_Bass"]
    # everything else survives untouched
    assert [n for n in out if not n.startswith("Inst_")] == [
        "Density_2", "Poly_1", "Range_1"]


def test_with_instruments_keeps_canonical_order_and_drops_duplicates():
    out = with_instruments([], ["Drums", "Bass", "Bass", "Piano"])
    assert out == ["Inst_Piano", "Inst_Bass", "Inst_Drums"]


def test_with_instruments_respects_the_header_length_cap():
    out = with_instruments([], INSTRUMENT_FAMILIES + [DRUMS])
    assert len(out) == MAX_INST_TOKENS


def test_with_instruments_rejects_a_non_family():
    with pytest.raises(ValueError):
        with_instruments([], ["Bass", "kazoo"])


def test_header_stays_in_canonical_family_order():
    out = sort_header(["Density_1", "Task_accomp", "Inst_Bass", "Source_lakh"])
    prefixes = [next(p for p in HEADER_PREFIXES if n.startswith(p)) for n in out]
    assert prefixes == sorted(prefixes, key=HEADER_PREFIXES.index)


def _three_part_score():
    """Piano + Strings + Guitar, one bar each -- a multitrack upload."""
    symusic = pytest.importorskip("symusic")
    score = symusic.Score(480)
    score.time_signatures.append(symusic.TimeSignature(0, 4, 4))
    for program in (0, 40, 24):        # GM: Piano, Strings, Guitar
        track = symusic.Track(program=program)
        for beat in range(4):
            track.notes.append(symusic.Note(beat * 480, 240, 60 + beat, 80))
        score.tracks.append(track)
    return score


def test_asking_for_an_instrument_narrows_a_multitrack_header():
    """A multitrack upload's own header reads as 'put all of these back'.
    Asking for a part replaces the Inst_ list with condition + request."""
    score = _three_part_score()
    assert {"Inst_Piano", "Inst_Guitar", "Inst_Strings"} <= set(header_for_score(score))

    names, family = accompaniment_header(score, cond_index=0, instrument="bass")
    assert family == "Bass"
    assert [n for n in names if n.startswith("Inst_")] == ["Inst_Piano", "Inst_Bass"]
    # density / poly / range still describe the window, untouched
    assert [n for n in names if not n.startswith("Inst_")] == \
           [n for n in header_for_score(score) if not n.startswith("Inst_")]


def test_the_condition_track_decides_which_family_is_kept():
    """cond_index picks the part being answered, so its family -- not the
    file's busiest or first -- is the one that stays in the header."""
    score = _three_part_score()
    names, _ = accompaniment_header(score, cond_index=2, instrument="drums")
    assert [n for n in names if n.startswith("Inst_")] == ["Inst_Guitar", "Inst_Drums"]


def test_no_instrument_leaves_the_upload_header_alone():
    """The pre-existing default: nothing is overridden and the model picks."""
    score = _three_part_score()
    for asked in (None, "", "auto"):
        names, family = accompaniment_header(score, cond_index=0, instrument=asked)
        assert family is None
        assert names == header_for_score(score)


def test_a_typo_is_refused_rather_than_generated_from():
    score = _three_part_score()
    with pytest.raises(ValueError, match="kazoo"):
        accompaniment_header(score, cond_index=0, instrument="kazoo")


def test_asking_for_the_conditions_own_family_is_a_single_token():
    score = _three_part_score()
    names, family = accompaniment_header(score, cond_index=0, instrument="piano")
    assert family == "Piano"
    assert [n for n in names if n.startswith("Inst_")] == ["Inst_Piano"]
