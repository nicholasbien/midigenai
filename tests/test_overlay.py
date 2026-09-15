"""Stacking a generated accompaniment on the part it answers.

The failure this guards against already shipped once: a score decoded from
tokens carries the tokenizer's tick rate and one parsed from a file carries
the file's, and merging them without resampling reinterpreted every
generated tick against the wrong grid. Audibly, the whole accompaniment
fired at one instant instead of spreading across the window.
"""
import pytest
from symusic import Note, Score, Track

from midigenai.generate import densest_track, overlay


def _score(tpq, beats, program=0):
    s = Score(tpq)
    tr = Track(program=program)
    for b in beats:
        tr.notes.append(Note(time=int(b * tpq), duration=tpq // 2,
                             pitch=60 + int(b), velocity=80))
    s.tracks.append(tr)
    return s


def test_overlay_keeps_both_parts_in_beat_time_across_tick_rates():
    condition = _score(480, [0, 1, 2, 3])      # as parsed from a file
    answer = _score(16, [0.5, 1.5, 2.5])       # as decoded from tokens

    out = overlay(condition, answer)
    tpq = out.ticks_per_quarter
    beats = sorted(n.time / tpq for tr in out.tracks for n in tr.notes)
    assert beats == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_overlay_does_not_collapse_the_answer_onto_one_instant():
    """The shipped bug, stated as a property: the answer must still span the
    window after the merge, not pile up at its start."""
    condition = _score(480, [0, 1, 2, 3])
    answer = _score(16, [0.5, 1.5, 2.5])

    out = overlay(condition, answer)
    tpq = out.ticks_per_quarter
    added = sorted(n.time / tpq for tr in out.tracks[1:] for n in tr.notes)
    assert added[-1] - added[0] == pytest.approx(2.0)


def test_overlay_leaves_its_inputs_alone():
    condition = _score(480, [0, 1])
    answer = _score(16, [0.5])
    overlay(condition, answer)
    assert condition.ticks_per_quarter == 480
    assert answer.ticks_per_quarter == 16


def test_overlay_drops_empty_tracks():
    condition = _score(480, [0, 1])
    answer = _score(16, [])
    assert len(overlay(condition, answer).tracks) == 1


def test_densest_track_picks_the_busiest_part():
    s = Score(480)
    sparse, dense = Track(program=0), Track(program=1)
    for b in range(2):
        sparse.notes.append(Note(time=b * 480, duration=240, pitch=60, velocity=80))
    for b in range(9):
        dense.notes.append(Note(time=b * 240, duration=120, pitch=64, velocity=80))
    s.tracks.extend([sparse, dense])
    assert densest_track(s) == 1


def test_densest_track_is_a_no_op_for_single_track_input():
    assert densest_track(_score(480, [0, 1, 2])) == 0


def test_densest_track_rejects_an_empty_score():
    with pytest.raises(ValueError):
        densest_track(Score(480))
