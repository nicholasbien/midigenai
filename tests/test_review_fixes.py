"""Two bugs the accompaniment/probe paths shipped with.

Both are silent: one hands the judge the wrong tracks, the other pools
activations from the wrong positions. Neither raises, so only a test that
asserts on the boundary itself catches them.
"""
import json

import numpy as np
import pytest
import torch
from symusic import Note, Score, Tempo, TimeSignature, Track

from midigenai.pairgen import repair_cond_tracks
from midigenai.reward_probe import fit_to_context
from midigenai.tokenizer import build_tokenizer


def _track(program, base, n=4, drum=False):
    t = Track(program=program, is_drum=drum)
    for i in range(n):
        t.notes.append(Note(i * 480, 240, base + i, 80))
    return t


def _score(*tracks):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    for t in tracks:
        s.tracks.append(t)
    return s


# ---------- the condition track count the judge slices at ---------- #

@pytest.fixture(scope="module")
def tok():
    return build_tokenizer(scheme="v4")


def test_same_program_condition_tracks_decode_to_one(tok):
    """The premise: one_token_stream merges by program, so a source count is
    not a mix count. If this ever stops being true the slice fix is moot."""
    cond = _score(_track(0, 60), _track(0, 72))
    assert len(cond.tracks) == 2
    assert len(tok.decode(list(tok(cond).ids)).tracks) == 1


def test_distinct_programs_survive_as_separate_tracks(tok):
    cond = _score(_track(0, 60), _track(33, 40))
    assert len(tok.decode(list(tok(cond).ids)).tracks) == 2


def test_mix_slice_at_decoded_count_keeps_every_generated_track(tok):
    """The bug end to end: slicing at the SOURCE count drops generated
    parts, slicing at the decoded count keeps exactly them."""
    cond = _score(_track(0, 60), _track(0, 72))          # 2 source -> 1 decoded
    acc = tok.decode(list(tok(_score(_track(33, 40), _track(25, 50))).ids))
    cond_score = tok.decode(list(tok(cond).ids))

    mix = cond_score.copy()
    for t in acc.tracks:
        mix.tracks.append(t)
    mix.tempos = [Tempo(time=0, qpm=120)]
    mix = Score.from_midi(mix.dumps_midi())              # as the judge reads it

    n_generated = len(acc.tracks)
    assert len(mix.tracks) == len(cond_score.tracks) + n_generated

    kept_decoded = mix.tracks[len(cond_score.tracks):]
    assert len(kept_decoded) == n_generated              # fixed behaviour

    kept_source = mix.tracks[len(cond.tracks):]
    assert len(kept_source) < n_generated                # the bug: parts lost


def test_repair_recomputes_stored_counts_from_cond_ids(tmp_path, tok):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    cond_ids = list(tok(_score(_track(0, 60), _track(0, 72))).ids)
    (pairs / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "mode": "accompany",
        "n_cond_tracks": 2, "cond_ids": cond_ids}))        # the wrong count
    (pairs / "p2.json").write_text(json.dumps({
        "pair_id": "p2", "mode": "continue", "n_cond_tracks": 9}))

    dry = repair_cond_tracks(pairs, tok, apply=False)
    assert dry == [("p1", 2, 1)]
    assert json.loads((pairs / "p1.json").read_text())["n_cond_tracks"] == 2

    assert repair_cond_tracks(pairs, tok, apply=True) == [("p1", 2, 1)]
    assert json.loads((pairs / "p1.json").read_text())["n_cond_tracks"] == 1
    # continuation pairs are left alone, and a second run is a no-op
    assert json.loads((pairs / "p2.json").read_text())["n_cond_tracks"] == 9
    assert repair_cond_tracks(pairs, tok, apply=True) == []


# ---------- the prompt boundary under truncation ---------- #

def test_boundary_untouched_when_the_sequence_fits():
    seq, n_prompt = fit_to_context(list(range(10)), list(range(10, 20)), 64, "cpu")
    assert seq.shape[1] == 20 and n_prompt == 10


def test_boundary_moves_with_the_cut():
    """The prompt is cut from the front, so the boundary has to move by
    exactly what was dropped -- otherwise the pooled slice is offset."""
    prompt, cont = list(range(100)), list(range(100, 150))
    seq, n_prompt = fit_to_context(prompt, cont, 120, "cpu")
    assert seq.shape[1] == 120
    assert n_prompt == 70                      # 100 - (150 - 120)
    # the continuation still starts exactly at the boundary, whole
    assert seq[0, n_prompt:].tolist() == cont
    assert n_prompt != len(prompt)             # what the bug used


def test_continuation_always_survives_truncation_whole():
    prompt, cont = list(range(500)), list(range(500, 560))
    seq, n_prompt = fit_to_context(prompt, cont, 100, "cpu")
    assert seq[0, n_prompt:].tolist() == cont


def test_a_prompt_cut_away_entirely_leaves_a_usable_slice():
    """With the boundary at 0 the pooled slice would be empty, which reaches
    the caller as a NaN and silently drops the pair."""
    seq, n_prompt = fit_to_context(list(range(50)), list(range(50, 150)), 100, "cpu")
    assert n_prompt >= 1
    sl = slice(n_prompt - 1, -1)
    assert len(range(*sl.indices(seq.shape[1]))) > 0


def test_the_pooled_slice_is_never_empty_across_shapes():
    for n_p, n_c, cap in [(10, 10, 64), (100, 50, 120), (500, 60, 100),
                          (2040, 500, 2048), (5, 8, 8)]:
        seq, n_prompt = fit_to_context(list(range(n_p)), list(range(n_c)), cap, "cpu")
        sl = slice(n_prompt - 1, -1) if seq.shape[1] > n_prompt else slice(-1, None)
        pooled = torch.zeros(seq.shape[1], 3)[sl]
        assert pooled.shape[0] > 0, (n_p, n_c, cap)
        assert np.isfinite(pooled.float().mean(0).numpy()).all()
