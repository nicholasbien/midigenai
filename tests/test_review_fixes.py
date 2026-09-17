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


# ---------- over-long sequences keep the prompt, not the tail ---------- #
#
# The probe's whole value is that its activations attended over the prompt.
# Cutting the prompt's head changes what "fits what came before" means, and
# because the two sides of a pair share one prompt but differ in continuation
# length, a head cut removes a DIFFERENT amount from each side. These pin the
# policy: prompt whole, continuation tail trimmed, skip when there is no room.


def test_nothing_is_cut_when_the_sequence_fits():
    seq, n_prompt = fit_to_context(list(range(10)), list(range(10, 20)), 64, "cpu")
    assert seq.shape[1] == 20 and n_prompt == 10
    assert seq[0, n_prompt:].tolist() == list(range(10, 20))


def test_the_prompt_survives_whole_and_the_tail_goes():
    prompt, cont = list(range(100)), list(range(100, 200))
    seq, n_prompt = fit_to_context(prompt, cont, 120, "cpu")
    assert n_prompt == len(prompt)                 # boundary unmoved
    assert seq[0, :n_prompt].tolist() == prompt    # prompt intact
    assert seq.shape[1] == 120
    assert seq[0, n_prompt:].tolist() == cont[:20]  # tail dropped, head kept


def test_both_sides_of_a_pair_see_the_same_prompt():
    """The bug this replaces: sides share a prompt but differ in continuation
    length, so a total-length cut conditioned them on different context."""
    prompt = list(range(1700))
    short, long_ = list(range(300)), list(range(1088))
    seq_a, n_a = fit_to_context(prompt, short, 2048, "cpu")
    seq_b, n_b = fit_to_context(prompt, long_, 2048, "cpu")
    assert n_a == n_b == len(prompt)
    assert seq_a[0, :n_a].tolist() == seq_b[0, :n_b].tolist() == prompt


def test_a_pair_with_no_room_to_score_is_skipped():
    """Better no row than a row scored on three tokens of continuation."""
    assert fit_to_context(list(range(2045)), list(range(500)), 2048, "cpu") is None
    assert fit_to_context(list(range(3000)), list(range(500)), 2048, "cpu") is None


def test_a_too_short_continuation_is_skipped():
    assert fit_to_context(list(range(10)), list(range(3)), 2048, "cpu") is None


def test_feature_vector_skips_rather_than_returning_bad_features():
    from midigenai.model import ModelConfig, MusicTransformer
    from midigenai.reward_probe import feature_vector
    cfg = ModelConfig(vocab_size=590, d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=64)
    torch.manual_seed(0)
    model = MusicTransformer(cfg).eval()
    assert feature_vector(model, list(range(60)), list(range(100)), "cpu") is None
    v = feature_vector(model, list(range(20)), list(range(30)), "cpu")
    assert v is not None and np.isfinite(v).all()


def test_the_pooled_slice_is_never_empty_across_shapes():
    for n_p, n_c, cap in [(10, 10, 64), (100, 50, 120), (500, 60, 600),
                          (1700, 1088, 2048), (5, 8, 32)]:
        fitted = fit_to_context(list(range(n_p)), list(range(n_c)), cap, "cpu")
        if fitted is None:
            continue
        seq, n_prompt = fitted
        sl = slice(n_prompt - 1, -1) if seq.shape[1] > n_prompt else slice(-1, None)
        pooled = torch.zeros(seq.shape[1], 3)[sl]
        assert pooled.shape[0] > 0, (n_p, n_c, cap)
        assert np.isfinite(pooled.float().mean(0).numpy()).all()
