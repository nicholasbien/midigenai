"""The reward scorer must reproduce the fit's own feature computation."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from symusic import Note, Score, TimeSignature, Track

from midigenai.reward import Reward
from midigenai.tokenizer import build_tokenizer


def _score(n_notes=64, step=240, pitches=(60, 62, 64, 67)):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    for i in range(n_notes):
        t.notes.append(Note(i * step, step - 20, pitches[i % len(pitches)], 80))
    s.tracks.append(t)
    return s


@pytest.fixture(scope="module")
def spec():
    return json.loads(Path("evals/reward/reward_v3_same_20260914.json").read_text())


def test_matches_reward_align_features(spec):
    """Same continuation, same numbers as the fitting code path."""
    from midigenai.reward_align import FEATURES, drift_vector, feature_vector
    tok = build_tokenizer()
    ids = tok(_score()).ids
    d = Path(tempfile.mkdtemp())
    tok.decode(ids).dump_midi(d / "c.mid")

    r = Reward(spec)
    mine = r.feature_vector(tok, ids)
    theirs = np.concatenate([feature_vector(d / "c.mid"), drift_vector(ids, tok)])
    assert r.features == FEATURES + ["repetition_drift", "density_drift", "pce_drift"]
    assert np.allclose(mine, theirs, atol=1e-6)


def test_score_is_finite_and_ordered(spec):
    tok = build_tokenizer()
    r = Reward(spec)
    sparse = r.score(tok, tok(_score(n_notes=32, step=480)).ids)
    dense = r.score(tok, tok(_score(n_notes=128, step=120)).ids)
    assert sparse is not None and dense is not None
    # the fit's dominant weights are negative on density/entropy: the labeler
    # preferred sparser continuations, so sparse must score higher
    assert sparse > dense


def test_rejects_degenerate_and_unknown_features(spec):
    tok = build_tokenizer()
    r = Reward(spec)
    assert r.score(tok, []) is None
    assert r.score(tok, tok(_score(n_notes=2)).ids) is None
    with pytest.raises(ValueError):
        Reward({**spec, "features": ["not_a_metric"]})


def test_probe_reward_roundtrip():
    """The probe scores from the model's own activations and needs the prompt."""
    from dataclasses import asdict
    import torch
    from midigenai.model import ModelConfig, MusicTransformer
    from midigenai.reward_probe import ProbeReward, feature_vector

    tok = build_tokenizer()
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=2, n_heads=2,
                      d_ff=64, max_seq_len=512)
    torch.manual_seed(0)
    model = MusicTransformer(cfg).eval()
    dev = torch.device("cpu")

    ids = tok(_score()).ids
    prompt, cont = ids[:40], ids[40:120]
    v = feature_vector(model, prompt, cont, dev)
    assert v is not None and v.shape == (cfg.d_model + 1,)
    assert np.isfinite(v).all()
    assert v[-1] <= 0                          # last feature is a mean log-prob

    spec = {"kind": "probe", "checkpoint": "x", "d_model": cfg.d_model,
            "weights": np.ones(cfg.d_model + 1).tolist(),
            "diff_std": np.ones(cfg.d_model + 1).tolist(),
            "heldout_accuracy": 0.7}
    r = ProbeReward(spec, model, dev)
    assert isinstance(r.score(tok, cont, prompt_ids=prompt), float)
    assert r.score(tok, [], prompt_ids=prompt) is None
    with pytest.raises(ValueError):
        r.score(tok, cont)                     # prompt is not optional here
