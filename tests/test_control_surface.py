"""Attribute controls end to end: validation, header building, and the
control-adherence scorecard (mechanism only -- a random tiny model has no
adherence to measure)."""
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai.attributes import (control_vocab, realized_buckets,
                                  validate_controls)
from midigenai.eval_checkpoint import evaluate_checkpoint
from midigenai.generate import Generator
from midigenai.model import ModelConfig, MusicTransformer
from midigenai.tokenizer import build_tokenizer, save_tokenizer


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tok = build_tokenizer(scheme="v4")
    d = tmp_path_factory.mktemp("ckpt")
    save_tokenizer(tok, d / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=512)
    torch.manual_seed(0)
    model = MusicTransformer(cfg)
    torch.save({"model": model.state_dict(), "model_config": asdict(cfg)},
               d / "ckpt.pt")
    return d


@pytest.fixture(scope="module")
def gen(ckpt):
    return Generator(ckpt / "ckpt.pt", ckpt / "tokenizer.json", backend="torch",
                     dtype=torch.float32, inference_seq_len=512)


def _score(n_bars=4, notes_per_beat=1, pitch=60):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    for b in range(n_bars):
        for beat in range(4):
            for k in range(notes_per_beat):
                start = b * 1920 + beat * 480 + k * (480 // max(notes_per_beat, 1))
                t.notes.append(Note(start, 200, pitch + (beat + k) % 12, 80))
    s.tracks.append(t)
    return s


def test_validate_controls_rejects_junk():
    assert validate_controls(density=2, instruments=["Piano", "Drums"]) == {
        "density": 2, "instruments": ["Piano", "Drums"]}
    assert validate_controls(density="1") == {"density": 1}     # form values are strings
    with pytest.raises(ValueError, match="unknown instrument"):
        validate_controls(instruments=["Kazoo"])
    with pytest.raises(ValueError, match="out of range"):
        validate_controls(poly=9)
    with pytest.raises(ValueError, match="integer bucket"):
        validate_controls(density="loud")
    with pytest.raises(ValueError, match="unknown genre"):
        validate_controls(genres=["vaporwave"])


def test_control_vocab_matches_the_buckets_the_builder_uses():
    v = control_vocab()
    assert v["density"] == [0, 1, 2, 3] and v["poly"] == [0, 1, 2]
    assert "Drums" in v["instruments"]
    dense = realized_buckets(_score(notes_per_beat=8))
    sparse = realized_buckets(_score(n_bars=8, notes_per_beat=1))
    assert dense["density"] > sparse["density"]
    assert set(dense) == {"density", "poly", "pitch_range"}


def test_header_can_be_built_from_a_parsed_score(gen):
    inv = {v: k for k, v in gen.tokenizer.vocab.items()}
    score = _score()
    from_score = [inv[t] for t in gen.make_header(score=score)]
    assert any(n.startswith("Density_") for n in from_score)
    assert "Inst_Piano" in from_score

    overridden = [inv[t] for t in gen.make_header(score=score, density=3,
                                                  instruments=["Drums"])]
    assert "Density_3" in overridden and "Inst_Drums" in overridden
    assert "Inst_Piano" not in overridden


def test_control_scorecard_asks_for_every_bucket(ckpt, tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for i in range(2):
        _score(pitch=55 + i * 5).dump_midi(str(prompts / f"val_lakh_p{i}.mid"))

    card = evaluate_checkpoint(str(ckpt / "ckpt.pt"), str(ckpt / "tokenizer.json"),
                               prompts, n_prompts=2, gens_per_prompt=1,
                               prompt_tokens=128, max_new_tokens=24,
                               temperature=1.0, top_k=50, seed=0, mode="control")

    assert card["mode"] == "control"
    # 2 prompts x (4 density + 3 poly + 3 range) buckets x 1 generation
    assert card["n_generations"] == 20
    assert set(card["families"]) <= {"density", "poly", "pitch_range"}
    for family, fam in card["families"].items():
        n = len(fam["confusion"])
        assert all(len(r) == n for r in fam["confusion"])
        assert sum(sum(r) for r in fam["confusion"]) == fam["n"]
        assert 0.0 <= fam["accuracy"] <= 1.0
        assert len(fam["mean_realized_by_request"]) == n
    assert {r["family"] for r in card["rows"]} <= {"density", "poly", "pitch_range"}
    assert {r["requested"] for r in card["rows"] if r["family"] == "density"} == {0, 1, 2, 3}

    repeated = evaluate_checkpoint(str(ckpt / "ckpt.pt"), str(ckpt / "tokenizer.json"),
                                   prompts, n_prompts=1, gens_per_prompt=2,
                                   prompt_tokens=128, max_new_tokens=24,
                                   temperature=1.0, top_k=50, seed=0, mode="control")
    assert repeated["n_generations"] == 20        # 1 prompt x 10 buckets x 2


def test_control_mode_refuses_a_non_v4_checkpoint(tmp_path):
    tok = build_tokenizer(scheme="midilike")
    save_tokenizer(tok, tmp_path / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=256)
    torch.manual_seed(0)
    torch.save({"model": MusicTransformer(cfg).state_dict(),
                "model_config": asdict(cfg)}, tmp_path / "ckpt.pt")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    _score().dump_midi(str(prompts / "a.mid"))
    with pytest.raises(SystemExit, match="needs a v4"):
        evaluate_checkpoint(str(tmp_path / "ckpt.pt"), str(tmp_path / "tokenizer.json"),
                            prompts, n_prompts=1, gens_per_prompt=1,
                            prompt_tokens=64, max_new_tokens=8, temperature=1.0,
                            top_k=50, seed=0, mode="control")


def test_split_then_stitch_restores_the_original(gen):
    """The infill stitch is only correct if the middle spans exactly the bars
    it replaced: put the real middle back and the piece must be unchanged."""
    score = _score(n_bars=6)
    ids = gen.tokenizer(score).ids
    at_bar, n_bars = 2, 2
    prefix, suffix = gen.split_bars(ids, at_bar, n_bars)

    body = [t for t in ids if t not in gen.sp.header_ids and t != gen.bos_id]
    edges = [i for i, t in enumerate(body) if t == gen.bar_id]
    middle = body[edges[at_bar]:edges[at_bar + n_bars]]
    assert gen.count_bars(middle) == n_bars

    stitched = gen.stitch_bars(prefix, middle, suffix, n_bars)
    before = [(n.start, n.pitch) for t in gen.tokenizer.decode(body).tracks
              for n in t.notes]
    after = [(n.start, n.pitch) for t in gen.tokenizer.decode(stitched).tracks
             for n in t.notes]
    assert after == before


def test_stitch_keeps_the_suffix_in_place_whatever_the_model_wrote(gen):
    score = _score(n_bars=6)
    ids = gen.tokenizer(score).ids
    body = [t for t in ids if t not in gen.sp.header_ids and t != gen.bos_id]
    prefix, suffix = gen.split_bars(ids, 2, 2)
    bar, four_four = gen.bar_id, gen.tokenizer.vocab["TimeSig_4/4"]
    tail = sorted((n.start, n.pitch)
                  for t in gen.tokenizer.decode(body).tracks
                  for n in t.notes)[-8:]

    answers = {
        "short": [bar, four_four],                       # stopped after one bar
        "long": [bar, four_four] * 5,                    # ran past the span
        # a meter change inside the span: REMI would re-bar everything after
        # it and pull the suffix earlier
        "wrong meter": [bar, gen.tokenizer.vocab["TimeSig_3/4"]],
    }
    for name, middle in answers.items():
        stitched = gen.stitch_bars(prefix, middle, suffix, 2)
        assert gen.count_bars(stitched) == gen.count_bars(body), name
        got = sorted((n.start, n.pitch)
                     for t in gen.tokenizer.decode(stitched).tracks
                     for n in t.notes)[-8:]
        assert got == tail, name
