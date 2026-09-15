"""Batched rollout, v4 prompt construction, and headless pair generation.

These cover the three things that were wrong when GRPO was pointed at a v4
checkpoint: prompts arrived without their attribute header, EOS never stopped
generation, and structural tokens that are never valid output could be
sampled. Plus the batched decode that makes a real RL run affordable.
"""
from dataclasses import asdict
from pathlib import Path

import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai.grpo import PromptSpec
from midigenai.model import ModelConfig, MusicTransformer
from midigenai.pairgen import PairConfig, _slice_with_program
from midigenai.tokenizer import build_tokenizer, is_v4


def _tiny(vocab: int) -> MusicTransformer:
    torch.manual_seed(0)
    return MusicTransformer(ModelConfig(vocab_size=vocab, d_model=32, n_layers=2,
                                        n_heads=2, d_ff=64, max_seq_len=512))


def _midi(path: Path, n: int = 48) -> Path:
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    for i in range(n):
        t.notes.append(Note(i * 240, 220, 60 + i % 8, 80))
    s.tracks.append(t)
    s.dump_midi(path)
    return path


def test_generate_batch_shape_and_length():
    m = _tiny(64)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = m.generate_batch(ids, n=5, max_new_tokens=12, top_k=8)
    assert len(out) == 5
    assert all(len(s) <= 12 for s in out)
    assert all(isinstance(t, int) for s in out for t in s)


def test_generate_batch_never_emits_banned_tokens():
    m = _tiny(64)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    banned = [7, 8, 9]
    out = m.generate_batch(ids, n=6, max_new_tokens=32, top_k=None, ban_ids=banned)
    assert not any(t in banned for s in out for t in s)


def test_generate_batch_stops_at_eos_and_strips_it():
    """With every other token banned, EOS is the only thing left to sample."""
    m = _tiny(32)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    eos = 5
    banned = [i for i in range(32) if i != eos]
    out = m.generate_batch(ids, n=4, max_new_tokens=16, top_k=None,
                           eos_id=eos, ban_ids=banned)
    assert all(s == [] for s in out), out      # stopped immediately, EOS dropped


def test_generate_batch_matches_unbatched_support(tmp_path):
    """Batched and unbatched decode draw from the same distribution: under a
    one-token vocabulary both must produce exactly that token."""
    m = _tiny(32)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    only = 11
    banned = [i for i in range(32) if i != only]
    batched = m.generate_batch(ids, n=3, max_new_tokens=6, top_k=None, ban_ids=banned)
    single = list(m.generate(ids, max_new_tokens=6, top_k=None, ban_ids=banned))
    assert all(s == [only] * 6 for s in batched)
    assert single == [only] * 6


def test_prompt_spec_non_v4_has_no_header(tmp_path):
    tok = build_tokenizer()                     # MIDILike, as v3 uses
    spec = PromptSpec(tok)
    assert not spec.v4 and spec.header_for(_midi(tmp_path / "a.mid")) == []
    ids = spec.prompt_ids(_midi(tmp_path / "b.mid"), 64)
    assert ids and len(ids) <= 64


def test_prompt_spec_v4_prepends_header_and_bans_structural_tokens(tmp_path):
    tok = build_tokenizer(scheme="v4")
    assert is_v4(tok)
    spec = PromptSpec(tok)
    head = spec.header_for(_midi(tmp_path / "c.mid"))
    assert head, "a v4 prompt must carry its attribute header"
    inv = {v: k for k, v in tok.vocab.items()}
    assert all(inv[i].split("_")[0] in {"Inst", "Density", "Poly", "Range",
                                        "Source", "Genre"} for i in head)
    assert spec.ban_ids, "SEP / MASK / BOS are never valid continuation output"
    ids = spec.prompt_ids(_midi(tmp_path / "d.mid"), 64)
    assert ids[:len(head)] == head, "the header must survive truncation"
    assert len(ids) <= 64


def test_prompt_spec_rejects_too_short(tmp_path):
    spec = PromptSpec(build_tokenizer())
    assert spec.prompt_ids(_midi(tmp_path / "e.mid", n=2), 64) is None


def test_slice_restores_the_sounding_program(tmp_path):
    """A window cut after a Program token decodes as the wrong instrument."""
    tok = build_tokenizer()

    class G:
        tokenizer = tok

    prog = next(i for name, i in tok.vocab.items() if name.startswith("Program_"))
    ids = [prog] + list(range(20, 60))
    out = _slice_with_program(G(), ids, start=10, length=8)
    assert out[0] == prog and out[1:] == ids[10:18]


def test_pair_config_defaults_match_the_labelled_corpus():
    """The judge is validated on pairs made with these settings; drifting from
    them puts it out of distribution."""
    cfg = PairConfig()
    assert (cfg.max_new_tokens, cfg.temperature, cfg.top_k) == (256, 1.1, 50)
    assert cfg.prompt_tokens == 256 and cfg.max_cont_seconds == 8.0
