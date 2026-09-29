"""A config names its own scheme, and a mismatched pair must not load quietly.

Reported symptom: "the tokenizer declares 683 tokens, the checkpoint was
trained with 590". The published artifacts were fine -- both 590. The 683
came from build_tokenizer(v4_config()), which ignored the config's scheme
and built MIDILike, producing a v3-scheme vocabulary carrying v4's special
tokens. Nothing failed; it just decoded to the wrong tokens.
"""
import pytest
from miditok import MIDILike, REMI

from midigenai.tokenizer import build_tokenizer, default_config, v4_config


def test_v4_config_builds_remi_not_midilike():
    """The bug: an explicit config always built MIDILike."""
    tok = build_tokenizer(v4_config(8))
    assert isinstance(tok, REMI)
    assert "Bar_None" in tok.vocab and "NoteOn_60" not in tok.vocab


def test_explicit_v4_config_matches_the_scheme_shortcut():
    """build_tokenizer(v4_config()) and scheme='v4' are the same tokenizer."""
    assert len(build_tokenizer(v4_config(8)).vocab) == len(build_tokenizer(scheme="v4").vocab)


def test_v4_config_vocab_is_the_next_rebuild_size():
    """590 is what v4/ckpt_final.pt was trained with; the Tempo_ header
    family (7 buckets) and Source_fma make the next corpus rebuild 598.
    Shipped checkpoints load their own tokenizer.json, so this is the size
    for new runs only; unplanned drift from 598 is the bug."""
    assert len(build_tokenizer(v4_config(8)).vocab) == 598


def test_header_builder_skips_families_an_old_tokenizer_lacks():
    """A 590-token v4 checkpoint predates Tempo_: the newer header builder
    must drop that family for it, not raise, and still refuse a bad name."""
    from midigenai.sequence_format import Specials
    from miditok import TokenizerConfig

    cfg = v4_config(8)
    old = TokenizerConfig(**{**cfg.to_dict(), "special_tokens": [
        s for s in cfg.special_tokens
        if not s.startswith("Tempo_") and s != "Source_fma"]})   # both post-date 590
    tok = build_tokenizer(old)
    assert len(tok.vocab) == 590
    sp = Specials.from_tokenizer(tok)
    inv = {v: k for k, v in tok.vocab.items()}
    ids = sp.header_ids_for(tok, ["Inst_Bass", "Tempo_3", "Density_1"])
    assert [inv[t] for t in ids] == ["Inst_Bass", "Density_1"]
    with pytest.raises(KeyError):
        sp.header_ids_for(tok, ["Inst_Theremin"])


def test_legacy_config_still_builds_midilike():
    tok = build_tokenizer(default_config())
    assert isinstance(tok, MIDILike)
    assert "NoteOn_60" in tok.vocab and "Bar_None" not in tok.vocab


def test_mismatched_tokenizer_and_checkpoint_raise(tmp_path, monkeypatch):
    """A tokenizer from another run must not load silently."""
    import torch

    from midigenai.generate import Generator
    from midigenai.model import ModelConfig
    from midigenai.tokenizer import save_tokenizer

    tok = build_tokenizer(scheme="v4")                  # 590
    tok_path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, tok_path)

    cfg = ModelConfig(vocab_size=len(tok.vocab) + 7,    # a different run
                      d_model=64, n_layers=1, n_heads=2, d_ff=128, max_seq_len=64)
    ckpt = tmp_path / "ckpt.pt"
    from midigenai.model import MusicTransformer
    torch.save({"model": MusicTransformer(cfg).state_dict(),
                "model_config": cfg.__dict__}, ckpt)

    with pytest.raises(ValueError, match="different runs"):
        Generator(ckpt, tok_path, device=torch.device("cpu"))
