"""Regression: generation must never run the KV cache past the RoPE table.

Before the fix, a prompt of ~7.7k+ tokens at the 8192 serving context crashed
mid-decode with `Expected size for first two dimensions of batch2 tensor to
be: [B*H, 8192] but got: [B*H, 8193]` and every upload of that file 500'd.
"""
import torch

from midigenai.generate import fit_to_context
from midigenai.model import ModelConfig, MusicTransformer

BOS = 1


def test_short_prompt_untouched():
    ids, n = fit_to_context([BOS, 5, 6, 7], 512, max_seq_len=8192, bos_id=BOS)
    assert ids == [BOS, 5, 6, 7] and n == 512


def test_long_prompt_keeps_tail_and_bos():
    prompt = [BOS, *range(10, 10 + 9000)]
    ids, n = fit_to_context(prompt, 512, max_seq_len=8192, bos_id=BOS)
    assert n == 512
    assert len(ids) + n == 8192
    assert ids[0] == BOS
    assert ids[1:] == prompt[-(8192 - 512 - 1):]  # most recent tokens survive


def test_long_prompt_without_bos():
    prompt = list(range(9000))
    ids, n = fit_to_context(prompt, 512, max_seq_len=8192, bos_id=BOS)
    assert ids == prompt[-(8192 - 512):] and n == 512


def test_huge_max_new_tokens_is_clamped_not_prompt():
    prompt = list(range(1000))
    ids, n = fit_to_context(prompt, 100_000, max_seq_len=8192, bos_id=None)
    assert len(ids) + n == 8192
    assert len(ids) >= 256  # min_prompt_tokens reserved for context


def _tiny_model(max_seq_len: int) -> MusicTransformer:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32,
                      max_seq_len=max_seq_len)
    return MusicTransformer(cfg).eval()


def test_model_generate_stops_at_context_edge():
    m = _tiny_model(max_seq_len=64)
    prompt = torch.randint(2, 32, (1, 60))
    # Without the guard this raised RuntimeError at position 65.
    out = list(m.generate(prompt, max_new_tokens=100, top_k=None))
    # Positions 60..63 each yield a token, and the token sampled at position
    # 63 is returned but never fed back (there is no room for it).
    assert len(out) == 64 - 60 + 1


def test_fitted_prompt_decodes_full_budget_without_error():
    m = _tiny_model(max_seq_len=64)
    long_prompt = list(range(2, 32)) * 10  # 300 tokens >> context
    ids, n = fit_to_context(long_prompt, 16, max_seq_len=64, bos_id=None,
                            min_prompt_tokens=8)
    assert len(ids) + n == 64
    out = list(m.generate(torch.tensor([ids]), max_new_tokens=n, top_k=None))
    assert len(out) == n
