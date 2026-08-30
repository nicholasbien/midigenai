"""
v2 model: small GPT-style decoder-only transformer for music tokens.

Design:
- RoPE positional encoding (extrapolates beyond training length)
- SwiGLU FFN (Llama-style, d_ff ~ 4*d_model)
- RMSNorm (pre-norm)
- SDPA via torch.nn.functional.scaled_dot_product_attention
  (uses FlashAttention-2 on supported hardware automatically)
- Tied embedding/output projection (saves params, fine for next-token loss)

Two configs:
- ModelConfig.pilot()      ~25M params  — validate the pipeline
- ModelConfig.production() ~200M params — full run
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 641
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 4096
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @classmethod
    def pilot(cls, vocab_size: int = 641) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            d_model=512, n_layers=6, n_heads=8, d_ff=2048,
            max_seq_len=1024,
        )

    @classmethod
    def medium(cls, vocab_size: int = 641) -> "ModelConfig":
        # ~113M params: middle ground between pilot (25M) and production (200M).
        return cls(
            vocab_size=vocab_size,
            d_model=768, n_layers=12, n_heads=12, d_ff=3072,
            max_seq_len=2048,
        )

    @classmethod
    def production(cls, vocab_size: int = 641) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            d_model=1024, n_layers=12, n_heads=16, d_ff=4096,
            max_seq_len=2048,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean-of-squares can overflow fp16's max (~65504); compute in fp32.
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)             # (S, head_dim/2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, S, D) — applies RoPE to last dim."""
    # split last dim into pairs: x[..., 2i], x[..., 2i+1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, : x.size(-2), :]
    sin = sin[None, None, : x.size(-2), :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack((rot1, rot2), dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # If we have a KV cache, apply RoPE using the absolute position offset.
        offset = 0 if kv_cache is None else kv_cache[0].size(2)
        cos_slice = cos[offset : offset + S]
        sin_slice = sin[offset : offset + S]
        q = apply_rope(q, cos_slice, sin_slice)
        k = apply_rope(k, cos_slice, sin_slice)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # is_causal only valid when q and k have the same length (training / no cache).
        # During cached single-token decode, q has length 1 and we don't need a mask.
        is_causal = kv_cache is None and S > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class MusicTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        # tied output projection
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.weight.numel()  # head is tied so we only subtract once
        return n

    def _rope(self, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].device != device
            or self._rope_cache[0].dtype != dtype
        ):
            self._rope_cache = precompute_rope(
                self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_base,
                device, dtype,
            )
        return self._rope_cache

    def forward(
        self,
        ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        x = self.embed(ids)
        cos, sin = self._rope(x.device, x.dtype)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cos, sin, cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ):
        """Single-batch streaming generator. Yields token IDs one at a time."""
        self.eval()
        kv_caches: list | None = None
        cur = ids
        for _ in range(max_new_tokens):
            logits, kv_caches = self.forward(cur, kv_caches)
            # fp32 for top-k/softmax so half-precision inference samples cleanly
            logits = logits[:, -1, :].float() / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            yield int(next_id.item())
            if eos_id is not None and int(next_id.item()) == eos_id:
                return
            cur = next_id  # only feed the new token; KV cache holds the rest


def build_model(cfg: ModelConfig | None = None) -> MusicTransformer:
    return MusicTransformer(cfg or ModelConfig())


if __name__ == "__main__":
    for name, cfg in [("pilot", ModelConfig.pilot()), ("production", ModelConfig.production())]:
        m = build_model(cfg)
        n_total = m.num_params()
        n_no_embed = m.num_params(exclude_embedding=True)
        print(f"{name}: total={n_total/1e6:.1f}M  non-embed={n_no_embed/1e6:.1f}M  cfg={cfg}")
