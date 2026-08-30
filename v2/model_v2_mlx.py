"""
MLX backend for v2 inference on Apple silicon.

Same architecture as model_v2.MusicTransformer, implemented with mlx.nn so the
GPU pays off for batch-1 decode. PyTorch MPS dispatches one Metal kernel per op
and the launch overhead swamps a 100M model (47 t/s vs 146 on CPU); MLX's lazy
evaluation fuses the step into few kernels (577 t/s decode, 50 ms TTFT at
2048-token prompts, measured on M3 Max, fp16).

Weights are converted from the PyTorch checkpoint at load time — no separate
checkpoint format. Output is token-identical to the PyTorch model under greedy
decoding (verified over 100-token rollouts; max logit diff ~5e-3).

Only imported when the MLX backend is selected, so torch-only platforms never
need mlx installed.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from .model_v2 import ModelConfig


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_base = cfg.rope_base
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, x, kv_cache):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = 0 if kv_cache is None else kv_cache[0].shape[2]
        # traditional=True is interleaved-pair rotation, matching model_v2.apply_rope
        q = mx.fast.rope(q, self.head_dim, traditional=True,
                         base=self.rope_base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=True,
                         base=self.rope_base, scale=1.0, offset=offset)

        if kv_cache is not None:
            k = mx.concatenate([kv_cache[0], k], axis=2)
            v = mx.concatenate([kv_cache[1], v], axis=2)
        new_cache = (k, v)

        mask = "causal" if (kv_cache is None and S > 1) else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.ffn = SwiGLU(cfg)

    def __call__(self, x, kv_cache):
        h, new_cache = self.attn(self.attn_norm(x), kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class MusicTransformerMLX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)

    def __call__(self, ids, kv_caches=None):
        x = self.embed(ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = x @ self.embed.weight.T  # tied output projection
        return logits, new_caches

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> Iterator[int]:
        """Single-batch streaming generator. Same sampling semantics as
        model_v2.MusicTransformer.generate (fp32 logits, temperature, top-k)."""
        cur = mx.array([prompt_ids])
        kv_caches = None
        for _ in range(max_new_tokens):
            logits, kv_caches = self(cur, kv_caches)
            logits = logits[:, -1, :].astype(mx.float32) / max(temperature, 1e-6)
            if top_k is not None and top_k < logits.shape[-1]:
                kth = mx.sort(logits, axis=-1)[:, -top_k]
                logits = mx.where(logits < kth[:, None], -mx.inf, logits)
            next_id = mx.random.categorical(logits)
            token = int(next_id.item())
            yield token
            if eos_id is not None and token == eos_id:
                return
            cur = next_id[None]  # (1, 1); KV cache holds the rest


def load_from_torch_state_dict(state_dict: dict, cfg: ModelConfig,
                               dtype=mx.float16) -> MusicTransformerMLX:
    """Build an MLX model from a torch state dict (as returned by torch.load)."""
    def w(name):
        return mx.array(state_dict[name].float().numpy()).astype(dtype)

    model = MusicTransformerMLX(cfg)
    model.embed.weight = w("embed.weight")
    model.norm.weight = w("norm.weight")
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        block.attn_norm.weight = w(p + "attn_norm.weight")
        block.ffn_norm.weight = w(p + "ffn_norm.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(block.attn, proj).weight = w(p + f"attn.{proj}.weight")
        for proj in ("gate", "up", "down"):
            getattr(block.ffn, proj).weight = w(p + f"ffn.{proj}.weight")
    mx.eval(model.parameters())
    return model
