"""
MLX backend for v2 inference on Apple silicon.

Same architecture as model_v2.MusicTransformer, implemented with mlx.nn so the
GPU pays off for batch-1 decode. PyTorch MPS dispatches one Metal kernel per op
and the launch overhead swamps a 100M model (47 t/s vs 146 on CPU); MLX's lazy
evaluation fuses the step into few kernels.

Speed (v2-100m, M3 Max, fp16): 819 t/s decode short-context / 583 at ~1500
tokens; TTFT 50 ms at 2048-token prompts. Three tricks beyond the plain port:
- preallocated KV cache growing in 256-token steps (no per-step concat)
- one-step-ahead pipelining: the host syncs on token n while the GPU already
  runs step n+1 (mx.async_eval)
- converted weights cached as a .safetensors sidecar next to the torch
  checkpoint, so warm loads take ~15 ms instead of ~1.3 s of torch.load

Output is token-identical to the PyTorch model under greedy decoding (verified
over 100-token rollouts; max logit diff ~5e-3).

Only imported when the MLX backend is selected, so torch-only platforms never
need mlx installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from .model_v2 import ModelConfig


class KVCache:
    """Preallocated per-layer cache, grown in fixed steps to avoid per-token
    reallocation and concat traffic."""

    step = 256

    def __init__(self):
        self.k = None
        self.v = None
        self.offset = 0

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        B, H, S, D = k.shape
        prev = self.offset
        if self.k is None or prev + S > self.k.shape[2]:
            new_size = ((prev + S + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, new_size, D), k.dtype)
            new_v = mx.zeros((B, H, new_size, D), v.dtype)
            if self.k is not None:
                new_k[..., :prev, :] = self.k[..., :prev, :]
                new_v[..., :prev, :] = self.v[..., :prev, :]
            self.k, self.v = new_k, new_v
        self.k[..., prev:prev + S, :] = k
        self.v[..., prev:prev + S, :] = v
        self.offset = prev + S
        return self.k[..., :self.offset, :], self.v[..., :self.offset, :]


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

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset
        # traditional=True is interleaved-pair rotation, matching model_v2.apply_rope
        q = mx.fast.rope(q, self.head_dim, traditional=True,
                         base=self.rope_base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=True,
                         base=self.rope_base, scale=1.0, offset=offset)
        k, v = cache.update(k, v)

        mask = "causal" if (offset == 0 and S > 1) else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.ffn = SwiGLU(cfg)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cache)
        return x + self.ffn(self.ffn_norm(x))


class MusicTransformerMLX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)

    def __call__(self, ids: mx.array, kv_caches: list[KVCache]) -> mx.array:
        """kv_caches (one KVCache per layer) are updated in place."""
        x = self.embed(ids)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, cache)
        x = self.norm(x)
        return x @ self.embed.weight.T  # tied output projection

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def _sample(self, logits: mx.array, temperature: float, top_k: int | None) -> mx.array:
        """Same sampling semantics as model_v2: fp32 logits, temperature, top-k."""
        logits = logits[:, -1, :].astype(mx.float32) / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.shape[-1]:
            kth = mx.sort(logits, axis=-1)[:, -top_k]
            logits = mx.where(logits < kth[:, None], -mx.inf, logits)
        return mx.random.categorical(logits)  # (B,)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> Iterator[int]:
        """Single-batch streaming generator, pipelined one step ahead: while
        the host syncs on token n (.item()), the GPU is already computing
        step n+1. Worth ~1.4x over a synchronous loop."""
        if max_new_tokens <= 0:
            return
        caches = self.new_caches()
        logits = self(mx.array([prompt_ids]), caches)
        token = self._sample(logits, temperature, top_k)
        mx.async_eval(token)
        for i in range(max_new_tokens):
            if i + 1 < max_new_tokens:
                next_logits = self(token[:, None], caches)
                next_token = self._sample(next_logits, temperature, top_k)
                mx.async_eval(next_token)
            else:
                next_token = None
            t = int(token.item())
            yield t
            if eos_id is not None and t == eos_id:
                return
            token = next_token


# ---------- weight loading ---------- #

def _assign_weights(model: MusicTransformerMLX, get) -> MusicTransformerMLX:
    model.embed.weight = get("embed.weight")
    model.norm.weight = get("norm.weight")
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        block.attn_norm.weight = get(p + "attn_norm.weight")
        block.ffn_norm.weight = get(p + "ffn_norm.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(block.attn, proj).weight = get(p + f"attn.{proj}.weight")
        for proj in ("gate", "up", "down"):
            getattr(block.ffn, proj).weight = get(p + f"ffn.{proj}.weight")
    mx.eval(model.parameters())
    return model


def load_from_torch_state_dict(state_dict: dict, cfg: ModelConfig,
                               dtype=mx.float16) -> MusicTransformerMLX:
    """Build an MLX model from a torch state dict (as returned by torch.load)."""
    def get(name):
        return mx.array(state_dict[name].float().numpy()).astype(dtype)
    return _assign_weights(MusicTransformerMLX(cfg), get)


def load_model(checkpoint_path: str | Path,
               dtype=mx.float16) -> tuple[MusicTransformerMLX, ModelConfig]:
    """Load from a torch checkpoint, keeping a converted .safetensors sidecar
    next to it so subsequent loads skip torch.load (~1.3 s -> ~15 ms)."""
    checkpoint_path = Path(checkpoint_path)
    dtype_name = str(dtype).rsplit(".", 1)[-1]  # e.g. "float16"
    sidecar = checkpoint_path.with_suffix(f".mlx_{dtype_name}.safetensors")

    if sidecar.exists() and sidecar.stat().st_mtime >= checkpoint_path.stat().st_mtime:
        weights, metadata = mx.load(str(sidecar), return_metadata=True)
        cfg = ModelConfig(**json.loads(metadata["model_config"]))
        model = _assign_weights(MusicTransformerMLX(cfg), weights.__getitem__)
        return model, cfg

    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["model_config"]
    cfg = ModelConfig(**cfg_dict)
    model = load_from_torch_state_dict(ckpt["model"], cfg, dtype=dtype)

    flat = {}
    def flatten(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                flatten(f"{prefix}.{i}", v)
        else:
            flat[prefix] = node
    flatten("", model.parameters())
    try:
        mx.save_safetensors(str(sidecar), flat,
                            metadata={"model_config": json.dumps(cfg_dict)})
    except OSError:
        pass  # read-only checkpoint dir: just skip the cache
    return model, cfg
