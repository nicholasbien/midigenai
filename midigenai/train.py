"""
Training loop for v2 music transformer.

Designed for a single H100 (Lambda Labs). Single-GPU only — keeping it simple
since 200M fits comfortably. If we later need multi-GPU, wrap with accelerate.

Data path:
    out_dir/shards/train_*.npy  uint16 token streams
    out_dir/shards/val_*.npy

Each step:
    1. Sample random offset into a concatenated train stream
    2. Slice (block_size + 1) tokens, split into (input, target)
    3. Forward, loss, backward, accumulate, step

Run examples:
    # pilot
    python -m midigenai.train --data /data/v2_corpus --size pilot \\
        --batch-size 16 --grad-accum 4 --max-steps 5000

    # production
    python -m midigenai.train --data /data/v2_corpus --size production \\
        --batch-size 8 --grad-accum 16 --max-steps 200000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from midigenai.model import ModelConfig, MusicTransformer


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    size: str = "pilot"            # "pilot" | "medium" | "production"
    batch_size: int = 16
    grad_accum: int = 4
    block_size: int = 1024
    max_steps: int = 5_000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 2_000
    log_interval: int = 20
    dtype: str = "bfloat16"        # "bfloat16" | "float16" | "float32"
    compile: bool = False
    seed: int = 0
    schedule: str = "cosine"       # "cosine" | "wsd" (warmup-stable-decay)
    decay_steps: int = 0           # wsd only; 0 -> 10% of max_steps
    augment: bool = True           # on-the-fly pitch shift + velocity jitter
    resume: Path | None = None     # checkpoint to resume from


# ----------------------------- data --------------------------------------- #

class ShardedTokenStream:
    """
    Memory-maps every shard once, then samples random (block_size+1)-length
    windows from a uniformly random shard. Shards are concatenations of
    BOS-separated documents — windowing across BOS boundaries is fine
    (it teaches the model to handle context resets).
    """

    def __init__(self, shard_paths: list[Path], block_size: int):
        if not shard_paths:
            raise ValueError("no shards provided")
        self.block_size = block_size
        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths]
        # weight sampling by shard length so all tokens are equally likely
        self.weights = np.array([max(len(s) - block_size - 1, 0) for s in self.shards],
                                dtype=np.float64)
        if self.weights.sum() <= 0:
            raise ValueError(
                f"all shards shorter than block_size+1={block_size+1}; "
                "use a smaller block_size or larger shards"
            )
        self.weights /= self.weights.sum()

    def sample_batch(self, batch_size: int, rng: np.random.Generator,
                     augmenter=None):
        idxs = rng.choice(len(self.shards), size=batch_size, p=self.weights)
        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.block_size), dtype=np.int64)
        for i, si in enumerate(idxs):
            shard = self.shards[si]
            start = int(rng.integers(0, len(shard) - self.block_size - 1))
            chunk = shard[start : start + self.block_size + 1].astype(np.int64)
            if augmenter is not None:
                # augment the full window so input and target stay consistent
                chunk = augmenter(chunk, rng)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]
        return torch.from_numpy(x), torch.from_numpy(y)


def collect_shards(data_dir: Path) -> tuple[list[Path], list[Path]]:
    shards_dir = data_dir / "shards"
    train = sorted(shards_dir.glob("train_*.npy"))
    val = sorted(shards_dir.glob("val_*.npy"))
    if not train:
        raise FileNotFoundError(f"no train shards in {shards_dir}")
    return train, val


# --------------------------- training ------------------------------------ #

def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """
    Warmup-stable-decay: constant LR after warmup, linear decay to min_lr over
    the final `decay_steps`. Unlike cosine, extending a run just means resuming
    with a larger --max-steps before the decay phase starts.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_steps = cfg.decay_steps or max(1, cfg.max_steps // 10)
    decay_start = cfg.max_steps - decay_steps
    if step < decay_start:
        return cfg.lr
    progress = min(1.0, (step - decay_start) / max(1, decay_steps))
    return cfg.lr + progress * (cfg.min_lr - cfg.lr)


def get_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.schedule == "wsd":
        return wsd_lr(step, cfg)
    return cosine_lr(step, cfg)


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def amp_context(device: torch.device, dtype):
    # autocast is CUDA-only here (amp_dtype is fp32 off-CUDA anyway, and MPS
    # rejects the device_type outright)
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: MusicTransformer, stream: ShardedTokenStream,
             iters: int, cfg: TrainConfig, device, dtype) -> float:
    model.eval()
    rng = np.random.default_rng(cfg.seed + 1)
    losses = []
    for _ in range(iters):
        x, y = stream.sample_batch(cfg.batch_size, rng)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_context(device, dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.detach().item())
    model.train()
    return float(np.mean(losses))


def make_optimizer(model: MusicTransformer, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No decay on 1D params (norms, embeddings)
        if p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused)


def train(cfg: TrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.out_dir / "train_config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                  f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map[cfg.dtype] if device.type == "cuda" else torch.float32
    print(f"[train] device={device}  amp_dtype={amp_dtype}")

    # ---- data
    train_shards, val_shards = collect_shards(cfg.data_dir)
    train_stream = ShardedTokenStream(train_shards, cfg.block_size)
    val_stream = ShardedTokenStream(val_shards, cfg.block_size) if val_shards else None
    print(f"[train] train_shards={len(train_shards)}  val_shards={len(val_shards)}")

    # ---- tokenizer (just to pull vocab size)
    vocab_size = json.loads((cfg.data_dir / "manifest.json").read_text())["vocab_size"]

    # ---- model
    if cfg.size == "pilot":
        model_cfg = ModelConfig.pilot(vocab_size)
    elif cfg.size == "medium":
        model_cfg = ModelConfig.medium(vocab_size)
    elif cfg.size == "production":
        model_cfg = ModelConfig.production(vocab_size)
    else:
        raise ValueError(f"unknown size: {cfg.size}")
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, cfg.block_size)
    model = MusicTransformer(model_cfg).to(device)
    print(f"[train] params={model.num_params()/1e6:.1f}M  vocab={vocab_size}")
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # ---- resume
    start_step = 0
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print("[train] WARNING: no optimizer state in checkpoint; "
                  "resuming with fresh AdamW moments")
        start_step = ckpt.get("step", 0)
        print(f"[train] resumed from {cfg.resume} at step {start_step}")

    # ---- augmentation
    augmenter = None
    if cfg.augment:
        from midigenai.data.augment import TokenAugmenter
        from midigenai.tokenizer import build_tokenizer, load_tokenizer
        tok_path = cfg.data_dir / "tokenizer.json"
        tokenizer = load_tokenizer(tok_path) if tok_path.exists() else build_tokenizer()
        augmenter = TokenAugmenter(tokenizer)
        print(f"[train] augmentation on: pitch ±{augmenter.max_pitch_shift}, "
              f"velocity ±{augmenter.max_velocity_jitter} bins")

    # ---- metrics log
    metrics_path = cfg.out_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("step,lr,train_loss,val_loss,tok_per_s,elapsed_s\n")

    def log_metrics(step, lr, train_loss="", val_loss="", tps="", elapsed=""):
        with metrics_path.open("a") as f:
            f.write(f"{step},{lr:.6e},{train_loss},{val_loss},{tps},{elapsed}\n")

    # ---- loop
    rng = np.random.default_rng(cfg.seed + start_step)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        set_lr(opt, lr)

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = train_stream.sample_batch(cfg.batch_size, rng, augmenter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_context(device, amp_dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / cfg.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.detach().item() * cfg.grad_accum
            running_count += 1

        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            tokens_seen = ((step - start_step) + 1) * cfg.batch_size * cfg.grad_accum * cfg.block_size
            tps = tokens_seen / max(elapsed, 1e-3)
            train_loss = running_loss / max(running_count, 1)
            print(f"step {step:6d}  lr {lr:.2e}  "
                  f"loss {train_loss:.4f}  "
                  f"tok/s {tps:,.0f}  elapsed {elapsed:.0f}s")
            log_metrics(step, lr, train_loss=f"{train_loss:.4f}",
                        tps=f"{tps:.0f}", elapsed=f"{elapsed:.0f}")
            running_loss = 0.0
            running_count = 0

        if val_stream is not None and step > 0 and step % cfg.eval_interval == 0:
            vloss = evaluate(model, val_stream, cfg.eval_iters, cfg, device, amp_dtype)
            print(f"step {step:6d}  val_loss {vloss:.4f}  val_ppl {math.exp(vloss):.2f}")
            log_metrics(step, lr, val_loss=f"{vloss:.4f}")

        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(cfg.out_dir / f"ckpt_{step:06d}.pt",
                            model, opt, model_cfg, cfg, step)

    # final save
    save_checkpoint(cfg.out_dir / "ckpt_final.pt",
                    model, opt, model_cfg, cfg, cfg.max_steps)


def save_checkpoint(ckpt_path: Path, model, opt, model_cfg, cfg: TrainConfig,
                    step: int) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {"model": raw_model.state_dict(),
         "optimizer": opt.state_dict(),
         "model_config": asdict(model_cfg),
         "train_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
         "step": step},
        ckpt_path,
    )
    print(f"[ckpt] saved {ckpt_path}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_dir", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--size", choices=["pilot", "medium", "production"], default="pilot")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=5_000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--decay-steps", type=int, default=0,
                   help="wsd only: final decay length (0 = 10%% of max-steps)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="on-the-fly pitch shift ±6 / velocity jitter ±1 bin")
    p.add_argument("--resume", type=Path, default=None,
                   help="checkpoint to resume from (restores optimizer + step)")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
