"""
GRPO fine-tuning against a validated reward.

Group Relative Policy Optimization: for each prompt, sample G continuations
from the policy, score them, and use the *group mean* as the baseline instead
of a learned value network (that is the whole difference from PPO — no critic,
half the memory, and a baseline that is unbiased by construction).

    advantage_i = (r_i - mean(r_group)) / std(r_group)
    loss = -mean_i( advantage_i * mean_t log pi(token_t) ) + beta * KL(pi || ref)

The KL term is not decoration: the reward is a 10-feature linear fit that
agrees with the labeler only ~0.65 of the time, so it is trivially gameable
(its largest weights say "sparser and less rhythmically varied is better" —
optimize that without a leash and the model will happily emit almost
nothing). `beta` keeps the policy near the checkpoint that people already
liked, and the run is only believable if a blind A/B against that base
checkpoint prefers the result.

    python -m midigenai.grpo --checkpoint runs/.../ckpt_final.pt \\
        --tokenizer ~/midigenai_data/corpus_full_v4/tokenizer.json \\
        --reward evals/reward/reward_v3_same_20260914.json \\
        --prompts evals/prompts_heldout --steps 200 --group-size 8
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from midigenai.model import ModelConfig, MusicTransformer
from midigenai.reward import Reward
from midigenai.tokenizer import load_tokenizer, normalize_drums


@dataclass
class GRPOConfig:
    checkpoint: Path
    tokenizer: Path
    reward: Path
    prompts: Path
    out_dir: Path
    steps: int = 200
    prompts_per_step: int = 2      # groups per optimizer step
    group_size: int = 8            # samples per prompt
    prompt_tokens: int = 192
    max_new_tokens: int = 192
    temperature: float = 1.0
    top_k: int = 50
    lr: float = 1e-6               # RL on a pretrained LM wants a tiny LR
    beta: float = 0.04             # KL penalty toward the frozen reference
    grad_clip: float = 1.0
    save_every: int = 50
    log_every: int = 1
    seed: int = 0
    device: str = ""


def _device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(path: Path, device: torch.device) -> tuple[MusicTransformer, ModelConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = MusicTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def token_logprobs(model: MusicTransformer, seq: torch.Tensor,
                   n_prompt: int) -> torch.Tensor:
    """log pi(token_t | prefix) for the generated tail of one sequence."""
    logits, _ = model(seq[:, :-1])      # the model returns (logits, kv_caches)
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = seq[:, 1:]
    picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked[:, n_prompt - 1:]          # only the sampled tokens


def sample_group(model, prompt_ids, cfg: GRPOConfig, device) -> list[list[int]]:
    model.eval()
    out = []
    with torch.no_grad():
        for _ in range(cfg.group_size):
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            new = list(model.generate(x, max_new_tokens=cfg.max_new_tokens,
                                      temperature=cfg.temperature, top_k=cfg.top_k))
            out.append(new)
    return out


def train(cfg: GRPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = _device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "grpo_config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))

    tokenizer = load_tokenizer(cfg.tokenizer)
    reward = Reward.load(cfg.reward)
    print(f"[grpo] reward: {len(reward.features)} features, held-out "
          f"{reward.heldout_accuracy:.2f} vs labeler {reward.self_consistency:.2f}")

    policy, model_cfg = load_policy(cfg.checkpoint, device)
    ref, _ = load_policy(cfg.checkpoint, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    print(f"[grpo] policy {policy.num_params()/1e6:.1f}M on {device}")

    from symusic import Score
    files = sorted(cfg.prompts.glob("*.mid"))
    if not files:
        raise SystemExit(f"no prompts in {cfg.prompts}")

    def prompt_of(f: Path) -> list[int] | None:
        try:
            sc = Score(str(f))
        except Exception:
            return None
        normalize_drums(sc, f.name)
        ids = tokenizer(sc).ids
        if len(ids) < 32:
            return None
        if len(ids) > cfg.prompt_tokens:
            ids = ids[:cfg.prompt_tokens]
        return ids

    metrics_path = cfg.out_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("step,reward_mean,reward_std,kl,loss,n_scored,elapsed_s\n")
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        policy.train()
        opt.zero_grad(set_to_none=True)
        step_rewards, step_kls, losses = [], [], []
        groups_used = 0

        for _ in range(cfg.prompts_per_step):
            prompt_ids = None
            for _ in range(10):
                prompt_ids = prompt_of(rng.choice(files))
                if prompt_ids:
                    break
            if not prompt_ids:
                continue
            samples = sample_group(policy, prompt_ids, cfg, device)
            scored = [(s, reward.score(tokenizer, s)) for s in samples]
            scored = [(s, r) for s, r in scored if r is not None and len(s) > 1]
            if len(scored) < 2:
                continue                      # nothing to compare within the group
            rs = np.array([r for _, r in scored], dtype=np.float64)
            adv = (rs - rs.mean()) / (rs.std() + 1e-6)
            step_rewards.extend(rs.tolist())
            groups_used += 1

            n_prompt = len(prompt_ids)
            for (sample, _), a in zip(scored, adv):
                seq = torch.tensor([prompt_ids + sample], dtype=torch.long, device=device)
                lp = token_logprobs(policy, seq, n_prompt)
                with torch.no_grad():
                    lp_ref = token_logprobs(ref, seq, n_prompt)
                # k3 estimator: unbiased, non-negative, low variance
                ratio = (lp_ref - lp).clamp(-10, 10)
                kl = (ratio.exp() - ratio - 1).mean()
                loss = -(float(a) * lp.mean()) + cfg.beta * kl
                (loss / (cfg.prompts_per_step * len(scored))).backward()
                step_kls.append(float(kl.detach()))
                losses.append(float(loss.detach()))

        if losses:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            rm = st.mean(step_rewards)
            rs_ = st.pstdev(step_rewards) if len(step_rewards) > 1 else 0.0
            kl_m = st.mean(step_kls)
            with metrics_path.open("a") as f:
                f.write(f"{step},{rm:.4f},{rs_:.4f},{kl_m:.5f},{st.mean(losses):.4f},"
                        f"{len(step_rewards)},{time.time()-t0:.0f}\n")
            if step % cfg.log_every == 0:
                print(f"[grpo] step {step:4d}  reward {rm:+.3f} (sd {rs_:.3f})  "
                      f"KL {kl_m:.4f}  groups {groups_used}  {time.time()-t0:.0f}s", flush=True)
        else:
            # every sample in every group was rejected by the reward (all
            # degenerate, or too short to score) — no update, but the run must
            # still checkpoint on schedule
            print(f"[grpo] step {step}: no usable groups, no update", flush=True)
        if step % cfg.save_every == 0 or step == cfg.steps:
            out = cfg.out_dir / f"ckpt_{step:06d}.pt"
            torch.save({"model": policy.state_dict(),
                        "model_config": asdict(model_cfg),
                        "grpo_step": step}, out)
            print(f"[grpo] saved {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--reward", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("runs/grpo"))
    for name, typ, default in (("steps", int, 200), ("prompts-per-step", int, 2),
                               ("group-size", int, 8), ("prompt-tokens", int, 192),
                               ("max-new-tokens", int, 192), ("temperature", float, 1.0),
                               ("top-k", int, 50), ("lr", float, 1e-6),
                               ("beta", float, 0.04), ("save-every", int, 50),
                               ("seed", int, 0)):
        p.add_argument(f"--{name}", type=typ, default=default)
    p.add_argument("--device", default="")
    a = p.parse_args()
    train(GRPOConfig(checkpoint=a.checkpoint, tokenizer=a.tokenizer, reward=a.reward,
                     prompts=a.prompts, out_dir=a.out_dir, steps=a.steps,
                     prompts_per_step=a.prompts_per_step, group_size=a.group_size,
                     prompt_tokens=a.prompt_tokens, max_new_tokens=a.max_new_tokens,
                     temperature=a.temperature, top_k=a.top_k, lr=a.lr, beta=a.beta,
                     save_every=a.save_every, seed=a.seed, device=a.device))


if __name__ == "__main__":
    main()
