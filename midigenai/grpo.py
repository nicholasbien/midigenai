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
from midigenai.tokenizer import is_v4, load_tokenizer, normalize_drums


@dataclass
class GRPOConfig:
    checkpoint: Path
    tokenizer: Path
    reward: Path
    prompts: Path
    out_dir: Path
    # Accompaniment as a second task in the same run. GRPO's advantage is
    # group-relative — (r - mean) / std within eight samples of ONE prompt —
    # so a continuation probe and an accompaniment probe never need a shared
    # scale, and one KL reference (the base) anchors both tasks. Two
    # separate runs would give two checkpoints and no way to serve both.
    accompany_prompts: Path | None = None   # multi-track seeds; None = continuation only
    accompany_frac: float = 0.5             # share of prompts per step that are accompaniment
    reward_accompany: Path | None = None    # probe fitted on accompaniment pairs
    bars: int = 16                          # accompaniment window (training default)
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
    eval_every: int = 10           # 0 disables
    eval_prompts: int = 8
    eval_samples: int = 4
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


class PromptSpec:
    """Prompts and sampling rules for the checkpoint being optimised.

    A v4 checkpoint is never prompted with bare note tokens: production and
    the pair generator both prepend an attribute header describing the music
    (instruments, density, polyphony, range), and both ban SEP / MASK / BOS,
    which are structural tokens and never valid continuation output. GRPO
    that skips this optimises the policy on prompts shaped like nothing it
    will ever see — including the pairs its own reward model was fit on.

    This mirrors `Generator.make_header` without loading a second copy of the
    model: the header only needs the tokenizer and the special-token table.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.v4 = is_v4(tokenizer)
        self.sp = None
        vocab = tokenizer.vocab
        self.eos_id = vocab.get("EOS_None")
        self.ban_ids: list[int] = []
        if self.v4:
            from midigenai.sequence_format import Specials
            self.sp = Specials.from_tokenizer(tokenizer)
            self.ban_ids = [i for i in (self.sp.sep, self.sp.mask, self.sp.bos)
                            if i is not None]

    def header_for(self, path: Path) -> list[int]:
        if not self.v4:
            return []
        from symusic import Score
        from midigenai.attributes import HEADER_PREFIXES, header_for_score
        names = header_for_score(Score(str(path)))
        rank = {pre: i for i, pre in enumerate(HEADER_PREFIXES)}
        names.sort(key=lambda n: rank[next(pre for pre in HEADER_PREFIXES
                                           if n.startswith(pre))])
        return self.sp.header_ids_for(self.tokenizer, names)

    def count_bars(self, ids) -> int:
        return sum(1 for t in ids if t == self.sp.bar) if self.v4 else 0

    def pad_to_bars(self, ids: list[int], n_bars: int) -> list[int]:
        """Generator.pad_to_bars without a Generator: append empty Bar/TimeSig
        pairs so the condition spans the whole window."""
        have = self.count_bars(ids)
        if have > n_bars:
            raise ValueError(f"condition spans {have} bars > {n_bars}")
        vocab = self.tokenizer.vocab
        ts_ids = {v for k, v in vocab.items() if k.startswith("TimeSig_")}
        ts = next((t for t in ids if t in ts_ids), vocab["TimeSig_4/4"])
        return list(ids) + [self.sp.bar, ts] * (n_bars - have)

    def drum_ids(self) -> list[int]:
        from midigenai.pairgen import _drum_token_ids
        return _drum_token_ids(self.tokenizer)

    def accompany_item(self, path: Path, bars: int, rng: random.Random) -> dict | None:
        """One accompaniment prompt, sampled the way training and pairgen
        sample them, laid out exactly as production prompts the model:
        BOS Task_accomp <header> <condition padded to `bars`> SEP.
        Returns prompt_ids, the sampling rules for it, and the task."""
        from symusic import Score
        from midigenai.attributes import header_for_score
        from midigenai.pairgen import sample_accompany_window
        from midigenai.sequence_format import accompaniment_prompt
        if not self.v4:
            return None
        try:
            score = Score(str(path))
        except Exception:
            return None
        got = sample_accompany_window(score, bars, rng, name=path.name)
        if got is None:
            return None
        cond, tgt, header_src, kind = got
        cond_ids = self.tokenizer(cond).ids
        have = self.count_bars(cond_ids)
        n_bars = max(bars, have)
        if have > bars * 2:
            return None
        header = self.sp.header_ids_for(self.tokenizer, header_for_score(header_src))
        ban = list(self.ban_ids)
        if not any(t.is_drum for t in tgt.tracks) and not any(t.is_drum for t in cond.tracks):
            ban += self.drum_ids()             # no uninvited kit, as in pairgen and /api/accompany
        return {"task": "accompany", "prompt_ids": accompaniment_prompt(
                    self.sp, list(header), self.pad_to_bars(list(cond_ids), n_bars)),
                "ban_ids": ban, "bars": n_bars, "kind": kind}

    def trim_leading_bars(self, ids: list[int]) -> list[int]:
        """Drop empty Bar/TimeSig tokens before the first note, as
        Generator.generate_ids does, so RL samples match what pairgen wrote
        and the probe was fitted on."""
        if not self.v4:
            return list(ids)
        vocab = self.tokenizer.vocab
        ts_ids = {v for k, v in vocab.items() if k.startswith("TimeSig_")}
        i = 0
        while i < len(ids) and (ids[i] == self.sp.bar or ids[i] in ts_ids):
            i += 1
        return list(ids[i:])

    def prompt_ids(self, path: Path, max_tokens: int) -> list[int] | None:
        """Header + a prompt slice. The header is never truncated away: it is
        the conditioning, not content."""
        from symusic import Score
        try:
            sc = Score(str(path))
        except Exception:
            return None
        normalize_drums(sc, path.name)
        ids = self.tokenizer(sc).ids
        if len(ids) < 32:
            return None
        head = self.header_for(path)
        room = max(16, max_tokens - len(head))
        return [*head, *ids[:room]]


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


def sample_group(model, prompt_ids, cfg: GRPOConfig, device,
                 spec: "PromptSpec | None" = None, item: dict | None = None) -> list[list[int]]:
    """`group_size` continuations of one prompt.

    `eos_id` and `ban_ids` are not optional decoration: without eos_id the
    stop branch in model.generate is dead code, every sample runs the full
    max_new_tokens, and a sampled EOS is followed by whatever the model emits
    afterwards — which the reward then scores as if it were music.
    """
    model.eval()
    eos_id = spec.eos_id if spec else None
    ban_ids = (spec.ban_ids if spec else None) or None
    max_new = cfg.max_new_tokens
    bar_kw = {}
    if item and item.get("task") == "accompany":
        # accompaniment: its own ban list (drum tokens when no kit was asked
        # for), the token budget accompany() uses, and stop at the window
        ban_ids = item["ban_ids"] or None
        max_new = 64 * item["bars"] + 64
        bar_kw = dict(stop_after_bars=item["bars"], bar_id=spec.sp.bar)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kw = dict(max_new_tokens=max_new, temperature=cfg.temperature,
              top_k=cfg.top_k, eos_id=eos_id, ban_ids=ban_ids)
    with torch.no_grad():
        if hasattr(model, "generate_batch"):
            # the whole group decodes in one pass; generate_batch already
            # drops the EOS it stopped on
            outs = model.generate_batch(x, cfg.group_size, **kw, **bar_kw)
            if bar_kw and spec is not None:
                outs = [spec.trim_leading_bars(o) for o in outs]
            return outs
        out = []
        for _ in range(cfg.group_size):
            new = list(model.generate(x, **kw))
            if eos_id is not None and new and new[-1] == eos_id:
                new = new[:-1]          # a terminator, not content to score
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
    spec = json.loads(cfg.reward.read_text())

    policy, model_cfg = load_policy(cfg.checkpoint, device)
    ref, _ = load_policy(cfg.checkpoint, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    def load_reward(spec_dict, path):
        if spec_dict.get("kind") != "probe":
            r = Reward.load(path)
            print(f"[grpo] reward: {len(r.features)} features, held-out "
                  f"{r.heldout_accuracy:.2f} vs labeler {r.self_consistency:.2f}")
            return r
        from midigenai.reward_probe import ProbeReward
        want_vocab = spec_dict.get("vocab_size")
        if want_vocab is not None and int(want_vocab) != int(model_cfg.vocab_size):
            raise SystemExit(
                f"probe spec {path} was fitted on a {want_vocab}-token vocab; this checkpoint "
                f"has {model_cfg.vocab_size}. Token ids are not interchangeable across "
                "vocabs (corpus_v5 is 598, v4 is 590): refit the probe on this checkpoint.")
        want_sha = spec_dict.get("checkpoint_sha256_8mb"); want_bytes = spec_dict.get("checkpoint_bytes")
        if want_sha or want_bytes:
            import hashlib
            h = hashlib.sha256()
            with open(cfg.checkpoint, "rb") as fh:
                h.update(fh.read(8 << 20))
            if (want_sha and h.hexdigest() != want_sha) or (want_bytes and cfg.checkpoint.stat().st_size != want_bytes):
                raise SystemExit(f"probe spec {path} was fitted on a different checkpoint than "
                                 f"{cfg.checkpoint}: probe features are a property of the "
                                 "checkpoint that produced them")
        r = ProbeReward(spec_dict, ref, device)
        print(f"[grpo] reward {path.name}: probe on {spec_dict.get('layers', ['norm'])} of the "
              f"frozen reference, held-out {spec_dict['heldout_accuracy']:.3f}")
        return r

    rewards = {"continue": load_reward(spec, cfg.reward)}
    if cfg.accompany_prompts is not None:
        if cfg.reward_accompany is None:
            raise SystemExit("--accompany-prompts needs --reward-accompany (a probe fitted on accompaniment pairs)")
        rewards["accompany"] = load_reward(json.loads(cfg.reward_accompany.read_text()), cfg.reward_accompany)
    reward = rewards["continue"]
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    print(f"[grpo] policy {policy.num_params()/1e6:.1f}M on {device}")

    files = sorted(cfg.prompts.glob("*.mid"))
    if not files:
        raise SystemExit(f"no prompts in {cfg.prompts}")
    acc_files = sorted(cfg.accompany_prompts.glob("*.mid")) if cfg.accompany_prompts else []
    if cfg.accompany_prompts is not None and not acc_files:
        raise SystemExit(f"no accompaniment seeds in {cfg.accompany_prompts}")
    spec = PromptSpec(tokenizer)
    print(f"[grpo] prompts: v4={spec.v4}, header={'yes' if spec.v4 else 'n/a'}, "
          f"eos_id={spec.eos_id}, banned={spec.ban_ids or 'none'}")

    def prompt_of(f: Path) -> list[int] | None:
        return spec.prompt_ids(f, cfg.prompt_tokens)

    def next_item() -> dict | None:
        """A prompt for this step: accompaniment with prob accompany_frac."""
        if acc_files and rng.random() < cfg.accompany_frac:
            for _ in range(10):
                it = spec.accompany_item(rng.choice(acc_files), cfg.bars, rng)
                if it:
                    return it
            return None
        for _ in range(10):
            ids = prompt_of(rng.choice(files))
            if ids:
                return {"task": "continue", "prompt_ids": ids}
        return None

    # A held-out set of prompts, fixed for the whole run and scored with a
    # fixed seed. The per-step reward_mean cannot answer "is this working":
    # every step samples different prompts, and prompts differ far more in
    # baseline reward than training moves any one of them (observed -2.8 to
    # -5.4 on consecutive steps of an untrained loop). Group-relative
    # advantage cancels that inside a step; the logged average does not.
    eval_files = [f for f in files[::max(1, len(files) // max(cfg.eval_prompts, 1))]
                  ][:cfg.eval_prompts]
    eval_items = [{"task": "continue", "prompt_ids": ids}
                  for ids in (prompt_of(f) for f in eval_files) if ids]
    if acc_files:
        erng = random.Random(cfg.seed + 1)
        for f in acc_files[::max(1, len(acc_files) // max(cfg.eval_prompts, 1))][:cfg.eval_prompts]:
            it = spec.accompany_item(f, cfg.bars, erng)
            if it:
                eval_items.append(it)

    def eval_reward() -> dict[str, float] | None:
        """Mean reward per task on a fixed prompt set with a fixed seed: a
        paired comparison across steps, reported per task so a gain on one
        cannot hide a loss on the other."""
        if not eval_items:
            return None
        ecfg = GRPOConfig(**{**asdict(cfg), "group_size": cfg.eval_samples})
        ecfg.checkpoint, ecfg.tokenizer = cfg.checkpoint, cfg.tokenizer
        ecfg.reward, ecfg.prompts, ecfg.out_dir = cfg.reward, cfg.prompts, cfg.out_dir
        state = torch.random.get_rng_state()
        torch.manual_seed(cfg.seed)          # same draws every time it is called
        try:
            scores: dict[str, list[float]] = {}
            for it in eval_items:
                pids = it["prompt_ids"]
                for smp in sample_group(policy, pids, ecfg, device, spec, it):
                    r = rewards[it["task"]].score(tokenizer, smp, prompt_ids=pids)
                    if r is not None:
                        scores.setdefault(it["task"], []).append(r)
        finally:
            torch.random.set_rng_state(state)
            policy.train()
        return {k: st.mean(v) for k, v in scores.items() if v} or None

    metrics_path = cfg.out_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "step,reward_mean,reward_std,kl,loss,n_scored,eval_reward,elapsed_s\n")
    t0 = time.time()
    base_eval = eval_reward() if cfg.eval_every else None
    if base_eval is not None:
        print("[grpo] eval reward before training: " + "  ".join(f"{k} {v:+.4f}" for k, v in base_eval.items())
              + f"  ({len(eval_items)} fixed prompts x {cfg.eval_samples} samples)")

    for step in range(1, cfg.steps + 1):
        policy.train()
        opt.zero_grad(set_to_none=True)
        step_rewards, step_kls, losses = [], [], []
        groups_used = 0

        for _ in range(cfg.prompts_per_step):
            item = next_item()
            if not item:
                continue
            prompt_ids = item["prompt_ids"]
            task_reward = rewards[item["task"]]
            samples = sample_group(policy, prompt_ids, cfg, device, spec, item)
            scored = [(s, task_reward.score(tokenizer, s, prompt_ids=prompt_ids))
                      for s in samples]
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
            ev = (eval_reward() if cfg.eval_every and step % cfg.eval_every == 0
                  else None)
            ev_txt = "" if ev is None else "|".join(f"{k}={v:.4f}" for k, v in ev.items())
            with metrics_path.open("a") as f:
                f.write(f"{step},{rm:.4f},{rs_:.4f},{kl_m:.5f},{st.mean(losses):.4f},"
                        f"{len(step_rewards)},{ev_txt},{time.time()-t0:.0f}\n")
            if step % cfg.log_every == 0:
                extra = ""
                if ev is not None:
                    parts = []
                    for k, v in ev.items():
                        d = f" ({v - base_eval[k]:+.4f})" if base_eval and k in base_eval else ""
                        parts.append(f"{k} {v:+.4f}{d}")
                    extra = "  EVAL " + "  ".join(parts)
                print(f"[grpo] step {step:4d}  reward {rm:+.3f} (sd {rs_:.3f})  "
                      f"KL {kl_m:.5f}  groups {groups_used}  "
                      f"{time.time()-t0:.0f}s{extra}", flush=True)
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
                               ("eval-every", int, 10), ("eval-prompts", int, 8),
                               ("eval-samples", int, 4), ("seed", int, 0)):
        p.add_argument(f"--{name}", type=typ, default=default)
    p.add_argument("--device", default="")
    p.add_argument("--accompany-prompts", type=Path, default=None,
                   help="multi-track seeds; enables accompaniment as a second task")
    p.add_argument("--accompany-frac", type=float, default=0.5)
    p.add_argument("--reward-accompany", type=Path, default=None,
                   help="probe spec fitted on accompaniment pairs")
    p.add_argument("--bars", type=int, default=16, help="accompaniment window")
    a = p.parse_args()
    train(GRPOConfig(checkpoint=a.checkpoint, tokenizer=a.tokenizer, reward=a.reward,
                     prompts=a.prompts, out_dir=a.out_dir, steps=a.steps,
                     prompts_per_step=a.prompts_per_step, group_size=a.group_size,
                     prompt_tokens=a.prompt_tokens, max_new_tokens=a.max_new_tokens,
                     temperature=a.temperature, top_k=a.top_k, lr=a.lr, beta=a.beta,
                     save_every=a.save_every, eval_every=a.eval_every,
                     eval_prompts=a.eval_prompts, eval_samples=a.eval_samples,
                     seed=a.seed, device=a.device,
                     accompany_prompts=a.accompany_prompts, accompany_frac=a.accompany_frac,
                     reward_accompany=a.reward_accompany, bars=a.bars))


if __name__ == "__main__":
    main()
