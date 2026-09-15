"""Compare two checkpoints on held-out prompts, without a judge or a human.

RL makes the reward go up — that is what optimising it means, so a rising
reward is not evidence that anything improved. This asks the questions a
rising reward cannot answer:

  * does an *unoptimised* second opinion agree? GRPO trains against one
    reward, so scoring the same samples with the other one is a check the
    policy was never able to game.
  * did the music degenerate? Reward hacking on a taste that likes restraint
    looks exactly like playing less: fewer notes, narrower range, more
    repetition, eventually near-silence that scores well and sounds like
    nothing.

Neither replaces a blind A/B. They catch the failures that would make a blind
A/B a waste of your evening.

    python -m midigenai.compare_ckpt --a base.pt --b runs/grpo/ckpt_001000.pt \\
        --tokenizer tokenizer.json --prompts evals/prompts_heldout -n 80
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch


def _stats(vals: list[float]) -> str:
    if not vals:
        return "     n/a"
    m = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:+8.3f} ±{sd:5.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True, help="baseline checkpoint")
    p.add_argument("--b", type=Path, required=True, help="the trained one")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("-n", type=int, default=80, help="prompts to sample")
    p.add_argument("--samples", type=int, default=2, help="continuations per prompt")
    p.add_argument("--metric-reward", type=Path,
                   default=Path("evals/reward/reward_v4_autolabel.json"))
    p.add_argument("--probe", type=Path, default=None,
                   help="probe spec; scored with checkpoint A, which fitted it")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--device", default="")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    from midigenai.eval import (note_density_hz, pitch_class_entropy, pitch_range,
                                repetition_rate, scale_consistency)
    from midigenai.grpo import GRPOConfig, PromptSpec, _device, load_policy, sample_group
    from midigenai.reward import Reward
    from midigenai.tokenizer import load_tokenizer

    device = _device(a.device)
    tokenizer = load_tokenizer(a.tokenizer)
    spec = PromptSpec(tokenizer)
    files = sorted(a.prompts.glob("*.mid"))[:a.n]
    prompts = [ids for ids in (spec.prompt_ids(f, 256) for f in files) if ids]
    print(f"[compare] {len(prompts)} prompts x {a.samples} samples per checkpoint")

    metric = Reward.load(a.metric_reward) if a.metric_reward.exists() else None
    models = {}
    for name, path in (("A", a.a), ("B", a.b)):
        models[name], _ = load_policy(path, device)
    probe = None
    if a.probe and a.probe.exists():
        # scored with A: the probe's features are a property of the checkpoint
        # that fitted them, so B's own activations would mean something else
        from midigenai.reward_probe import ProbeReward
        probe = ProbeReward(json.loads(a.probe.read_text()), models["A"], device)

    cfg = GRPOConfig(checkpoint=a.a, tokenizer=a.tokenizer, reward=Path("x"),
                     prompts=a.prompts, out_dir=Path("x"), group_size=a.samples,
                     max_new_tokens=a.max_new_tokens, temperature=a.temperature)

    FEATS = {"density_hz": note_density_hz, "repetition": repetition_rate,
             "pitch_range": pitch_range, "scale_consistency": scale_consistency,
             "pitch_class_entropy": pitch_class_entropy}
    res = {k: {"metric": [], "probe": [], "n_tokens": [], "empty": 0,
               **{f: [] for f in FEATS}} for k in ("A", "B")}

    for i, pids in enumerate(prompts, 1):
        for name, model in models.items():
            for smp in sample_group(model, pids, cfg, device, spec):
                r = res[name]
                r["n_tokens"].append(len(smp))
                if len(smp) < 8:
                    r["empty"] += 1
                    continue
                if metric is not None:
                    v = metric.score(tokenizer, smp)
                    if v is not None:
                        r["metric"].append(v)
                if probe is not None:
                    v = probe.score(tokenizer, smp, prompt_ids=pids)
                    if v is not None:
                        r["probe"].append(v)
                try:
                    sc = tokenizer.decode(list(smp))
                    for f, fn in FEATS.items():
                        r[f].append(float(fn(sc)))
                except Exception:
                    pass
        if i % 20 == 0:
            print(f"[compare] {i}/{len(prompts)} prompts", flush=True)

    print(f"\n{'':22s} {'A (baseline)':>18s} {'B (trained)':>18s}   delta")
    rows = ["probe", "metric", "n_tokens", *FEATS]
    for k in rows:
        va, vb = res["A"][k], res["B"][k]
        d = (st.mean(vb) - st.mean(va)) if va and vb else float("nan")
        tag = "  <- optimised" if k == "probe" else ("  <- independent" if k == "metric" else "")
        print(f"{k:22s} {_stats(va)} {_stats(vb)}  {d:+8.3f}{tag}")
    print(f"{'degenerate samples':22s} {res['A']['empty']:18d} {res['B']['empty']:18d}")

    print("\nread it like this: probe up is expected and proves nothing. metric up "
          "(or flat) means\nthe gain survives a reward the policy never saw. density "
          "or range collapsing means\nthe policy found that playing less scores well — "
          "that is the hack, not the goal.")
    if a.out:
        a.out.write_text(json.dumps(
            {k: {kk: (vv if isinstance(vv, int) else
                      {"mean": st.mean(vv), "sd": st.pstdev(vv), "n": len(vv)} if vv else None)
                 for kk, vv in v.items()} for k, v in res.items()}, indent=1))
        print(f"[compare] wrote {a.out}")


if __name__ == "__main__":
    main()
