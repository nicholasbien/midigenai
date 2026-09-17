# v5 RLAIF runbook

Everything below is a command. Nothing needs a decision except the pool
weighting and the GRPO length, both marked. Total wall-clock after the
checkpoint lands: ~1.5 h of prep, then a ~5–8 h GRPO run.

## 0. The checkpoint
`midigenai-models/v5/ckpt_final.pt` + `tokenizer.json` (vocab **598**).
Nothing from v4's 590-vocab loads against it; every command takes
`--tokenizer` explicitly and `grpo` refuses a vocab mismatch.

    modal run midigenai/modal_grpo.py::checkpoint_identity --version v5
    mkdir -p runs/v5 && modal volume get midigenai-models v5/ckpt_final.pt runs/v5/ \
      && modal volume get midigenai-models v5/tokenizer.json runs/v5/

## 1. Prompt pool  (DECISION: weights)
One pool, one judge, one reward. Default weights: Ableton 0.5, FMA 0.25,
val 0.25 — Ableton highest because it is the target distribution and the
one the old judge was worst on.

    python -m midigenai.prompt_pool --out evals/prompts_pool_v5 -n 1200 \
      --source ableton=evals/prompts_ableton:0.5 \
      --source fma=~/midigenai-v4/evals/prompts_fma:0.25 \
      --source val=~/midigenai-v4/evals/prompts_heldout:0.25

## 2. On-policy pairs  (~70 min for 4,000 at ~1 s/pair)
512-token bar-aligned windows: dense Ableton clips need it, the others do
not mind.

    python -m midigenai.pairgen --prompts evals/prompts_pool_v5 --out evals/autolabel_v5 \
      -n 4000 --mode continue --prompt-tokens 512 \
      --checkpoint runs/v5/ckpt_final.pt --tokenizer runs/v5/tokenizer.json --label v5

## 3. Judge  (luna, fit_only; ~$5 for 4,000; resumable, can run while 2 fills)

    python -m midigenai.llm_judge label --pairs evals/autolabel_v5/pairs \
      --out evals/autolabel_v5/labels.jsonl --concurrency 12

Health to check in the output: tie rate ~0.12–0.22, side balance ~0.5,
swap-consistency ~0.8. A lopsided side balance means it is reading position.

## 4. Reward  (~5 min GPU + minutes of CPU)

    python -m midigenai.probe_layers cache --checkpoint runs/v5/ckpt_final.pt \
      --tokenizer runs/v5/tokenizer.json --labels evals/autolabel_v5/labels.jsonl \
      --out evals/reward/cache/v5_luna_layers.npz --device mps
    python -m midigenai.probe_layers sweep --cache evals/reward/cache/v5_luna_layers.npz \
      --labels evals/autolabel_v5/labels.jsonl
    python -m midigenai.probe_layers confirm --cache evals/reward/cache/v5_luna_layers.npz \
      --labels evals/autolabel_v5/labels.jsonl --layers <sweep winner> --l2 10 \
      --checkpoint runs/v5/ckpt_final.pt --out evals/reward/probe_v5.json

Block 1 is the first guess (it won on the 113M v4), but the sweep decides.
Gate: the confirm number should be at least the 113M v4's **0.789**. Below
~0.75 do not launch.

## 5. GRPO  (DECISION: steps; ~$8 per 1000 steps at 113M with batched rollouts)

    modal run midigenai/modal_grpo.py --run-name grpo_v5_001 --version v5 \
      --reward probe_v5.json --prompts evals/prompts_pool_v5 \
      --steps 1000 --prompts-per-step 8 --lr 5e-6 --beta 0.04 --eval-every 25

No repetition penalty (user decision); `compare_ckpt` is the gate afterward.

## 6. Verify — reward going up is not evidence

    modal volume get midigenai-runs grpo_v5_001/ckpt_001000.pt runs/grpo_v5_001/
    python -m midigenai.compare_ckpt --a runs/v5/ckpt_final.pt --b runs/grpo_v5_001/ckpt_001000.pt \
      --tokenizer runs/v5/tokenizer.json --prompts evals/prompts_pool_v5 -n 80 \
      --probe evals/reward/probe_v5.json --device mps

Then a blind A/B via the labeling hub (pairgen on the pool with
`--checkpoint runs/v5/ckpt_final.pt` vs the GRPO checkpoint through
`label_app --checkpoint-b`, or two pairgen runs merged), and the 512-token
scorecard against the base before anything ships:

    python -m midigenai.eval_checkpoint --checkpoint <ckpt> --tokenizer runs/v5/tokenizer.json \
      --prompts evals/prompts_pool_v5 --max-new-tokens 512 --out evals/scorecards/<name>.json
