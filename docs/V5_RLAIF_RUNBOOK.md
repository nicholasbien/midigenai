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

## 2b. Accompaniment pairs, judge, reward  (the second task; same recipe)

Accompaniment seeds must be MULTI-TRACK files. On day one that is the
held-out val set (235 of its 400 files have two live tracks); the Ableton
clip set and the FMA transcriptions are single-track and cannot seed
accompaniment (see "Gap" below).

    python -m midigenai.pairgen --prompts ~/midigenai-v4/evals/prompts_heldout \
      --out evals/autolabel_v5_acc -n 3000 --mode accompany --bars 16 \
      --checkpoint runs/v5/ckpt_final.pt --tokenizer runs/v5/tokenizer.json --label v5
    python -m midigenai.llm_judge label --pairs evals/autolabel_v5_acc/pairs \
      --out evals/autolabel_v5_acc/labels.jsonl --prompt accompany --concurrency 12
    python -m midigenai.probe_layers cache --checkpoint runs/v5/ckpt_final.pt \
      --tokenizer runs/v5/tokenizer.json --labels evals/autolabel_v5_acc/labels.jsonl \
      --out evals/reward/cache/v5_acc_luna_layers.npz --device mps
    python -m midigenai.probe_layers sweep   --cache evals/reward/cache/v5_acc_luna_layers.npz --labels evals/autolabel_v5_acc/labels.jsonl
    python -m midigenai.probe_layers confirm --cache evals/reward/cache/v5_acc_luna_layers.npz \
      --labels evals/autolabel_v5_acc/labels.jsonl --layers <winner> --l2 10 \
      --checkpoint runs/v5/ckpt_final.pt --out evals/reward/probe_v5_acc.json

The accompaniment judge (`accompany` rubric, chosen from the pair meta) was
validated at sol 0.83 / luna 0.72–0.76 on 47 human votes; its probe has no
prior number to gate against, so treat ~0.72 (the judge's own band) as the
floor.

## 5. GRPO — one run, both tasks  (DECISIONS: steps, accompany-frac)

    modal run midigenai/modal_grpo.py --run-name grpo_v5_001 --version v5 \
      --reward probe_v5.json --prompts evals/prompts_pool_v5 \
      --accompany-prompts ~/midigenai-v4/evals/prompts_heldout --reward-accompany probe_v5_acc.json \
      --accompany-frac 0.5 --bars 16 \
      --steps 1000 --prompts-per-step 8 --lr 5e-6 --beta 0.04 --eval-every 25

Each step samples prompts from both tasks at `--accompany-frac`; each task
scores against its own probe (group-relative advantage means the two need
no shared scale); one KL reference anchors both. EVAL is reported per task
— watch both curves. No repetition penalty (user decision); `compare_ckpt`
is the gate afterward, per task.

**Gap:** there is no Ableton accompaniment seed set yet. `ableton_clips.py`
exports single clips; accompaniment needs multi-track windows from whole
sets. A `--arrangements` mode on the extractor (per-project multi-track
windows, same exclusion and dedup) would close it — a couple of hours, and
worth doing before the second v5 run, since Ableton is the target.

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
