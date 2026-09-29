# v5 RLAIF runbook

Everything below is a command. Nothing needs a decision except the pool
weighting and the GRPO length, both marked. Total wall-clock after the
checkpoint lands: ~1.5 h of prep, then a ~5–8 h GRPO run.

## 0. The checkpoint

**Header note for run 1:** `grpo-v4` predates #51, so pairgen/grpo build
headers WITHOUT the Tempo_/Source_fma families v5 trained with — the same
as the header dropout v5 saw, and Tempo_ was measured inert on v5 (drum
density identical under Tempo_1 vs Tempo_5). The probe is fit on those
headers and GRPO rolls out with the same code, so the layout is consistent
end to end. Do NOT merge main into the branch between the probe fit and the
GRPO run; merge after run 1 and refit for run 2.
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

    python -m midigenai.prompt_pool --out evals/prompts_pool_v5_noabl -n 600 \
      --source fma=~/midigenai-v4/evals/prompts_fma:0.4 \
      --source val=~/midigenai-v4/evals/prompts_heldout:0.6
    # run 2 adds --source ableton=evals/prompts_ableton:0.5 (-> evals/prompts_pool_v5)

## 2. On-policy pairs  (~70 min for 4,000 at ~1 s/pair)
512-token bar-aligned windows: dense Ableton clips need it, the others do
not mind.

    python -m midigenai.pairgen --prompts evals/prompts_pool_v5_noabl --out evals/autolabel_v5 \
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

Accompaniment seeds must be MULTI-TRACK files: the user's Ableton
arrangements (`evals/prompts_ableton_arr`, 223 sets from
`ableton_clips.py --arrangements`, model and test lanes excluded, ~175
usable per draw) and the held-out val set (235 of 400 files have two live
tracks). FMA transcriptions are single-track and cannot seed accompaniment.
The 202M has SEEN the Ableton sets (165 of them were in the v4b corpus), so
202M accompaniments on them are partly recall; v5 excludes every Ableton
file and is clean by construction.

    python -m midigenai.pairgen --prompts ~/midigenai-v4/evals/prompts_heldout \   # run 2: evals/prompts_ableton_arr
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

The accompaniment judge (`accompany` rubric, chosen from the pair meta) sits
at sol 0.74 / luna 0.70 on 43/37 human-decided val pairs (after the
condition-track slicing fix; earlier 0.83 was on a subset). Its probe has no
prior number to gate against; expect it below the continuation probe and
treat ~0.68 as the floor, reporting the number rather than hiding it.

## 5. GRPO — one run, both tasks  (DECISIONS: steps, accompany-frac)

**Run 1 has NO Ableton data (user decision, 2026-09-17):** continuation
seeds from `evals/prompts_pool_v5_noabl` (FMA 0.4 / val 0.6), accompaniment
seeds from val `prompts_heldout`. The goal of run 1 is the fastest clean
GRPO-vs-base comparison on v5. Ableton continuation + accompaniment seeds
join run 2 once the accompaniment judge is re-validated on v5 pairs over
`evals/prompts_ableton_arr` (~40 human votes). Substitute the Ableton dirs
below for run 2.

    modal run midigenai/modal_grpo.py --run-name grpo_v5_001 --version v5 \
      --reward probe_v5.json --prompts evals/prompts_pool_v5_noabl \
      --accompany-prompts evals/prompts_accomp_v5_train --reward-accompany probe_v5_acc.json \
      --accompany-frac 0.5 --bars 16 \
      --steps 1000 --prompts-per-step 8 --lr 5e-6 --beta 0.04 --eval-every 25

Each step samples prompts from both tasks at `--accompany-frac`; each task
scores against its own probe (group-relative advantage means the two need
no shared scale); one KL reference anchors both. EVAL is reported per task
— watch both curves. No repetition penalty (user decision); `compare_ckpt`
is the gate afterward, per task.

**Accompaniment seeds** for the mixed run are the Ableton arrangements; add
the val set (`--accompany-prompts` takes one dir — build a pool with
`prompt_pool.py` if both are wanted).

## 6. Verify — reward going up is not evidence

Checkpoints are saved every 50 steps; EVAL runs every 25. Pick the
checkpoint among SAVED steps (900/950/1000), by the sum of per-task EVAL
for a mixed run — a picker over EVAL steps chose 925 once and found no file.

    modal volume get midigenai-runs grpo_v5_001/ckpt_001000.pt runs/grpo_v5_001/
    python -m midigenai.compare_ckpt --a runs/v5/ckpt_final.pt --b runs/grpo_v5_001/ckpt_001000.pt \
      --tokenizer runs/v5/tokenizer.json --prompts evals/prompts_pool_v5_noabl -n 80 \
      --probe evals/reward/probe_v5.json --device mps

**Held-out for the A/B:** `evals/prompts_ab_v5_heldout` (55 val + 25 FMA seeds
removed from the GRPO pool; the 55 val files are also removed from the
accompaniment seeds `evals/prompts_accomp_v5_train`). Run the blind A/B on
THESE, never on pool prompts — otherwise the comparison is on prompts the
policy trained on.

Then a blind A/B via the labeling hub (pairgen on prompts_ab_v5_heldout with
`--checkpoint runs/v5/ckpt_final.pt` vs the GRPO checkpoint through
`label_app --checkpoint-b`, or two pairgen runs merged):

    scripts/labeling_hub.sh   # every evals/labeling_*/ set, one server on :7789

The hub only serves what is on the machine it runs on — pair MIDI is
gitignored, so it has to be the worktree pairgen wrote to. Then the
512-token scorecard against the base, before anything ships:

    python -m midigenai.eval_checkpoint --checkpoint <ckpt> --tokenizer runs/v5/tokenizer.json \
      --prompts evals/prompts_pool_v5_noabl --max-new-tokens 512 --out evals/scorecards/<name>.json
