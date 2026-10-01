# Next: picking this up again

Written 2026-09-29, when work paused. What shipped, what to do next, and the
traps that already cost time once. The recipes live in
[V5_RLAIF_RUNBOOK.md](V5_RLAIF_RUNBOOK.md) and [v5.md](v5.md); this is the
order to do things in.

## Where it stands

- **v5-rl is the default model** (served everywhere; `DEFAULT_VERSION` in
  `midigenai/hub.py`, `midigenai/modal_serve.py`). It is the 113M v5 base after
  one GRPO run that trained continuation and accompaniment together
  (`grpo_v5_mixed_001`, step 950).
- **The loop works.** Blind A/B, v5-rl vs v5, models hidden:
  continuation 13–1 (p = 0.001), accompaniment 9–1 (p = 0.01). The same recipe
  took v4 19–4 earlier. The chain that produces this:
  LLM judge (validated against the labeler's own consistency) →
  linear probe on the frozen base's hidden states as the reward →
  GRPO against it, KL-anchored to the base → `compare_ckpt` on a metric
  the policy never saw → blind A/B.
- **The transcription data earned its place.** v5 added 7,009 MuScriptor
  transcriptions of Free Music Archive clips (8 genres the MIDI corpus lacks).
  It is the one corpus change that shows up in the base model (repetition on
  genre prompts −0.07, t −3.1), and the author wants more of it.
- **Serving, site and API state:** [docs/v5.md §4](v5.md#4-serving-and-site).
- **Model size did not matter.** 202M vs 113M was near-even blind. The model is
  data-limited: spend on data, not parameters.

## The plan

### 1. More transcription data (the main lever)

`midigenai/data/transcribe_modal.py` runs MuScriptor (large) on Modal L4s,
about $0.39 per hour of audio, writing to the `midigenai-transcribed` volume.
v5 used `fma_small` (8,000 × 30 s clips).

- **Scale the source.** `fma_large` is 106,574 thirty-second clips across
  161 genres (13x `fma_small`); `fma_full` is the same tracks untrimmed.
  `fma_large` needs no new code. Full tracks would need a change to
  `transcribe_modal.py`: segment into ~20 s windows with a fresh decoder
  state per window (the transcriber drifts on long passes; v5 cut clips at
  20 s for that reason).
- **Filter exactly as v5 did:** `transcript_filter --preset loose
  --max-seconds 20` (tail cut + noise gate). Keep the held-out split:
  carve held-out ids BEFORE building, add them to
  `~/midigenai_data/exclude_ids_all.txt`, and pass `--exclude-ids` on every
  `build_dataset` pass. An empty id list once let 183 of 200 held-out clips
  back into training silently.
- **Tag and weight.** `Source_fma` (or a new `Source_` value if full tracks
  differ enough); v5 weighted transcriptions 15× with `aria:0.5`. With
  10× more data the weight should come down; decide from a short ablation.
- **Licence.** MuScriptor weights are CC BY-NC: keep the output
  non-commercial.
- **Tempo.** 54.5% of `fma_small` carried MuScriptor's 120 BPM placeholder
  and got no `Tempo_` token. The `Tempo_` header turned out inert on v5, so this
  matters less than it looks; fix it with a real beat tracker at
  transcription time only if it is cheap.

### 2. A new base model

Two choices, in this order:

- **v5.x: same 598-token vocabulary, more data.** Retrain the 113M on the
  v5 corpus plus the new transcriptions. No tokenizer change means
  everything downstream (serving, pairgen, probes, GRPO) works unchanged.
  This is the fastest path to "new improved model" and tests the data bet
  directly. Always launch training with `--resume`: a preempted run once
  restarted from step 0.
- **Either way, train more infill.** Keep-some-bars-redo-the-rest is now
  served (`/api/infill`) and is an everyday edit for the author, but the
  builder default is one infill window per file (~9%; check `build_v5.sh`
  for what v5 actually passed) and only interior spans. See
  PLAN.md workstream 10 and the v6 proposal §2 for the data changes; the
  data-only ones need no vocabulary change and can go into v5.x.
- **v6: vocabulary changes.**
  [proposals/v6-role-conditioning.md](proposals/v6-role-conditioning.md)
  queues Role_ tokens for the accompaniment target, fragment EOS,
  one accompaniment window length, grid resolution, and whether to keep
  `Tempo_`. Its gate: v5 shipped (done) and the role approximation audible on
  v5. Do v6 only if that gate passes; otherwise put the budget into data (v5.x).

### 3. Repeat the v5-rl loop on the new base

Everything is in [V5_RLAIF_RUNBOOK.md](V5_RLAIF_RUNBOOK.md). About a day end to
end: ~3 h pairs, ~1 h judge, ~2 h probe fits, 4–8 h GRPO, then listening.

- **Refit both reward probes.** A probe is specific to its checkpoint: re-cache,
  re-sweep the layer (the best layer moved from block 1 on v4 to block 4 on
  v5), confirm with cold leave-one-prompt-out. v5 reached 0.771–0.775
  (continuation) and 0.782 (accompaniment) held-out; do not launch GRPO on a
  continuation probe below 0.75.
- **The accompaniment probe must be refit regardless.** v5-rl's was fitted
  before #57, with ~4–5% of sequences truncated at the wrong offset
  (details in [v5.md](v5.md)).
  Accompaniment decoding has also changed since: answers keep their leading
  empty bars and get a bigger token budget (`accompaniment_budget`), so ~6%
  of samples now enter later and ~10% run longer than the pairs that probe
  was fitted on.
- **Judge:** `gpt-5.6-luna`, rubric `fit_only` for continuation and `accompany`
  for accompaniment (chosen automatically from each pair). Luna is ~17×
  cheaper than sol and fits the same reward. Top up the OpenAI project that
  the key in `.env` belongs to; a top-up to a different project leaves the
  key at zero.
- **One GRPO run, both tasks** (`modal_grpo --accompany-prompts
  --reward-accompany --accompany-frac 0.5`), seeds from val + FMA held-out +
  Ableton clips/arrangements. Build the blind A/B seed set FIRST and keep it
  out of the GRPO pool.
- **Pick the checkpoint** among saved steps (every 50) by summed per-task
  EVAL, then `compare_ckpt`, then the blind A/B through the labeling hub
  (`scripts/labeling_hub.sh`; ~25 decided votes per set is enough).

## Open questions worth an experiment

- **Repetition penalty.** Both GRPO runs raised repetition (0.22 → 0.40) while
  bar-level loops stayed flat, i.e. more groove, not stuck. The author
  declined a penalty so far; revisit if a new run's samples sound stuck.
- **Longer generations.** The judge and GRPO work at ~16 bars. Production can
  run longer; validate the judge on 512-token pairs before training at that
  length.
- **Accompaniment judge on the author's own music.** Validated at 0.74 on the
  held-out val set; on Ableton arrangements it was only ever tested on a model
  that had trained on them (contaminated). Re-validate on the new base with
  ~40 votes before leaning on Ableton accompaniment seeds.
- **Quality tokens** are trained in but unused at inference: try sampling
  from the top quartile.

## Where things live

- **Corpora, manifests, raw data:** `~/midigenai_data/` (not in git).
  Build scripts: `build_v5.sh` there.
- **Modal volumes:** `midigenai-models` (published checkpoints; `v5/`),
  `midigenai-runs` (training and GRPO runs, incl. `grpo_v5_mixed_001`),
  `midigenai-transcribed` (MuScriptor output).
- **Reward artefacts:** `evals/reward/` keeps the probe specs, compare results
  and human labels; `judge_summary.json` has every judge run's headline
  numbers. Per-pair judge outputs are archived in
  `~/midigenai_data/evals_reward_archive/`.
- **The author's votes:** `evals/labeling_*/labels.jsonl` (committed).
- **Labeling hub:** `scripts/labeling_hub.sh`. It uses a free Cloudflare quick
  tunnel whose URL changes whenever it restarts (three times in ten days); a
  named tunnel would give a stable address.
