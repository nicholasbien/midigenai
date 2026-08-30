# v2 model improvement plan

Working plan for getting from `v2-100m` (current live checkpoint) to a meaningfully
better model at the same size, then scaling up. This doc lives in an open PR and is
updated as work lands. Last updated: 2026-08-30.

## Where we are

`v2-100m` (113M, medium config) was trained for ~1B tokens (15k steps × 16 batch ×
4 accum × 1024 block) against an ~8B-token corpus — roughly 0.12 epochs and well
short of even compute-optimal for this size. Known gaps found in a pipeline audit:

- Augmentation (`v2/data/augment.py`) exists but was never wired into training.
- Trained at block 1024 despite the medium config's 2048 context.
- Train/val split hashes file *paths*; dedup is exact-content only, so near-dups
  across Lakh/LAMD leak into val and flatter the loss numbers.
- No resume support in the training loop; cosine schedule locks in `max_steps`.
- Quantitative eval used n=5 prompts.
- Preference collection: all recoverable historical data is 62 records over 38
  A/B pairs of **v1** generations (consolidated in `evals/preferences/`); ~166
  other clicks were lost because old servers didn't log the pair filename. The
  Railway service writes `responses.csv` to ephemeral disk (no volume), so
  anything submitted since the last deploy is lost on restart.

## Strategy

Fix the pipeline and eval trust *before* spending on the long run; collect human
preference data at volume locally; validate programmatic rewards against those
preferences before using them as RL signal. Only after a good full run at ~100M do
we scale to 202M.

## Workstreams

### 1. Training pipeline fixes (prereq for the long run)

- [x] Wire pitch-shift / velocity-jitter augmentation into the training data path
      (on-the-fly in `ShardedTokenStream`, cheap token-level remap)
- [x] Train at block 2048 for medium+ (config default, not just a flag)
- [x] Checkpoint resume (`--resume`) — long Modal runs must survive preemption
- [x] WSD (warmup–stable–decay) LR schedule option so run length can be extended
      without re-committing to a cosine horizon
- [x] Structured metrics logging (CSV/JSONL per step; wandb optional)

### 2. Data quality (current corpus)

- [ ] Near-duplicate dedup: MinHash/LSH over pitch-interval n-grams, applied
      before the train/val split (fixes leakage *and* wasted compute)
- [ ] Stronger heuristic filters in `clean.py`: drum-only files, extreme note
      density (broken quantization), long internal silences, single-pitch spam
- [ ] Perplexity filter: score corpus with the 25M pilot, inspect and drop tails
- [ ] Source-weighted sampling: upsample curated sets (MAESTRO, POP909,
      GiantMIDI) relative to raw LAMD/Lakh
- [ ] Ablate each filter at pilot scale (~$1.50/run) before trusting it

### 3. More data

- [ ] GigaMIDI (~1.4M files, 2025, largest open MIDI corpus; includes
      expressiveness annotations usable as quality signal)
- [ ] Aria-MIDI (~1.2M piano transcriptions, high quality)
- [ ] MetaMIDI / MMD (~436k) and PDMX (~250k public domain)
- [ ] All new sources go through clean + near-dup dedup against existing corpus
- [ ] Later: own audio→MIDI transcription pipeline (unbounded data; heavy lift)

### 4. Evals we can trust

- [ ] Expand fixed eval prompt set to 50–100 held-out prompts (post-dedup)
- [ ] Automatic metric suite runs per checkpoint (extend `eval_v2.py` battery)
- [ ] Human A/B via labeling app (below) as the ground-truth eval

### 5. Human preference data at volume (local labeling app)

- [x] `v2/label_app.py`: local Flask app that *generates pairs live* from the
      current model (hub checkpoint or local), pre-generates a queue in the
      background so labeling never waits, plays via html-midi-player, keyboard
      shortcuts (A / B / tie / both-bad / replay / skip)
- [x] Log everything needed for training: prompt file + token ids, both
      continuations' token ids, model version, sampling params, seed, timestamps
      — JSONL + saved MIDIs, append-only
- [x] Supports cross-model pairs (checkpoint A vs checkpoint B) for eval, and
      same-model pairs for preference/reward data
- [ ] Fix the public site loop separately: Railway volume (or durable store) +
      full metadata logging + v2 serving (site currently serves v1)

### 6. Reward alignment → RL

- [x] `v2/reward_align.py`: compute the metric vector for each continuation,
      fit Bradley–Terry over metric differences on human pairs, report held-out
      agreement per metric and for the fitted reward
- [ ] Gate: only use a programmatic reward for GRPO if it predicts human
      preference clearly above chance on held-out labels; re-fit as labels grow
- [ ] GRPO prototype against the validated reward with meaningful KL penalty;
      A/B every RL checkpoint against its base in the labeling app
- [ ] DPO becomes viable once labeled pairs reach the low thousands; revisit then

### 7. The long run, then scale

- [ ] Pilot-scale sweeps (LR, augmentation on/off, block 2048) — pick settings
- [ ] Full run at medium (113M): target ~8–10B tokens with WSD + resume
- [ ] Only after a good 100M model: production config (202M)

## Sequencing

1 and 2 first (they change what the long run trains on), 5 in parallel (labels
accumulate while everything else runs), then 4 → 7 → 6.

## Log

- 2026-08-30: Workstream 1 implemented (augmentation wired with drum-aware
  pitch shift, WSD schedule, resume with optimizer state, metrics.csv, block
  2048 default on Modal, periodic Modal volume flush). Labeling app
  (`v2/label_app.py`) and reward alignment (`v2/reward_align.py`) implemented
  and smoke-tested end-to-end. First alignment run on the 62 historical v1
  pairs: no single metric beats chance convincingly, but the fitted
  Bradley–Terry combination reaches 0.66 leave-one-out accuracy — right at
  the usability threshold; needs fresh v2-era labels to trust.
- 2026-08-30: Plan created. Historical preference data consolidated into
  `evals/preferences/` (62 records, 38 pairs, v1-era). Confirmed Railway service
  has no volume and no new preference rows since deploy.
