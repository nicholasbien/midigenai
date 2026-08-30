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

- [x] Near-duplicate dedup: MinHash/LSH over pitch-interval n-grams, applied
      before the train/val split (fixes leakage *and* wasted compute)
- [x] Stronger heuristic filters in `clean.py`: drum-only files, extreme note
      density (broken quantization), long internal silences, single-pitch spam
- [ ] Perplexity filter: score corpus with the 25M pilot, inspect and drop tails
- [ ] Source-weighted sampling: upsample curated sets (MAESTRO, POP909,
      GiantMIDI) relative to raw LAMD/Lakh
- [ ] Ablate each filter at pilot scale (~$1.50/run) before trusting it

### 3. More data

- [x] GigaMIDI fetcher (`--datasets gigamidi`; gated auto-approve — accept terms
      once at huggingface.co/datasets/Metacreation/GigaMIDI); ingestion pending
- [x] Aria-MIDI fetcher (`--datasets aria`, pre-deduped 2GB variant); ingestion pending
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

#### Labeling protocol

Seed prompts live in `evals/prompts/` (84 files: real user uploads + curated
Lakh validation picks). Sessions:

1. **Calibration set** (first ~2 sessions, ~150–200 pairs): same-model pairs
   from the live checkpoint (`python -m v2.label_app --prompts evals/prompts`).
   10% of serves are blind repeats (`--dup-rate`) to measure self-consistency —
   the ceiling any reward fit can reach. Historical labels were only ~0.65
   self-consistent; listen to both sides fully before voting to push this up.
2. **Ongoing** (~50 pairs/week): keeps the reward fit fresh; switch to
   cross-model mode (`--hub-version-b` / `--checkpoint-b`) whenever a new
   checkpoint needs a verdict.
3. **Post-RL gate**: after any GRPO/DPO checkpoint, a blind cross-model session
   of new-vs-base is the only accepted evidence of improvement.

#### Reward calibration gates (checked by `v2.reward_align`)

- Fit/eval split is by *prompt* (leave-one-prompt-out), never by pair.
- Ship a reward to GRPO only if, on ≥150 v2-era pairs: held-out accuracy ≥0.65,
  within ~0.05 of measured self-consistency, and the calibration table tracks
  the diagonal (no wild overconfidence at the extremes).
- Re-fit and re-check every ~100 new labels; after each RL run, re-validate
  on-policy — if the reward prefers the RL checkpoint but the human doesn't,
  the reward is being hacked: refit including the new labels before continuing.

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

- 2026-08-30 (night): Focus shifted to nailing quality of corpus_pilot before
  scaling data. Large downloads PAUSED mid-flight (gigamidi complete; lamd
  3.1/9.2GB, aria 0.6/5.4GB partial — resume for the big Modal run with:
  `DATA_ROOT=~/midigenai_data OUT=~/midigenai_data/corpus_full
  SOURCES="lakh maestro pop909 giantmidi lamd gigamidi aria"
  bash midigenai/data/build_pilot_corpus.sh` — downloads resume, clean
  manifests reuse). New quality tooling: explorer "dup clusters (audit)"
  plays real dedup clusters side by side (18,638 clusters found in the pilot
  corpus — audit for false positives); midigenai/score_corpus.py scores files
  by NLL under a checkpoint and "quality extremes (scored)" plays both tails
  (500-file scoring run against v2-100m in progress). corpus_pilot uploading
  to Modal volume for the pilot sweep.

- 2026-08-30 (evening): Post-merge (#11 + #13 rename). New on data-improvements
  branch: track-view sampling in build_dataset (full mix + up to 2 solo-track
  docs per multi-track file — matches how users jam single lines); EOS token
  appended to every training doc (docs previously had NO end marker: the model
  had never seen a piece end, a direct cause of 'generations run until the
  token cap'); normalize_drums() promotes mislabeled drum tracks (Ableton
  exports, v1-era files with drums on piano channels) before tokenizing, wired
  into build/label/explore paths; explorer shows per-sample pipeline +
  augmentation views. Further data ideas queued in workstream 2/3: trim
  leading/trailing silence at build time; per-source mixture weights; GigaMIDI
  expressiveness CSV + genre tags as quality/conditioning signal; genre or
  instrumentation control tokens; cap token share of very long repetitive
  files; tempo-stretch augmentation; user uploads + live-jam captures as a
  high-weight domain-adaptation set.

- 2026-08-30 (later still): Model scope decision: GENERAL — drums, mono, poly,
  chords all in scope. clean.py drum-only filter removed (drum files stay in
  training); culled drum prompts restored to evals/prompts. Workstream 3:
  verified sources + fetchers for GigaMIDI V2 (5.5GB zip, gated auto-approve)
  and Aria-MIDI (2GB pre-deduped tarball); PDMX deferred (MusicXML conversion).
  New v2/data/explore_app.py (port 7790): browser sampler for candidate
  corpora — GigaMIDI via parquet row-group range-reads, Aria/Lakh via
  streaming-tar sampling, playback + metadata + note stats per file. Labeling
  UX: continuation-only playback (decode with prompt context, trim at the
  boundary) — much faster to review; also purifies reward metrics. Parallel
  session's PR #13 renames v2/ -> midigenai/; plan is to merge this PR (#11)
  first, then the rename stack resolves via git rename tracking.

- 2026-08-30 (later): Workstream 2 core landed: v2/data/dedup.py (MinHash/LSH
  over pitch-interval 6-grams, transposition/velocity/quantization-invariant;
  19/20 synthetic near-dups caught on a 420-file test, ~0 false positives) is
  wired into setup_lambda.sh between clean and build_dataset; clean.py gained
  drum-only / single-pitch / density / long-silence filters with per-reason
  drop accounting. Labeling UX from user feedback: continuations cut 512->256
  tokens, per-side "degrades"/"too long" flags (q/w/e/r) recorded in labels,
  drum-only seeds culled from evals/prompts (78 remain). reward_align gained
  drift features (2nd-half minus 1st-half repetition/density/entropy from the
  continuation tokens) so degradation is directly optimizable once v2 labels
  exist. NOTE: repo working tree is shared with a parallel session
  (live-session branch); this branch now works out of a git worktree at
  ~/midigenai-improve.

- 2026-08-30: Labeling protocol + calibration gates added. label_app gained
  blind repeat serving (--dup-rate); reward_align now does leave-one-prompt-out
  CV, self-consistency measurement, and a calibration table. On historical v1
  data: reward 0.66 held-out vs 0.65 labeler ceiling — the old labels are the
  bottleneck, confirming fresh careful labels are the lever. evals/prompts/
  seeded with 84 held-out prompt MIDIs.

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
