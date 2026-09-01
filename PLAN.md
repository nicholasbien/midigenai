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
- [ ] **Fragment-EOS for the next run**: build the corpus with
      `build_dataset --fragment-under-seconds 30` so short files get BOS but
      no EOS. Why: GigaMIDI is loops/clips (median 27 s, 51% < 30 s, ~24% of
      training docs) and each one teaches "8 bars, then EOS" — ctx4096_ext
      puts P(EOS) > 0.2 on ~35% of random 8-bar Lakh excerpts even with
      natural note-offs. Keeps GigaMIDI's tokens (main drums/multi-track
      source), drops only the ending signal. Check: P(EOS)-at-bar-8 sweep
      (20 random Lakh files, see 2026-09-01 log) should fall well below the
      current 35% / mean 0.18. Inference guard already exists
      (`generate --min-new-tokens`) — the data fix is what removes the cause.

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

- [x] Pilot-scale sweeps (LR, augmentation on/off, block 2048) — settings picked
- [x] Full run at medium (113M): LAUNCHED as medium_full_v1 (24B-token target)
- [ ] Only after a good 100M model: production config (202M)

## Sequencing

1 and 2 first (they change what the long run trains on), 5 in parallel (labels
accumulate while everything else runs), then 4 → 7 → 6.

## Log

- 2026-09-01: Early-EOS diagnosis on ctx4096_ext. Two causes. (a) Prompts
  whose voices all stop at one instant (bar-line cut, truncated note-offs)
  get P(EOS)=0.51–0.93 as the *first* generated token vs 0.02 with natural
  overhanging note-offs — the model correctly imitating corpus endings;
  fixed at inference with `generate --min-new-tokens N` (+ `--seed`), added
  to both backends. (b) Even with natural note-offs, 35% of 20 random Lakh
  files cut at bar 8 had P(EOS) > 0.2 (mean 0.18): an "ends at bar 8" prior
  from GigaMIDI's short clips (`clean.py` has MIN_NOTES=32 but no duration
  floor). Fix queued for the next run: `build_dataset
  --fragment-under-seconds 30` (workstream 2). Sampling note from the same
  measurement: the model is very peaked (median top-10 mass 0.97–1.0,
  top-50 ≥ 0.999) so top-k 50 is a no-op and temperature is the only lever;
  a single-seed t=0.9/1.0/1.2 comparison had the multi-track seed lose its
  lead/bass at 1.2, so 1.0 stays the default.
- 2026-09-01 15:5x UTC: TRAINING COMPLETE — medium_full_v1: 180,000 steps,
  23.6B tokens (~1.8 epochs of the 13.2B-token corpus_full), final val loss
  0.7495 (still descending at cooldown end — extended-run headroom exists via
  WSD resume from ckpt_final). ~12.5h H100 ≈ $55-60 total across all legs
  (two client-kill incidents cost ~4 min of compute combined; deploy+spawn
  pattern eliminated the failure class). Training-evolution listening page
  (13 checkpoint columns, 10 fixed val prompts, piano-roll + notation views,
  fully self-contained HTML) attached to the midigenai-v3 todolist task.
  NEXT: behavior scorecard of ckpt_final vs pilots + v2-100m; blind A/B vs
  v2-100m in the labeling app (the ship/no-ship gate); publish to HF hub as
  v3 on a pass; mac-worker mp3 samples page pending its next session.

- 2026-09-01: CORRECTION — the "clean SIGINT detach" fix was wrong and killed
  the run a second time (kill at step 33,160, ~90s after the SIGINT; the
  "safe to close" verification read the final pre-kill metrics row). In this
  Modal client version, ANY exit of the `modal run --detach` client — clean
  or not — cancels the ephemeral app ~1-2 min later. Correct durable pattern
  now in use: `modal deploy` the app + Function.from_name().spawn() from a
  short-lived client (call fc-01M1DF2P5ZZT7M3TKSHYCBGWQ0, resumed from
  ckpt_033000, ~160 steps lost). Rule going forward: production runs launch
  ONLY via deploy+spawn; health claims ONLY on demonstrated step advancement
  across two timestamped reads, never a single metrics snapshot. Credit:
  todolist worker's pmset/app-log forensics for both root-cause corrections.

- 2026-08-31 (evening): ROOT CAUSE of the overnight run kill (credit:
  todolist worker's pmset forensics): laptop clamshell sleep at 01:45 EDT
  killed the still-attached `modal run --detach` client's TCP abruptly;
  Modal cancelled the "abandoned" ephemeral app 2 min later. --detach only
  survives a CLEAN client exit. Fix applied to the resumed run: SIGINT'd the
  local client (graceful detach), verified app still running (1 task) and
  stepping (32,840+). Lesson operationalized: after any detached launch,
  immediately SIGINT the local client; a 30-min stall/completion watcher now
  monitors the run. Run resumed from ckpt_031000 with ~3 min of compute lost.

- 2026-08-31 ~02:45: PRODUCTION RUN LAUNCHED — medium_full_v1: 113.8M params,
  corpus_full (13.2B tokens, 274 shards: lakh/lamd/aria/gigamidi/curated,
  strict cross-source dedup), 180k steps x 131k tok/step ≈ 24B tokens
  (~1.8 epochs), block 2048, batch 64, compile, lr 4e-4, WSD, aug on,
  doc-start 0.2, stage-local. First steps healthy: loss 6.63 -> 1.87 by step
  440, throughput 321k tok/s and climbing; ETA ~20h ≈ $80. GigaMIDI was
  recovered pre-launch (nested-zip packaging bug -> 273,766 files / 1.07B
  tokens after cross-corpus dedup; fetcher fixed, build guard added).
  Launch authorized by Nicholas via todolist relay ("Agent can launch once
  build is ready"), consistent with in-session instruction. Next: monitor
  val curve, eval_checkpoint scorecards on intermediate checkpoints, blind
  A/B vs v2-100m at cooldown.

- 2026-08-31 (early): PILOT SWEEP COMPLETE — 9 arms at 25M on Modal H100
  (~$0.30/run, ~4 min each at 1.3M tok/s with torch.compile). Val losses:
  best combined recipe (block 2048 + lr 6e-4) 0.822; lr 6e-4 alone 0.887 vs
  baseline 0.906; lr 1e-3 no better; augmentation and doc-start anchoring
  cost ~nothing on val. New eval harness (midigenai/eval_checkpoint.py)
  scored generation BEHAVIOR for every arm: EOS termination now works
  (17-37% self-termination in 1024 tokens vs structurally 0% for v2-100m,
  best at block 2048), repetition drift slightly negative everywhere (no
  degradation-over-length signature), no pathologies in any arm.
  medium_smoke validated 113M at the big-run config: stable, 336k tok/s,
  val 0.760 after just 260M tokens; resume-with-optimizer-state verified at
  this scale. Big-run recipe locked: 113M, block 2048, batch 64, compile,
  lr 4e-4, WSD, aug on, doc-start 0.2, no mixture weights v1, stage-local,
  ~24B tokens ≈ $65-80 / 17-20h. Awaiting corpus_full build + upload.

- 2026-08-30 (night, cont.): Dedup tightened after human audit found false
  positives at 0.5 verified-Jaccard: now 0.65 + low-shingle exemption
  (<64 distinct interval 6-grams never clustered). Strict rerun on the pilot
  manifest rescued 2,415 of 29,437 previously-dropped files (~8% FP rate at
  the old threshold); 27,022 true near-dups still removed. corpus_full
  builds with strict settings tonight; corpus_pilot on Modal retains the
  loose dedup (fine for config comparisons, not for the big run). Audit view
  now serves strict clusters for a second listening pass.

- 2026-08-30 (later): NLL scoring RETIRED as a quality signal — both tails
  human-audited: high-NLL files sound fine (unfamiliarity, not corruption)
  and low-NLL files are just more repetitive (predictability, not quality).
  Quality signal comes instead from: source-level curation via mixture
  weights, GigaMIDI expressiveness/genre metadata (future conditioning or
  weighting), and human preference labels. Open question for pilot scale:
  whether long repetitive files are over-weighted by token-uniform sampling
  (candidate remedy: repetition_rate-aware downweighting or a per-doc
  sampling cap — NOT truncation, which would corrupt EOS semantics).
- 2026-08-30 (late): DECISION — no NLL junk filter. Human audit of the
  worst-scored tail (500-file scoring under v2-100m, median 0.68, p99 2.13)
  found the high-NLL files sound completely fine: the scorer flags
  unfamiliar-to-v2-100m music, not corruption. Heuristic filters + dedup are
  evidently already catching real junk. NLL scoring is retained only for
  (a) a low-end repetition check (loop spam -> per-file token cap, not a
  filter) and (b) future re-scoring with a pilot-corpus-trained checkpoint,
  which removes the old model's style bias. corpus_full is no longer gated
  on threshold selection; the dup-cluster audit remains the one open
  quality gate.

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
