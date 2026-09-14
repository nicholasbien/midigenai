# v4 plan: structure, control, infilling, longer context

**Status**: approved 2026-09-13, in progress on branch `v4-tokenizer`.
`v3` = the `medium_full_v1` checkpoint (113M, MIDILike, 24B tokens), shipped
2026-09-02 (PR #33); v4 is the next full retrain.

## Progress log

- 2026-09-13: Phase 1 code landed on `v4-tokenizer` (tokenizer scheme,
  attribute header, three document formats, header re-injection + dropout in
  training, generator accompany/infill/bar-stop, structure + accompaniment
  evals, 21 tests). Pilot corpora building locally: `corpus_pilot_v4`
  (full mix), then `corpus_pilot_v4_cont` (continuation-only, arms B/D) and
  `corpus_pilot_v4_24` (24 positions/beat, arm C). Quality-labeling app +
  predictor in progress on branch `quality-labeling` (separate PR).
  Decisions made while implementing, all reflected in the code:
  - **Rests off** in v4: REMI with rests collapses empty bars into `Rest`
    tokens and loses the Bar count; an empty bar is now `Bar TimeSig`.
  - **Meter set widened** to 2/4 3/4 4/4 1/4 5/4 6/4 7/4 3/8 5/8 6/8 7/8
    9/8 12/8 2/2 3/2 4/2: a 4/4-3/4-6/8 set skipped 14% of multi-track
    Lakh/LAMD files (1/4 pickup bars are the main offender); the wide set
    skips ~3%. Anything else is dropped, never silently re-barred.
  - **Every segment is padded to its exact bar count** with empty
    `Bar TimeSig` pairs (REMI emits nothing after the last note), so both
    sides of SEP always agree and "stop after N bars" is exact. Windows use
    the file's own downbeats and restate the active meter at tick 0.
  - **Header re-injection at training time**: documents are ~20k tokens and
    windows 2k, so a mid-document window would never see its header. The
    training stream copies the document's header to every window
    (`BOS header body`), then applies dropout (per family 0.3, whole header
    0.1). The val stream injects without dropout.
  - **`Quality_0..3` header tokens** reserved now (4 vocab entries) for the
    human-rated quality predictor, plus `--quality` in the builder, which
    also splits train shards per bucket so `--mixture` can weight them.
  - **Arm B = arm D's corpus trained with `--header-drop-all 1.0`** (no
    separate no-header corpus needed).
  - Vocab: 588 tokens (v4), 908 (v4-24: Position and Duration both grow).
  - **Arm A baseline measured** (`pilot_best`, MIDILike 25M, same recipe the
    v4 arms use; `evals/scorecards/pilot_best_v4metrics.json`): downbeat
    alignment of continuations 0.20 vs 0.31 for their prompts (delta -0.12),
    first-onset bar offset 0.76 beats, EOS rate 0.07, prompt coherence 0.65.
    That is the off-the-downbeat symptom quantified; v4 arms must move the
    delta toward 0 and the bar offset toward 0.
  - **First pilot results** (25M, 5000 steps, block 2048, lr 6e-4; prompts
    cut at note boundaries, `evals/scorecards/`): arms D (continuation-only
    corpus, header) and E (full mix) both fix the structure metrics:
    downbeat delta -0.03 vs -0.13 baseline, first-note bar offset 0.66-0.70
    vs 0.85, and 0.30-0.37 when the prompt is closed to a bar line
    (`Generator.close_bar`). Coherence / scale consistency level with the
    baseline. D vs E isolated a problem: E self-terminates 47-57% of the time
    within 1024 tokens vs 17% for D and 13% baseline, because accompaniment /
    infill targets ended with EOS after a fixed window. Fix: segment targets
    carry no EOS (`--segment-eos` to restore); BOS added to the generator's
    stop set. Accompaniment mode on E: exactly-N-bars 58%, pitch-class
    overlap with the condition 0.23 vs 0.46 for the real parts. D (header)
    vs B (no header), same corpus: val loss 0.800 vs 0.820 at step 4500.
  - **Correction (arm E', no segment EOS)**: the early-stop rate did not
    move (42% / 60% closed-bar). Counting which stop token ended each
    generation: true EOS only 6/60 (below baseline); the rest were SEP (10-15),
    MASK, BOS. So it is task ambiguity, not EOS: a continuation prompt closed
    to a bar line looks like an accompaniment condition. Fix: `Task_accomp` /
    `Task_infill` header tokens open segment documents (never dropped), and
    the generator bans SEP/MASK/BOS from sampling in continuation mode
    (`ban_ids`, both backends). Vocab 590 (v4) / 910 (v4-24). Segment
    targets stay EOS-free (harmless; BOS is a fine terminator). Arm E'' on a
    rebuilt corpus validates; arm C (24/beat, pre-task-token corpus) is
    still a fair grid comparison against E'.
  - **Grid decision: 8/beat for v4.** Fair comparison (both scored with the
    sampling ban; paired per prompt, n=60): C (24/beat) vs E' (8/beat):
    prompt coherence -0.07 (plain) / -0.12 (closed bar, significant),
    repetition -0.07 / -0.09 (significant, but n-gram repetition on a finer
    grid is partly a metric artifact), scale consistency and bar offset
    equal. Per the rule "24 unless a clear loss", production is 8/beat.
    24/beat stays the first thing to retest at medium scale (a ~2B-token
    medium pair, ~$7 each), since the coherence gap may be a 25M/330M-token
    capacity effect. Also: with the ban, E' early stopping is 13% (=
    baseline) and repetition/coherence are at or above baseline, so the
    launch gate's stopping criterion is met before E''.
  - **Arm E'' (task tokens, 4 windows): gate met.** Early stop 18% / 23%
    closed-bar, coherence 0.62 (baseline 0.63), scale consistency 0.90,
    downbeat delta -0.03 / -0.01, bar offset 0.60 / 0.30. Accompaniment
    unchanged (52% exact bars, overlap 0.22). Watch item: corpora with
    segment docs run ~0.05-0.10 higher on 4-gram repetition than
    continuation-only (D 0.165 vs E/E'/E'' 0.21-0.27). **Production launch
    armed** (`~/midigenai_data/launch_v4_full.sh`): corpus_full_v4 (8/beat,
    task tokens, <30 s = fragment), 113M medium, batch 64, block 2048,
    180k steps, lr 4e-4, WSD, compile, stage-local, `--mixture aria:0.5`,
    header dropout 0.3/0.1, via modal_launch deploy+spawn.
- 2026-09-13 18:22: **PRODUCTION RUN LAUNCHED** — `v4_full`, call id
  fc-01M2EDVG38CS61VYKVXS1Z4SPW. corpus_full_v4: 392 shards / ~19.6B tokens
  (lakh 77, gigamidi 36, lamd 143, aria ~130, curated 3 shards), uploaded as
  gzip (-1, 4x smaller; `modal_train` inflates at staging) after a 1 MB/s
  link made the raw 36 GB impractical. Step 2000 at 18:38, 374k tok/s, loss
  0.96; ETA ~12:00 on 09-14. Day's lessons, all fixed in code/scripts:
  Pool.imap hangs forever when a worker dies (per-file SIGALRM + stall guard
  in build_dataset); a restart script must regenerate its inputs; one big
  `modal volume put` wedges after a sleep/network change (per-file resumable
  puts); macOS xargs -I has a 255-byte replacement limit (helper script).

- 2026-09-14 02:40: **Corpus bug found while labeling, fixed for the NEXT
  build (not this run).** `normalize_drums` matched its name hints as bare
  substrings, so "909"/"808"/"hat"/"tom" hit inside arbitrary ids and song
  titles. Damage, measured on the strict manifest:
  - **~9,900 single-track files force-promoted to drums by FILENAME**
    (aria 6,454 = 0.80% of aria, gigamidi 2,790, lamd 443, lakh 180). Aria
    ids are bare numbers, so `785909_0.mid` read as a TR-909 — piano
    transcriptions trained as drum kits.
  - **~0.6% of pitched tracks in multi-track files promoted by TRACK NAME**
    (sampled 2,979 tracks: 32 promoted, 19 of them false — "Beatles 1",
    "CT5909-FLY AWAY FROM HERE", "WHATTOOK").
  Fix (`name_says_drums`): unambiguous hints still match anywhere;
  drm/kit/hat/tom/beat/808/909 require a non-alphanumeric boundary. False
  promotions by filename drop to 0 across all 1.44M files, genuine names
  ("Drums", "HiHat", "TR-909", "909_kit") still match, regression test on
  `val_aria_909098_0.mid`. Present in v3's corpus too, so not a v4
  regression and it does not invalidate the v3-vs-v4 comparison — but
  **`corpus_full_v4` must be rebuilt with this fix before any further run.**
  - **Arm C is 24/beat, not 12**: measured off-grid share and error per
    source (Lakh/LAMD ~3 ms median error at 1/8 beat; Aria/MAESTRO/POP909
    15-23 ms, i.e. nearly every onset off-grid). At 1/24 beat the performed
    sources are within 5-7 ms, below the perceptual threshold, for ~200
    extra Position/Duration tokens. 24 becomes the default unless the pilot
    shows a clear loss.

## Why

Against the symbolic-music SOTA (Music Transformer, MuseNet, Anticipatory Music
Transformer, MuseCoco, REMI-family models) midigenai wins on one axis, latency
(50 ms TTFT, ~800 tok/s streaming), and lags on four:

| Gap | Symptom today | Who solved it |
|---|---|---|
| No bar/beat structure in the vocab | answers land off the downbeat; bar length enforced by a beat budget in `jam.py`, not by the model | REMI, Compound Word, MuseNet |
| Continuation only, no infilling / accompaniment | can't "harmonize what I just played", "add drums under these chords", or "redo bar 3" | Anticipatory Music Transformer, FIM in code LLMs |
| No control tokens | no way to ask for sparse vs dense, a genre, an instrument set; only the prompt steers | MuseNet (composer/instrument tokens), MuseCoco, FIGARO |
| 2048-token context | a few minutes of dense material; long-form structure drifts | Perceiver AR, long-context LLM recipes |

Text conditioning is a fifth gap, but it falls out of control tokens for free
(see phase 3), so it is not a training-time goal.

## Constraints that shape the plan

- **Every vocab change forces a full retrain** (~$80, ~20 h at 113M on Modal).
  So all tokenizer changes ship together as one "v4 tokenizer", validated by
  25M pilots (~$0.30, ~4 min each) first. One production retrain, not four.
- **Latency is the product.** Nothing here may add more than ~50 tokens to a
  jam prompt or change the single-stream autoregressive decode. Multi-codebook,
  encoder-decoder, and diffusion designs are out.
- **Val loss is not comparable across vocabs.** Pilots are judged on behavioral
  evals (`eval_checkpoint.py`) plus new structure/control metrics, then blind
  A/B in `label_app`.
- **Live simultaneous accompaniment is a causality problem, not a modeling
  problem**: the model cannot harmonize a note before the user plays it.
  Infilling targets call-and-response ("now add a bass line under that") and
  offline arrangement, not playing along in the same bar.

## Phase 0: ship v3 (prereq)

- [x] v3 shipped (PR #33, 2026-09-02). `v3` is the MIDILike baseline every v4
      pilot is compared against.
- [ ] Blind A/B `v3` vs `v2-100m` in `label_app` (~50 pairs) to record the
      baseline number the v4 gate is measured against.
- [ ] Freeze the 50–100 held-out eval prompt set (PLAN.md workstream 4) so v4
      numbers are on the same prompts.

## Phase 1: the v4 tokenizer (one design, pilot-ablated)

Code: `tokenizer.py` (scheme), `attributes.py` (header), `sequence_format.py`
(document layouts), `data/v4_docs.py` + `data/build_dataset.py --scheme v4`
(documents), `train.py` (header re-injection/dropout). `model.py` needs only
the new `vocab_size`.

Pilot launch (25M, H100, ~2500 steps at block 2048, lr 6e-4 per the v2
sweep), after `modal volume put openmusenet2-v2-corpus <local> /<name>`:

```
modal run midigenai/modal_train.py --size pilot --compile --block-size 2048 \
    --lr 6e-4 --max-steps 2500 --corpus corpus_pilot_v4 --run-name v4_pilot_E
# arm B: --corpus corpus_pilot_v4_cont --header-drop-all 1.0
# arm D: --corpus corpus_pilot_v4_cont
# arm C: --corpus corpus_pilot_v4_24
# arm F: --resume-from v4_pilot_E/ckpt_final.pt --block-size 4096 --rope-base 50000 --max-steps 250
```
Score each with `eval_checkpoint.py` (continue mode) and `--mode accompany`.

### 1a. Bar + Position tokens (REMI-style)

- MidiTok `REMI` with the same `beat_res` as today, so microtiming resolution
  is unchanged; only the time representation moves from relative `TimeShift`
  to `Bar` + `Position_x`. Keep `use_programs`, 32 velocity bins,
  `use_tempos=False`.
- `use_time_signatures=True` restricted to {4/4, 3/4, 2/4, 6/8}; everything
  else is dropped at clean time (already a minority of the corpus).
- Rider, pilot only: `beat_res` 12/beat to make triplets representable.
  Adopt only if it beats 8/beat on the downbeat and listening evals.
- Inference win this unlocks: `jam.py` stops the answer at the Nth `Bar`
  token instead of a beat budget, and appends a `Bar` token to the prompt to
  force the answer to start on the downbeat. Replaces the transport-sync
  approximation in the playback layer.

### 1b. Attribute control tokens (MuseCoco/FIGARO-style, no paired text)

Every training document gets a short header computed from the MIDI itself, so
no labels are needed:

| Token family | Values | Source |
|---|---|---|
| `Inst_*` | set of GM program families present, `Drums` flag | file |
| `Density_*` | 4 buckets of notes per bar | file |
| `Poly_*` | 3 buckets of mean simultaneous notes | file |
| `Range_*` | 3 buckets of pitch range | file |
| `Genre_*` | ~16 tags | GigaMIDI metadata only; token omitted elsewhere |
| `Source_*` | maestro / pop909 / giantmidi / lakh / lamd / aria / gigamidi | manifest |

- Header dropout at 30% per token family during training (classifier-free
  guidance style), so the unconditional model stays intact and inference can
  supply any subset. At inference the header is ~6–12 tokens.
- `Source_*` doubles as a quality knob: prompting with `Source_maestro`
  or `Source_pop909` biases toward curated material, which is the
  "source-weighted sampling" goal from PLAN.md workstream 2 without
  changing the data mix.
- Vocab impact: roughly +60 tokens.

### 1c. Infilling and accompaniment documents (FIM-style, bar-aligned)

Three document formats mixed in the build, all in one stream and one vocab:

1. **Continuation** (60%): `[header] [music] EOS`. Today's format.
2. **Accompaniment** (25%): `[header] [conditioning tracks, N bars] SEP
   [remaining tracks, same N bars] EOS`. Built from multi-track files by
   picking 1–2 tracks as the condition and 8 or 16 bars as the window. This
   directly extends the track-view sampling already in `build_dataset.py`.
3. **Span infill** (15%): `[header] [bars 1..i] MASK [bars j..end] SEP
   [bars i..j] EOS`. Bar-aligned because 1a gives us bar boundaries; span
   length 1–4 bars.

Bars are counted in `Bar` tokens, so the same bar-counting logic serves the
jam stop condition and the infill span builder. This is FIM rather than the
Anticipatory Music Transformer's arrival-time interleave: FIM needs no new
event semantics, keeps decode identical, and covers the call-and-response
cases; anticipation's streaming lookahead is only valuable for simultaneous
play, which causality rules out anyway.

New special tokens: `MASK`, and `SEP` is repurposed as the condition/target
boundary. Vocab impact: +1.

### 1d. Longer context

- Main run at block 2048 as today (attention cost, throughput already tuned).
- Final ~5% of tokens at block 4096 with RoPE base raised (the standard
  continued-pretraining length extension). Adds ~1 h to the run.
- Add a length-extrapolation eval: loss on 4096-token val windows for positions
  2048–4096. Inference already allows longer prompts; this makes them reliable.

### Pilot matrix (25M, ~$0.30 each, same data slice, same steps)

| Arm | Tokenizer | Header | Doc mix | Purpose |
|---|---|---|---|---|
| A | MIDILike (current) | no | continuation | baseline, re-run for parity |
| B | REMI 8/beat | no | continuation | isolates 1a |
| C | REMI 24/beat | yes | 60/25/15 | fine grid (triplets + performed timing) |
| D | REMI 8/beat | yes | continuation | isolates 1b |
| E | REMI 8/beat | yes | 60/25/15 | full v4 |
| F | REMI 8/beat | yes | 60/25/15, +4096 tail | full v4 + 1d |

Gate to the production run: E or F beats A on the structure metrics below and
does not lose the blind listening A/B. B vs A alone decides whether the
tokenizer change is worth it at all; if B loses, stop and rethink.

## Phase 2: evals for the new capabilities

Extend `eval.py` / `eval_checkpoint.py`; every pilot arm gets these
automatically.

- **Downbeat alignment**: fraction of onsets on beats 1 and 3, prompt vs
  continuation, plus phrase-start offset from the nearest bar line.
- **Bar-length adherence**: for a requested N-bar answer, how often the
  material actually ends within a quarter beat of bar N.
- **Control adherence**: requested `Density_*`/`Poly_*`/`Inst_*` vs realized
  bucket on the output; confusion matrix per family.
- **Accompaniment fit**: per-bar pitch-class overlap and onset synchrony
  between conditioning track and generated tracks, against the real
  accompaniment held out from the file.
- **Infill seam quality**: repetition and density drift across the two seams,
  plus NLL of the generated span under the baseline model.
- **Human**: `label_app` gains an accompaniment mode (condition track plays
  on both sides, only the generated tracks differ) and an infill mode. These
  feed the same Bradley–Terry reward fit in `reward_align.py`.

## Phase 3: inference and product surface (parallel with training)

Code paths, all non-blocking on the model, testable against the pilot
checkpoints:

- `Generator`: `accompany(condition_ids, bars, header)`, `infill(prefix_ids,
  suffix_ids, bars, header)`, and a `header=` kwarg on the existing calls.
  Stop conditions move from token budgets to `Bar` counting.
- `jam.py` / `live_session.py`: `--answer-bars` becomes exact; new
  `--mode harmonize` keeps the user's clip and answers with an accompaniment
  clip on the model track; `--density`, `--style`, `--instruments` flags map
  to header tokens.
- Web UI and `modal_serve.py`: expose the same three operations and the
  header controls.
- **Text control for free**: a text field is mapped to header tokens by an
  LLM call (Claude, with the token vocabulary in the prompt) before
  generation. This is MuseCoco's two-stage design with stage one outsourced,
  and costs nothing at training time. Ship it only after 1b proves control
  adherence.
- MLX backend: vocab size change only; the KV-cache pipeline is unchanged.

## Phase 4: the v4 production run

- 113M medium config, same recipe as `medium_full_v1` (block 2048, batch 64,
  WSD, aug on, deploy+spawn launch pattern per the PLAN.md log), with the
  winning pilot config and the 4096 tail. ~$80, ~20 h.
- Gate to ship: blind cross-model A/B vs the phase 0 baseline on continuation
  (must not lose), plus the new capability evals (must clearly win, since the
  baseline cannot do them at all).
- Only after that: the 202M production config, per the existing v2 plan.

## Sequencing and rough effort

| Step | Work | Wall time | Compute |
|---|---|---|---|
| 0 | eval + ship medium_full_v1, freeze eval prompts | 2 days | ~$0 |
| 1 | tokenizer config, build_dataset formats, header extraction, corpus re-tokenize (pilot slice) | 3–4 days | ~$0 |
| 2 | new metrics + label_app modes | 2–3 days, overlaps step 1 | ~$0 |
| 1' | pilot matrix A–F + listening | 1 day | ~$2 |
| 3 | Generator / jam / serving APIs against pilot checkpoints | 3–4 days, overlaps the run | ~$0 |
| 4 | full corpus re-tokenize, production run, A/B, ship | 2 days + 20 h GPU | ~$80 |

About three weeks end to end, one production retrain.

## Parked ideas (good, not now)

- **Best-of-n by model likelihood at inference.** Measured 2026-09-14 on 120
  same-model v3 pairs: the base model's own mean log-probability of a
  continuation predicts the human vote **0.717** of the time held-out, versus
  0.658 for the fitted 10-metric reward and a 0.88 labeler ceiling. Not a
  length artifact (corr with length difference -0.07). Combining it with the
  metrics scores *worse* (0.667) than likelihood alone. So: sample n
  candidates in `jam.py` / serving, keep the highest mean log-prob. Costs one
  forward pass per candidate, needs no training, and cannot mode-collapse the
  way optimizing likelihood with RL would. Independent of v4.
- **Neural reward head.** The proper version of `reward_probe`: scalar head on
  the music model, Bradley-Terry loss, trained once preference data reaches
  ~1000+ pairs. A 769-feature linear probe already memorizes at 122 pairs
  (0.99 train / 0.60 held-out), so more capacity needs more labels, not
  cleverer regularization.

## Explicitly deferred

- Anticipatory (arrival-time) interleaving: revisit only if a use case for
  streaming lookahead appears.
- Compound Word / Octuple multi-attribute tokens: 3–5x shorter sequences would
  help context and speed, but need a multi-head output and change the decode
  path the MLX backend is tuned for. Candidate for v5 if context is still the
  binding constraint after 1d.
- Native text encoder (T5 cross-attention as in MusicGen): the LLM-to-header
  route covers the need at zero training cost.
- Audio output: out of scope; MIDI into the DAW is the product.
