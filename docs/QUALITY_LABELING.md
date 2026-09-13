# Corpus quality labeling

Hand-label the quality of **training data** (not model outputs), fit a small
predictor on those labels, and use it to downweight low-quality files when
the training corpus is built. Two modules:

- `midigenai/rate_corpus_app.py` — local Flask app for blind 1–5 rating of
  ~30 s corpus excerpts.
- `midigenai/quality_predictor.py` — fits a ridge regressor on the logged
  features and applies it to a whole manifest.

## 1. Rating

```bash
python -m midigenai.rate_corpus_app --port 7795
# open http://localhost:7795   (stats at http://localhost:7795/stats)
```

Defaults: manifests `~/midigenai_data/manifest_all_dedup.jsonl` plus
`manifest_gigamidi_dedup.jsonl` if present; equal sampling weight per source
(lakh, lamd, aria, gigamidi, maestro, pop909, giantmidi — otherwise aria's
70% file share would dominate); files shorter than 20 s skipped; output under
`evals/quality/`. Useful flags:

| flag | meaning |
| --- | --- |
| `--manifests a.jsonl b.jsonl` | which manifests to sample from |
| `--source-weights lakh:2,aria:1` | override per-source sampling weights |
| `--seed 0` | deterministic file pool + window choice; change it to see new files |
| `--rater name` | who is rating (default `nicholas`) |
| `--repeat-rate 0.1` | fraction of served items that are blind repeats |
| `--out evals/quality` | where `ratings.jsonl` and `excerpts/` go |

Each item is a random **bar-aligned window** whose bar count is closest to
30 s at the file's tempo (downbeats from symusic; 4/4 from tpq if the file has
no time signature). Windows with fewer than 8 notes are re-drawn; files symusic
cannot parse are skipped and counted. A background thread keeps ~5 excerpts
pre-rendered so there is never a wait.

The UI is deliberately **blind**: only the player, a counter and the key
legend. No filename, source or path is ever sent to the browser.

### Keys

| key | action |
| --- | --- |
| `1`–`5` | rate and advance |
| `j` | broken / junk (logs rating 1 with flag `junk`) |
| `s` | skip (logged, no rating) |
| `r` | replay from the top |
| `space` | play / pause |

Playback starts automatically from the second item on (browsers need one
click or key press before audio is allowed).

### Rubric — rate craft, not taste

The goal is a predictor of **quality**, not of your genre preferences. A
well-made polka should score as high as a well-made piano nocturne; a sloppy
transcription of a song you love should score low.

- **5 — excellent.** Clearly made by someone who knew what they were doing:
  coherent harmony/rhythm, musical phrasing, well-arranged parts, clean timing
  or convincing expressive timing. You want the model to learn from this.
- **4 — good.** Solid, competent music with minor blemishes (a stiff
  quantization, a thin arrangement, a slightly awkward transition).
- **3 — okay.** Recognizably music and not broken, but bland, mechanical,
  repetitive, or sloppy; would neither help nor hurt much.
- **2 — poor.** Serious problems: wrong notes, drifting or messy timing,
  parts clashing, bad transcription artifacts, near-empty or aimless.
- **1 — junk.** Broken or non-musical: noise, corrupted note data, a single
  held chord, random-sounding transcription, drum tracks rendered as pitched
  gibberish, etc. (`j` records this with an explicit `junk` flag.)

Skip (`s`) when you cannot judge (player failed, excerpt is silence, you got
distracted) rather than guessing.

### Self-consistency check

10% of served items (`--repeat-rate`) are blind repeats of excerpts already
rated, in this or an earlier session, drawn once a session has at least 10
rated items. They are logged with `is_repeat: true`. `/stats` (and the summary
printed on Ctrl-C) shows exact agreement and mean |difference| over repeat
pairs. That number is the ceiling for any predictor fit on these labels: if
you agree with yourself only 50% of the time exactly and are off by 0.7 on
average, a model with MAE 0.7 has learned everything there is to learn.

### Log format

`evals/quality/ratings.jsonl`, append-only, one JSON object per event:
`ts, session_id, rater, item_id, excerpt_id, path, source, start_tick,
end_tick, start_seconds, end_seconds, n_bars, rating (int or null for skip),
flags, is_repeat, skipped, listen_seconds, features, excerpt_file`.

`features` is computed on the excerpt: the `eval.py` metrics
(`pitch_class_entropy, scale_consistency, polyphony_rate, note_density_hz,
pitch_range, repetition_rate, ioi_entropy`), plus `n_notes, n_pitched_notes,
n_tracks, n_programs, has_drums, drum_fraction, notes_per_second, tempo_bpm,
duration_seconds, velocity_mean, velocity_std, mean_note_beats`, and the
file-level `file_n_tracks, file_n_notes, file_duration_seconds` from the
manifest.

The rated excerpt is saved as `evals/quality/excerpts/<sha1(path|start|end)>.mid`
so any rating can be re-listened to; the directory is gitignored because the
app rebuilds a missing excerpt from `path` + window when it needs it.

## 2. Fitting the predictor

```bash
python -m midigenai.quality_predictor fit
```

Reads `evals/quality/ratings.jsonl`, drops skips, averages repeated ratings
per excerpt, and fits **ridge regression on standardized features** (numpy
closed form; missing feature → column median). Alpha is picked from
`{0.1, 1, 10, 100}` by grouped cross-validation. It reports:

- grouped 5-fold CV (folds split by file path) Spearman, Pearson and MAE,
  next to the trivial predict-the-mean MAE;
- the self-consistency ceiling from the repeats;
- standardized coefficients (rating change per 1 SD of each feature);
- **per-source bias**: mean predicted vs mean true rating for each source and
  the Spearman *within* each source. If within-source Spearman is near zero
  while the source means line up, the model only learned "which dataset is
  this" — a genre/source prior, not quality — and you should rate more items
  within the sources it is failing on before using it;
- if scikit-learn is installed, a GradientBoostingRegressor on the same folds
  for comparison (reported, not saved).

The model is saved to `evals/quality/predictor.json` (feature names,
imputation medians, standardization, coefficients, and the fit report).
Expect a few hundred labels to be plenty for this model; it has ~20 inputs.

## 3. Scoring a manifest

```bash
python -m midigenai.quality_predictor score \
    --manifest ~/midigenai_data/manifest_all_dedup.jsonl \
    --out ~/midigenai_data/quality_all --workers 8
```

Computes the same features on each file (the first `--max-seconds 60` seconds
of long files, to bound cost) with multiprocessing, and writes
`<out>.jsonl` rows:

```json
{"path": ".../raw/lakh/lmd_full/9/9f....mid", "source": "lakh", "quality": 3.41, "q_bucket": 2}
```

`quality` is the predicted 1–5 rating; `q_bucket` is the quartile of predicted
quality over the scored set (0 = worst 25%, 3 = best 25%). The per-source
distribution over buckets is printed at the end. Use `--limit N` to try it on
a prefix first; expect roughly 0.05–0.3 s per file per worker.

## 4. How `q_bucket` is meant to be used

The dataset builder (`midigenai/data/build_dataset.py`) already bakes a
`--tag` into shard names (`train_<tag>_NNNNN.npy`) and training's `--mixture`
flag applies per-shard weights by matching that tag in the filename
(`--mixture lakh:1,aria:2`). The intended integration, a follow-up to this
work, is to build shards tagged by `source+bucket` (e.g. `lakh_q0`, `lakh_q3`)
from the scored manifest, so a run can downweight or drop the low buckets with
the existing machinery, e.g. `--mixture q0:0.25,q1:0.5,q2:1,q3:1.5`. No
training-side changes are required; the bucket only has to reach the shard
name.
