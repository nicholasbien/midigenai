# Proposal: bar-aware tokenizer (next training round)

**Status**: proposed — for the run after `medium_full_v1`.
**Owner**: next model-improvement cycle.

## Problem

The current tokenizer (MidiTok **MIDILike**, 641 tokens) represents time only
as relative `TimeShift` tokens on a 1/8th-beat grid. There is **no Bar or
beat-position token**, so the model has no explicit concept of a downbeat —
meter is a statistical inference from rhythm patterns. Observable symptoms:

- jam answers are rhythmically coherent but can start or sit **displaced
  against the bar** (they don't reliably land phrases on "1")
- long generations drift metrically — no anchor to re-align to
- interactive use (jam.py, live_session.py) has to *approximate* bar
  alignment at the playback layer (transport sync) instead of the model
  producing bar-shaped music natively

## Proposal

Switch to a **REMI-style scheme** (MidiTok `REMI`) or MIDILike + structure
tokens:

- `Bar` token at every barline; `Position_x` tokens (grid position within the
  bar) replacing or complementing `TimeShift`
- keep the current design wins: 32 velocity bins, tempo-stripped training
  (`use_tempos=False`), beat-relative timing, multi-instrument interleave
  (`use_programs=True`)
- vocab impact: +~35 tokens (1 Bar + 32 positions at 1/8 grid) — still tiny

REMI's Position tokens make the downbeat explicit in every bar, so the model
learns "phrases start near position 0" as a direct statistical fact.

## Costs / risks

- **Full retrain** — new vocab means no checkpoint reuse. At post-#21 Modal
  throughput that's ~$61 / ~15 h for the 113M at 24B tokens (or a $0.35 pilot
  first, as always).
- Corpus re-tokenization (`data/build_dataset.py`) — hours, not days; the
  cleaned MIDI corpus is unchanged.
- REMI sequences are ~10–15% longer than MIDILike for the same music
  (Bar/Position overhead) — slightly less music per 2048-token context.
- All serving/jam decode paths read time from the decoded Score, not from
  tokens, so **inference code changes are minimal** (tokenizer.json swap +
  the TimeShift-based beat-budget stop in jam.py/live_session needs a
  Position-aware equivalent).

## Evaluation plan

- add a **downbeat-alignment metric** to `eval.py`: fraction of generated
  note onsets on strong beats (1, 3) vs the prompt's grid, compared
  prompt-vs-continuation
- A/B the same prompts against the MIDILike model in `grade_app`
- jam-feel test: transport-sync OFF, does the answer land on the bar?

## Pilot plan

1. Re-tokenize a pilot corpus slice with REMI config.
2. `modal run midigenai/modal_train.py --size pilot --compile ...` (~$0.35).
3. Downbeat metric + listening pass vs the MIDILike pilot.
4. If it wins, fold into the next production run's config.
