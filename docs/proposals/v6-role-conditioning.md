# v6: role-conditioned parts, and what else waits for the next vocab change

Status: proposal, started 2026-09-17 while v5_113m was training. Nothing here is
built. Every item on this page changes the tokenizer vocabulary, so all of it
lands together in one corpus rebuild and one retrain, the way Tempo_ and
Source_fma did for v5. Adding to this page is cheap; adding a token later is a
full rebuild.

## 1. Role_ tokens for the part being written (the reason this doc exists)

**The gap.** The header already has `Poly_0..2` (notes per onset: <1.5, 1.5-3,
>=3), `Range_`, `Density_` and `Inst_`. But for accompaniment and infill
documents the header is computed over condition AND target together
(`v4_docs.py`, the `_subscore(win, cond_idx + tgt_idx)` call), so `Poly_`
describes the whole window. A melody with chords under it gets a mid/high
polyphony bucket whichever part is the target. The model learned it as "this
is what the whole thing looks like", not "write the chordal part". There is no
way today to say *melody* or *chords* about the part the model is asked to
write.

**The proposal.** A `Role_` family emitted on the TARGET of accompaniment and
infill documents only (continuation has no distinct target):

    Role_melody   monophonic or near, mid/high register, moves stepwise
    Role_chords   >=3 notes per onset most of the time, sustained
    Role_bass     low register (median pitch < ~50), mostly monophonic
    Role_drums    is_drum track
    Role_pad      chordal, long durations, sparse onsets
    (Role_other for anything the rules can't place; dropout as for every family)

Derived automatically from the target track's own statistics -- polyphony,
median pitch, note duration, onset density, program -- no labels needed. Rules
first; a tiny classifier only if the rules disagree with a human on a sample
of 200 tracks. Emitted next to `Inst_`, so "add a Role_chords Inst_Piano part"
composes the way `Tempo_` composes with the rest.

**Cost.** ~6 tokens. Header specials sit at the front of the vocab, so every
musical id shifts: corpus rebuild + retrain, same as v5.

**Test that it took**, cheap, after the retrain: same melody condition,
`Role_chords` vs `Role_bass` vs `Role_melody` on the target. If the three
outputs do not differ in polyphony and register, the model ignored it.

**Can be approximated on v5 today**, without new tokens, to learn whether role
control is worth a rebuild: `Inst_` + `Poly_` + `Range_` via
`make_header(..., poly=, pitch_range=)` gets "low monophonic part" or "high
chordal part". If that already changes what the user hears, Role_ is
justified; if not, it probably won't either.

## 2. Also queued for the same rebuild

- **Fragment EOS** (PLAN.md workstream 2; noted again in PR #50). Documents
  under the fragment threshold currently end without EOS, so the model never
  learns that a short piece can *end*. Emit EOS on fragments that end on a bar
  line.
- **Accompaniment window default 8 bars.** v5 built the FMA source with
  `--window-bars 8` because 20 s clips are ~10 bars; the base sources still use
  16. Eval scores accompaniment at 8. Pick one and make it the default so
  training and eval agree everywhere.
- **Full-length transcriptions.** If the FMA clip data earns its place, the
  next transcription job is full tracks, segmented into ~20 s windows with a
  fresh MuScriptor decoder state per window (its 5 s chunk seams show a 1.5x
  gap rate; long passes would drift further). New `Source_` value if the
  distribution differs enough to matter.
- **Grid resolution.** v4/v5 tokenize at 8 positions per beat. Round-trip
  onset error on transcriptions is median 14 ms, p90 27 ms; authored MIDI is
  0 / 21 ms. 24 per beat was the original v4 preference (memory: v4 plan
  state) and was set aside for parity. Worth re-deciding with the transcription
  data in the mix, since it is the data that is off-grid.
- **Tempo for placeholder clips.** 54.5% of fma_small carries MuScriptor's
  120.0 placeholder and gets no `Tempo_` token. A better beat tracker at
  transcription time would recover it; note-based re-estimation was only
  37-52% right and was not applied.

## 3. Not vocab changes, can happen any time

- Ableton held-out split by project (done for v5; keep it).
- `--exclude-ids` on every build pass (done for v5; keep it).
- Quality tokens are trained in but unexploited at inference: sample from the
  top quartile and see if it is audible.

## Gate for starting v6

v5 has shipped or been rejected, AND the role approximation on v5 (section 1,
last paragraph) shows an audible effect. Without the second, put the rebuild
budget into data instead.
