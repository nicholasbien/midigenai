# Jam timing: architecture, bugs found, and how to verify

Everything learned making `midigenai/jam.py` answer *in time* (2026-09-01).
Read this before touching timing code — each lesson here was paid for.

## Where time lives, stage by stage

```
your playing (wall clock, continuous)
  → capture           NoteBuffer: seconds relative to first note
  → to_beats          seconds -> beats at --bpm  ⚠ must match Live's tempo
  → tokenize          quantized to the tokenizer grid (1/8 beat = 32nd note)
  → model             TimeShift tokens; on-grid BY CONSTRUCTION, relative
                      to its own timeline (prompt start = beat 0)
  → placement         answer rel-0 anchored to a wall-clock instant
  → delivery          Player thread -> mido -> IAC -> Live input
```

The model cannot be "slightly off" — every onset it emits snaps to the grid
at tokenization. If an answer sounds out of time, the bug is in placement or
delivery, or it's a musical-judgment issue (see "model ceiling" below).

## The three delivery bugs (all fixed in this PR)

### 1. The padding crop
`encode_segments` padded the phrase length to a whole bar and reported that
as the answer's origin. But the tokenizer emits nothing after the last note —
the model generated from the phrase's *actual* end while playback discarded
everything before the padded barline. **The first ~0–4 beats of every answer
were silently cropped and the rest time-shifted.**
Lesson: *the answer's origin is the encoded content's true end, never a
rounded value. Padding only exists for the model if it is materialized as
tokens (inter-segment gaps are; trailing silence is not).*

### 2. Phase destruction
The model's answer is on-grid **relative to the phrase** — e.g. phrase ends
at 3.75, first answer note at rel 0.25 = absolute 4.0, dead on the beat. But
playback dropped rel-0 on an arbitrary instant (or a raw bar line), which
destroyed that phase: answers played a consistent fraction of a beat off
even though every onset was grid-quantized (measured: +0.25 beats).
Fix: **phase-preserving placement** — rel-0 is anchored so the origin's grid
phase (`content_end mod beat`) is kept, against Live's transport when
playing (`--sync beat|bar`), else against the phrase's own wall clock.
Lesson: *aligning "the start of the answer" to the grid is wrong; align the
answer's internal grid to the external grid.*

### 3. The constant ~37 ms delivery lag
With phases correct, recorded notes still landed +0.073 beats (~37 ms) late —
a constant, from the song-position query's staleness + player wake
granularity + the mido→CoreMIDI→IAC→Live-input chain.
Fix: schedule early (`--latency-comp`), now **self-calibrating**:
- every sync query is RTT-corrected (position advanced by half the round trip)
- when Live is recording, each answer reads back its own recorded notes
  ~1 s after playing and nudges the compensation halfway toward the median
  landed-vs-intended error (≥4 matched notes, |err| < 0.4 beat, clamp
  0–150 ms). Converges in a couple of exchanges; tracks drift.

## How to verify timing (the method that found all three)

1. **Model side**: log the decoded answer onsets in beats (jam.py prints
   `answer onsets(beats)` per exchange). On-grid values here = the model and
   tokenizer are fine.
2. **Delivery side**: record the jam in Live's arrangement, then read the
   notes *inside* the recorded clip (`get_arrangement_clip_notes`) and take
   onsets mod grid. **Never trust clip boundaries — Live snaps them and they
   lie** (a clip "starting at beat 912.000" contained notes all +0.073 late).
3. Compare per-note: capture onsets vs answer onsets vs landed onsets
   localizes any offset to one stage immediately.

## Related facts that shape what "in time" can mean

- **Tokenizer grid**: `beat_res={(0,4): 8, (4,12): 4}` — 1/8-beat (32nd-note)
  resolution. No micro-timing, no swing, and **no true triplets** (1/3 beat
  is off-grid; triplet material was warped to 32nds in training too).
- **Model ceiling**: MIDILike has no Bar/Position tokens — no explicit
  downbeat concept. Phrase-level placement against the bar is statistical.
  Both limitations are addressed in
  [bar-aware-tokenizer.md](proposals/bar-aware-tokenizer.md) (REMI tokens +
  `beat_res` 12 or 24, one retrain tests both).
- **The silence window is not in the prompt**: the trigger pause happens
  after the last note-off and never becomes TimeShift tokens.
- `--bpm` must equal Live's session tempo or capture and playback are both
  proportionally wrong — the model sees a distorted rhythm and answers in it.
