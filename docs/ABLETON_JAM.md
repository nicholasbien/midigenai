# Interactive Ableton jamming — setup guide

Two ways to play live with the model. **The MIDI-bus jam (`jam.py`) is the
one that feels like jamming** — no buttons, you play and it answers when you
pause. The clip watcher (`live_session.py`) is the structured alternative
where answers land as bar-aligned session clips.

## The jam flow (midigenai/jam.py) — recommended

```
your keyboard → Ableton track ("MIDI To: IAC Bus 1") → jam.py hears you
                                                          ↓ pause ~1.5s
model answer  ← Ableton track ("MIDI From: IAC Bus 2") ← jam.py streams notes
```

### One-time prerequisites
1. **IAC buses**: Audio MIDI Setup → Window → Show MIDI Studio → double-click
   IAC Driver → "Device is online" with at least 2 ports (Bus 1, Bus 2).
   (Already configured on this machine.)
2. **Ableton template**: open one of the `midi_test*` sets
   (`~/Music/Ableton/process Project/midi_test_copy.als` or similar) — they
   have the routing below already. To build it fresh in any set:
   - **Three tracks, not two** — a MIDI track with an instrument outputs
     *audio*, so IAC buses don't appear in its output options:
     - "you": NO instrument, armed, Monitor Auto, **MIDI To → IAC Bus 1**
     - "you (sound)": your instrument, **MIDI From → IAC Bus 1**, Monitor In
     - "model": the model's instrument, **MIDI From → IAC Bus 2**, Monitor In
   - Or just run **`python -m midigenai.setup_jam_set`** — with the
     routing-tools remote script loaded (ableton-mcp-pro PR #6, needs a Live
     restart after install) it builds all of this with zero clicks; verified
     end-to-end on a fresh set 2026-09-01.

### Run it
```bash
cd ~/midigenai
env/bin/python -m midigenai.jam                      # hub checkpoint
env/bin/python -m midigenai.jam \
    --checkpoint runs/<run>/ckpt_XXXX.pt \
    --tokenizer  runs/<run>/tokenizer.json           # a specific/training ckpt
```
Play. Pause ~1.5 s → the model answers, streaming notes out as it generates
(first notes ~0.3 s after the pause). Every exchange feeds the next prompt.

Tuning flags: `--silence 1.5` (pause that triggers), `--min-notes 4`
(ignore stray taps), `--max-notes 100` (answer immediately at this size),
`--max-answer-bars 8`, `--bpm 120` (match Live's tempo!), `--temperature`,
`--in-port/--out-port`.

### Gotchas (each of these has burned a session)
- **Never load an instrument onto the 'you' track.** An instrument flips the
  track to audio output, silently replacing 'MIDI To IAC Bus 1' with 'Main' —
  the jam goes deaf. To change your sound, put the instrument on
  'you (sound)'; for the model's sound, on 'model'.
- **The play-in track's input must never be 'All Ins'** (Live's default for
  new tracks): All Ins includes IAC Bus 2, so the model's own answers loop
  back in — a runaway feedback storm. Pin it to your keyboard
  (setup_jam_set does this automatically).
- **`--bpm` must match Live's session tempo** or answers play at the wrong
  speed relative to the metronome.
- Playing through an armed track is enough for jam.py (it taps the MIDI bus).
  It is NOT enough for the clip watcher, which needs recorded clips.
- If nothing prints "hearing you...", the routing is wrong: check Track A's
  "MIDI To" really says IAC Bus 1 (not the port's *channel* submenu).
- Ableton's remote-script socket (9877) is unrelated to this flow — jam.py
  works even when AbletonMCP is broken or Live was restarted.

## The clip watcher (midigenai/live_session.py)

Needs Live running with the AbletonMCP control surface (ableton-mcp-pro).
Session-record a phrase on any track; the answer lands as a clip on an
auto-created "midigenai" track and fires on Live's launch quantization
(set the Q dropdown to 1/4). Gotchas:
- Live auto-arms newly created tracks: after the watcher creates its answer
  track, re-arm YOUR track or your playing goes to the wrong place.
- Restart the watcher after opening a different Live set (track indices go
  stale; its socket dies with a ConnectionReset).
- A stopped transport swallows fired clips — keep Live playing.

## Using a mid-training checkpoint

```bash
env/bin/modal volume ls openmusenet2-v2-runs <run_name>        # find latest ckpt
env/bin/modal volume get openmusenet2-v2-runs <run>/ckpt_NNN.pt runs/<run>/
env/bin/modal volume get openmusenet2-v2-corpus corpus_full/tokenizer.json runs/<run>/
```
Checkpoints are written atomically every 1,000 steps, so mid-run grabs are
safe. Training checkpoints include optimizer state (~3x larger); loading is
unchanged.
