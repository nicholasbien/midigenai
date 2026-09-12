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
     - "model": the model's instrument, **MIDI From → IAC Bus 2**, Monitor
       In, armed (jam.py flips it to Auto / disarmed itself for
       `--output arrange`)
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
Play. Pause ~0.8 s → the model answers. Every exchange feeds the next prompt.

### Bar-based call and response
```bash
env/bin/python -m midigenai.jam --call-bars 4        # fixed 4-bar calls
env/bin/python -m midigenai.jam                      # free phrases, answer after a pause
```
**One-time: give jam.py Live's MIDI clock.** Preferences → Link, Tempo &
MIDI → MIDI Ports. In the **Output** list, row `IAC Driver (Bus 1)`: tick
**Sync** (Track stays ticked). Leave Sync unticked on both Input rows —
ticking it there would make Live follow its own clock. Verified 2026-09-11:
Live then sends 24 clocks per beat, `continue`/`stop`, and a song-position
message in 16ths (SPP 48 = beat 12) down the bus jam.py already listens
to; jam.py prints "MIDI clock locked" on the first clocks and follows
Live's tempo (no `--bpm` needed; estimate over 8-16 beats, 0.1 bpm steps,
0.3% hysteresis). Without it, jam.py falls back to polling the
remote-script socket, which is 0.4-1.5s per call and can report positions
seconds stale while Live records — answers then slide or land off-grid.

**Delivery (`--output stream`, the default).** Answers are streamed over
IAC Bus 2 with each note scheduled on the wall clock from the transport
clock: measured ≤5 ms from the 16th grid inside recorded clips. Keep the
`model` track armed with Monitor In (setup_jam_set does) and hit Live's
Record: your call lands on the `you` lane, the answer on the `model` lane.
`--output arrange` writes the answer into the arrangement instead —
sample-accurate placement, but each write costs 0.6-1.5s of socket
latency, so an answer may slide to the following bar line.

**Placement (`--sync bar`, the default).** The answer's origin is the bar
line where the call ended (or the next one that can still be made); the
model's grid phase relative to that is preserved exactly, so its on-grid
notes fall on Live's grid. Free phrases are anchored to the nearest Live
beat to your first note (`--anchor note` to use the note itself). Answer
length matches the call (`--answer-bars match`): 1 bar back for a 1-bar
call, 4 for 4; an integer forces a length.

**Latency (`--call-bars N`).** With fixed-length calls there is no silence
wait: the call is N bars from the bar line it started on, generation starts
`--spec-lead` beats (default 1) before that bar line from what has been
played so far (held notes included), and the answer triggers exactly at the
bar line. If you add notes inside the last beat the draft is discarded and
regenerated (0.1-0.4s locally); the answer keeps its grid — notes that
would already be in the past are skipped rather than smeared late. Without
`--call-bars`, detection costs the silence window (0.4-0.8s) and the answer
goes to the next grid point it can make.

Tuning flags: `--silence 0.8` (pause that triggers), `--min-notes 4`
(ignore stray taps), `--max-notes 100` (answer immediately at this size),
`--answer-bars match|N`, `--max-answer-bars 8`, `--sync bar|beat|off`,
`--anchor beat|note`, `--output stream|arrange|clip`, `--bpm` (only if not
following MIDI clock — must match Live!), `--latency-comp`, `--temperature`,
`--in-port/--out-port`.

### Measured (2026-09-11, Live 12, M-series Mac, MLX)
Full autonomous take (`demo_song.py --record`, six 4-bar calls, six 4-bar
answers, every answer speculated ahead of its bar line, generation
0.13-0.36 s): recorded answer notes within **±8 ms** of their intended
positions; the MIDI clock and the socket clock agreed within 0.02-0.05
beats throughout. The downbeat note of each answer was ~50 ms late when
the trigger fired exactly at the bar line, hence the 0.1-beat early
trigger. Before the MIDI clock (socket-polled position) answers were
5-17 ms early and, while Live recorded, could slide whole bars.

### Autonomous demo (`midigenai/demo_song.py`)
Open a blank set (File → New), then:
```bash
env/bin/python -m midigenai.demo_song                  # backing + jam tracks + 6 calls
env/bin/python -m midigenai.jam --call-bars 4          # in another terminal, wait for "model track -> Monitor In, armed"
env/bin/python -m midigenai.demo_song --record --skip-build   # 56-bar recorded pass
```
The calls sit on the `you` lane and play out over IAC Bus 1 like your own
playing would; the answers are recorded onto the `model` lane. Save the set
afterwards (there is no save command over the socket). Then arm `you`,
press Record and play your own calls over the same backing.

### Techno / 1-bar calls (`--style techno`)
```bash
env/bin/python -m midigenai.demo_song --style techno            # 128 bpm, 64 bars, 20 one-bar calls, effects
env/bin/python -m midigenai.jam --call-bars 1 --min-notes 3 --spec-lead 0.75
env/bin/python -m midigenai.demo_song --style techno --skip-build --record
```
Measured 2026-09-11: 148 recorded answer notes, median 1 ms from the
model's 1/8-beat grid, 90% within 4 ms, worst 10 ms. With 1-bar calls the
speculation window is short (0.5 beat = 0.23 s at 128 was not always
enough for a 0.15-0.3 s generation, so the downbeat played up to 110 ms
late) — use `--spec-lead 0.75` and end each call by beat 3.5.

### The model as a band (`--band`): melody 1 bar, drums 1 bar, chords 4 bars
One jam.py per role on its own IAC bus pair (create Buses 3-6 in Audio
MIDI Setup → IAC Driver; Live's clock Sync stays on Bus 1 and the other
instances read it with `--clock-port`):
```bash
env/bin/python -m midigenai.demo_song --style techno --band
env/bin/python -m midigenai.jam --call-bars 1 --min-notes 3 --spec-lead 0.75
env/bin/python -m midigenai.jam --call-bars 1 --min-notes 3 --spec-lead 0.75 --drums \
    --role "model drums" --in-port "IAC Driver Bus 3" --out-port "IAC Driver Bus 4" --clock-port "IAC Driver Bus 1"
env/bin/python -m midigenai.jam --call-bars 4 --role "model chords" \
    --in-port "IAC Driver Bus 5" --out-port "IAC Driver Bus 6" --clock-port "IAC Driver Bus 1"
env/bin/python -m midigenai.demo_song --style techno --band-only --skip-build --record
```
`--drums` encodes the call on a drum track so the model continues with
drums; `--role` names the Live track the answers are recorded on (each
answer lane is armed with Monitor In on its bus). Kick and hats stay on
the backing lane; the drum calls carry the top of the kit.

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
