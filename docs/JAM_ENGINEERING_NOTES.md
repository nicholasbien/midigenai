# Live jam: engineering notes (session of 2026-09-11)

How the bar-based call-and-response jam (`midigenai/jam.py`) was brought
from "answers land a couple of bars late and off-grid" to "recorded answer
notes within a few ms", and how the autonomous demos were built. Written
for the next agent: every finding below cost real time to discover, and
every measurement can be re-run with `python -m midigenai.jam_diag`.

## Layout

| piece | what it is |
|---|---|
| `midigenai/jam.py` | the jam: listens on an IAC bus, answers on another; MIDI-clock transport, clock-scheduled streaming, fixed-length calls |
| `midigenai/setup_jam_set.py` | builds the `you` / `you (sound)` / `model` tracks with IAC routing |
| `midigenai/demo_song.py` | builds a whole set (chill or techno style, optional band lanes) and runs a recorded pass |
| `midigenai/live_client.py` | the socket client for the AbletonMCP remote script, plus lane/notes helpers |
| `midigenai/jam_diag.py` | diagnostics: lanes, recorded-note timing, socket latency, clock semantics, MIDI port probe |
| `docs/ABLETON_JAM.md` | the user-facing guide (setup, flags, recipes, measured numbers) |

## Timeline of findings

1. **Answers landed 2 bars late.** Every call to Live's remote-script socket
   (port 9877) takes 0.4 s with the transport stopped and 0.4-1.5 s while it
   plays — even a plain `get_arrangement_info` — because Live only gives the
   script's Python threads time between its ticks. The old path made four
   socket calls per answer, one of them inside the MIDI read loop, which
   stalled note capture by up to a second (bunched onsets `[0, 0, 0, 0, ...]`
   in the log are the signature). Rule: nothing time-critical waits on the
   socket. (`jam_diag socket-latency`)
2. **Socket time readings** are current at the moment the *response* arrives
   under light load (error ~0.02 beats over 16 samples) — the delay is on the
   request side. Under load (recording, clip writes) a reading can be seconds
   stale. So: no half-RTT correction, and a reading is never allowed to move a
   dead-reckoned clock backwards; only a big jump (a seek) resets it.
   (`jam_diag clock-check`)
3. **Phase error of 0.05-0.2 beats.** Answers were anchored at the call's
   content end instead of the grid. Fix: the answer's rel-0 belongs at
   `anchor + content_end`; snap that to the bar (or beat) and carry the
   fractional remainder into the notes. The model's on-grid notes then land
   on Live's grid; its off-grid notes stay where it put them.
4. **MIDI clock replaces the socket for timing.** Live → Preferences → MIDI
   Ports → Output `IAC Driver (Bus 1)` → Sync. Live sends 24 clocks/beat,
   `continue`/`stop`, and a song-position message in 16ths (SPP 48 = beat
   12). The clock is sample-derived and pushed, accurate to a few ms. The
   socket poller stays as a cross-check: a reading can be stale (behind) but
   never ahead, so if it is ahead of the MIDI clock by > 0.25 beat the MIDI
   clock missed a position message and is resynced. (`jam_diag midi-probe`)
5. **Streaming beats arrangement writes.** Writing a clip into the
   arrangement is sample-accurate but costs 0.6-1.5 s of socket latency, so
   an answer often slid a bar. Streaming over IAC with notes scheduled on the
   wall clock from the MIDI clock measured 0-4.5 ms in recorded clips and is
   the default; Live's Record captures it on the armed answer lane.
6. **Fixed-length calls** (`--call-bars N`) remove the silence wait: the
   call is anchored to the bar line nearest its first note, generation
   starts `--spec-lead` beats before the closing bar line from what has been
   played so far (held notes closed at that instant), and the answer fires
   0.1 beat before the bar line so its downbeat is scheduled ahead (fired
   exactly at the bar line it played ~50 ms late). A note added inside the
   lead window discards the draft; the regenerated answer keeps its grid and
   skips notes already in the past. At 128 bpm with 1-bar calls, 0.5 beat of
   lead was not always enough for a 0.15-0.3 s generation; 0.75 is.
7. **Tempo.** The MIDI-clock tempo estimate, rounded to 0.1 bpm, drifted
   answers ~7 ms over 4 bars at 128. Live's reported tempo (from the socket
   poller) is exact and is now the primary source.
8. **Monitor In mutes clips.** Answer lanes were on Monitor In; takes were
   silent on playback. An *armed* lane on Monitor Auto both monitors the bus
   (answers audible live) and plays back the take. jam.py sets it.
9. **Socket timeout.** jam.py's default 1 s socket timeout made the
   model-track arming fail silently, so Record captured nothing. 5 s now,
   and failures print.
10. **Live's play starts from the insert marker** (the last place clicked),
    not from the playhead set over the socket. A recorded pass began at bar
    65. The recorder now starts, relocates to 0 while playing, verifies, and
    deletes any empty clip left past the song end.
11. **New IAC ports invalidate open handles.** After Bus 3-6 were added, the
    jam instance opened earlier went deaf and its output vanished. Restart
    jam.py after changing IAC ports.
12. **Live auto-arms new tracks**, stealing the arm from the current one —
    arm last, and re-arm after creating any track. Exclusive-arm can also
    take the arm from `you` when jam.py arms `model`; jam.py re-arms `you`.
13. **Call lanes carry no instrument** (they send MIDI out), so composed
    drum/chord calls were inaudible until `... (sound)` lanes listening to
    them were added.
14. **Echo's mix parameter is `Dry Wet`** (Reverb's is `Dry/Wet`); a setter
    keyed on the slashed name silently left every echo at 70% wet.
    `set_device_parameter` accepts normalized 0-1 values only.
15. **Generation slowed 10x while Live recorded** (0.13 s → 2.4 s for a
    4-bar answer on MLX) in one session; not diagnosed. Speculation hides it
    when the call ends cleanly.

## Measured results

| take | placement of recorded answer notes |
|---|---|
| chill, 4-bar calls, socket clock | 5-17 ms early, one answer 2 bars late |
| chill, 4-bar calls, MIDI clock | ±8 ms; downbeats ~50 ms late (fixed by the early trigger) |
| techno, 1-bar calls, MIDI clock | median +1 ms, p90 +4 ms, worst 10 ms (148 notes) |
| techno band, chords 4-bar | median -2 ms, drift to -7 ms over 4 bars (fixed by exact tempo) |

## How the demos were composed

Constraints, not inspiration: one scale per song (A minor pentatonic /
G minor pentatonic), 3-5-note hooks with an internal repeat and a rest before
the bar line, backing from genre archetypes (four-on-the-floor, offbeat open
hats, sub bass off the kick, stabs on the "and" of 2 and 4, chord change every
8 bars, crash on section starts, snare roll into the drop). Calls, chords and
patterns are `(pitch, beats)` lists in `demo_song.py` — edit and rebuild.
The user's taste, for reference: call and answer on the same instrument,
minimal progressions, no 16th-note percussion layers, dub techno chord
treatment (LP filter + chorus + dotted-8th feedback echo + long reverb).

## Band mode

One jam.py per role on its own bus pair (IAC Buses 1-6, created in Audio
MIDI Setup; Track must be ticked for the new buses in Live's MIDI prefs):
melody 1-bar (Bus 1→2), drums 1-bar with `--drums` (Bus 3→4), chords 4-bar
(Bus 5→6). Live's clock Sync stays on Bus 1; the other instances read it
with `--clock-port`. Three MLX instances run fine side by side.
