"""
Interactive MIDI jamming: play, pause, and the model answers — no clicks.

The modern successor to openmusenet2's ableton_bridge.py + note_player.py:
- listens to your playing on a MIDI input port (default "IAC Driver Bus 1")
- when you go silent for --silence seconds (after at least --min-notes),
  the buffered phrase becomes the prompt and generation starts
- the answer STREAMS back out on a MIDI output port (default "IAC Driver
  Bus 2") as the model emits notes — first notes sound almost immediately
- keeps a running conversation (your phrases + its answers) as context

Ableton side (setup_jam_set builds it): 'you' with "MIDI To" -> IAC Bus 1
(no instrument), 'you (sound)' with your instrument listening to 'you',
'model' with the model's instrument, "MIDI From" -> IAC Bus 2. For tight
timing turn on Sync for IAC Bus 1 in Live's MIDI preferences (Output):
jam.py then follows Live's MIDI clock instead of the slow remote-script
socket, and schedules answers to the millisecond.

Usage:
    python -m midigenai.jam [--checkpoint ... --tokenizer ...] [--bpm 120]
        [--in-port "IAC Driver Bus 1"] [--out-port "IAC Driver Bus 2"]
        [--silence 1.5] [--min-notes 4] [--max-notes 100] [--max-answer-bars 8]
"""

from __future__ import annotations

import argparse
import collections
import heapq
import tempfile
import threading
import time
from pathlib import Path


class NoteBuffer:
    """Collects note_on/note_off wall-clock events into finished notes."""

    def __init__(self):
        self.open: dict[int, tuple[float, int]] = {}   # pitch -> (start_t, velocity)
        self.notes: list[dict] = []                     # {pitch,start,end,velocity} secs
        self.t0: float | None = None
        self.last_event: float | None = None
        self.n_on = 0                                   # note_ons this phrase

    def feed(self, msg, now: float):
        if msg.type == "note_on" and msg.velocity > 0:
            if self.t0 is None:
                self.t0 = now
            self.open[msg.note] = (now - self.t0, msg.velocity)
            self.last_event = now
            self.n_on += 1
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in self.open:
                start, vel = self.open.pop(msg.note)
                self.notes.append({"pitch": msg.note, "start": start,
                                   "end": now - self.t0, "velocity": vel})
                self.last_event = now

    def flush(self) -> list[dict]:
        """Close any held notes and return + reset the buffer."""
        now_rel = (self.last_event or 0) - (self.t0 or 0)
        for pitch, (start, vel) in self.open.items():
            self.notes.append({"pitch": pitch, "start": start,
                               "end": max(now_rel, start + 0.1), "velocity": vel})
        notes, self.notes = self.notes, []
        self.open.clear()
        self.t0 = None
        self.last_event = None
        self.n_on = 0
        return notes

    def snapshot(self, now: float) -> list[dict]:
        """Finished notes plus held ones closed at `now` (for speculating
        ahead of a bar line while keys are still down)."""
        out = list(self.notes)
        if self.t0 is not None:
            rel = now - self.t0
            for pitch, (start, vel) in self.open.items():
                out.append({"pitch": pitch, "start": start,
                            "end": max(rel, start + 0.1), "velocity": vel})
        return out


class Player:
    """Streams scheduled note events to a MIDI output with a small lookahead
    buffer so slightly out-of-order arrivals still play in order. A note_on
    that arrives more than MAX_LATE behind schedule is dropped (matching the
    old note_player's misfire behavior) — better to skip a note than smear
    the timing; its note_off still sends, which is harmless."""

    MAX_LATE = 0.15  # seconds a note_on may run behind schedule before dropping

    def __init__(self, outport):
        self.outport = outport
        self.heap: list[tuple[float, int, object]] = []
        self.lock = threading.Condition()
        self.seq = 0
        self.dropped = 0
        self.sent = collections.deque(maxlen=1024)  # (t, pitch) of recent sends
        threading.Thread(target=self._run, daemon=True).start()

    def schedule(self, when: float, msg):
        with self.lock:
            heapq.heappush(self.heap, (when, self.seq, msg))
            self.seq += 1
            self.lock.notify()

    def _run(self):
        while True:
            with self.lock:
                while not self.heap:
                    self.lock.wait()
                when, _, msg = self.heap[0]
                delay = when - time.monotonic()
                if delay > 0:
                    self.lock.wait(timeout=min(delay, 0.05))
                    continue
                heapq.heappop(self.heap)
            if msg.type == "note_on" and time.monotonic() - when > self.MAX_LATE:
                self.dropped += 1
                continue
            if msg.type == "note_on":
                self.sent.append((time.monotonic(), msg.note))
            self.outport.send(msg)

    def sent_recently(self, pitch: int, window: float = 0.35) -> bool:
        """True if we just sent this pitch — used to reject echoes of our own
        output looping back through a misrouted Ableton track."""
        now = time.monotonic()
        for t, p in reversed(self.sent):
            if now - t > window:
                return False
            if p == pitch:
                return True
        return False


class MidiClock:
    """Live's transport, from its MIDI sync output (Preferences > Link/Tempo/
    MIDI > Sync on the IAC port jam.py listens to): 24 clocks per beat plus
    start / continue / stop / song-position. Sample-derived and pushed, so
    it is accurate to a few ms with no request latency — unlike the
    remote-script socket (0.4-1.5s per call). Also yields the tempo."""

    def __init__(self):
        self.lock = threading.Lock()
        self.playing = False
        self.pos = 0.0            # beats at self.wall
        self.wall: float | None = None
        self.first = False        # next clock is the first after start/continue
        self.ticks = collections.deque(maxlen=385)  # wall times of recent clocks (16 beats)
        self.seen = False
        self.events: list[str] = []                 # transport events, for the log

    def feed(self, msg, now: float) -> None:
        t = msg.type
        with self.lock:
            if t == "clock":
                self.seen = True
                if not self.playing:
                    return
                if self.first:
                    self.first = False
                else:
                    self.pos += 1.0 / 24.0
                self.wall = now
                self.ticks.append(now)
            elif t == "songpos":
                self.pos = msg.pos * 0.25           # SPP counts 16ths
                self.wall = now
                self.events.append(f"songpos -> beat {self.pos:.2f}")
            elif t == "start":
                self.pos, self.wall = 0.0, now
                self.playing, self.first = True, True
                self.ticks.clear()
                self.events.append("start (beat 0)")
            elif t == "continue":
                self.wall = now
                self.playing, self.first = True, True
                self.ticks.clear()
                self.events.append(f"continue at beat {self.pos:.2f}")
            elif t == "stop":
                self.playing = False
                self.events.append(f"stop at beat {self.pos:.2f}")

    def resync(self, pos: float, wall: float) -> None:
        """Re-seat the clock (e.g. from an authoritative socket reading)."""
        with self.lock:
            self.pos, self.wall = pos, wall

    def ref(self, max_age: float = 0.5) -> tuple[float, float] | None:
        """(beat, wall) of the latest clock, or None when stopped / stale."""
        with self.lock:
            if not self.playing or self.wall is None:
                return None
            if time.monotonic() - self.wall > max_age:
                return None
            return self.pos, self.wall

    def spb(self) -> float | None:
        """Seconds per beat over the last 8-16 beats of clocks, rounded to
        0.1 bpm (read-loop jitter on individual clocks is ms-scale)."""
        with self.lock:
            n = len(self.ticks)
            if n < 193:
                return None
            bpm = 60.0 * ((n - 1) / 24.0) / (self.ticks[-1] - self.ticks[0])
            return 60.0 / (round(bpm * 10.0) / 10.0)


def main():
    import mido

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--in-port", default="IAC Driver Bus 1")
    ap.add_argument("--out-port", default="IAC Driver Bus 2")
    ap.add_argument("--bpm", type=float, default=None,
                    help="session tempo; default: follow Live's MIDI clock "
                         "(120 until the first clocks arrive)")
    ap.add_argument("--silence", type=float, default=0.8,
                    help="seconds of silence that triggers an answer")
    ap.add_argument("--speculate", action=argparse.BooleanOptionalAction, default=True,
                    help="start generating during the pause; if you stay quiet the "
                         "answer plays instantly (draft discarded if you keep playing)")
    ap.add_argument("--adaptive-silence", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="halve the silence window when your phrase ends on a "
                         "whole-bar boundary of its own grid")
    ap.add_argument("--min-notes", type=int, default=4)
    ap.add_argument("--max-notes", type=int, default=100,
                    help="answer immediately once this many notes are buffered")
    ap.add_argument("--max-answer-bars", type=int, default=8)
    ap.add_argument("--answer-bars", default="match",
                    help="answer length in bars: 'match' (default) answers a "
                         "1-bar call with 1 bar and a 4-bar call with 4 bars "
                         "(call rounded up to whole bars, 1/4-beat tolerance); "
                         "an integer forces a fixed length. Capped by "
                         "--max-answer-bars.")
    ap.add_argument("--context", choices=["phrase", "session"], default="phrase",
                    help="prompt with just your latest phrase (default, lowest "
                         "latency) or the whole running session history")
    ap.add_argument("--latency-comp", type=float, default=0.037,
                    help="seconds to schedule answers EARLY, compensating the "
                         "socket/IAC/Live-input delivery chain (measured ~37ms "
                         "from recorded arrangement clips; re-measure with the "
                         "onset diagnostics if your buffer settings change)")
    ap.add_argument("--output", choices=["stream", "arrange", "clip"],
                    default="stream",
                    help="stream (default): play the answer over the MIDI bus, "
                         "scheduled on the transport clock to the ms — record "
                         "the armed 'model' track to keep it. arrange: write "
                         "each answer into the ARRANGEMENT instead (sample-"
                         "accurate but each write costs 0.6-1.5s of Live socket "
                         "latency, so an answer may slide to the next bar "
                         "line). clip: fire session clips. Both fall back to "
                         "streaming when the transport is stopped.")
    ap.add_argument("--sync", choices=["off", "beat", "bar"], default="bar",
                    help="snap each answer's origin to Live's bar (default) or "
                         "beat grid, keeping the model's grid phase; off: "
                         "answer on the phrase's own clock")
    ap.add_argument("--call-bars", type=int, default=0,
                    help="fixed-length calls: your phrase is taken to be this "
                         "many bars from the bar line it started on, and the "
                         "answer triggers AT that bar line (no silence wait; "
                         "speculation starts --spec-lead beats before it). "
                         "0 (default): free phrases, answer after a pause")
    ap.add_argument("--anchor", choices=["beat", "note"], default="beat",
                    help="free phrases: take the call's grid to start on the "
                         "nearest Live beat to your first note (default), or "
                         "exactly at the first note ('note')")
    ap.add_argument("--spec-lead", type=float, default=1.0,
                    help="with --call-bars: beats before the bar line to start "
                         "generating from what has been played so far")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    if args.checkpoint:
        from .generate import Generator
        g = Generator(args.checkpoint, args.tokenizer)
    else:
        from .hub import load_from_hub
        g = load_from_hub()

    spb = 60.0 / (args.bpm or 120.0)  # seconds per beat (updated by MIDI clock)

    def adopt_tempo(new_spb: float) -> None:
        nonlocal spb
        if abs(new_spb - spb) / spb > 0.003:    # 0.3% hysteresis (~0.4 bpm)
            print(f"tempo from MIDI clock: {60.0 / new_spb:.1f} bpm", flush=True)
            spb = new_spb

    inport = mido.open_input(args.in_port)
    outport = mido.open_output(args.out_port)
    player = Player(outport)
    mclock = MidiClock()
    print(f"jamming: listening on '{args.in_port}', answering on '{args.out_port}' "
          f"({60.0 / spb:.0f} bpm{'' if args.bpm else ' until MIDI clock'}, "
          f"output {args.output}, sync {args.sync}, backend {g.backend})", flush=True)
    if args.call_bars:
        print(f"fixed {args.call_bars}-bar calls: answers trigger at the bar line. "
              f"Ctrl+C to stop.", flush=True)
    else:
        print(f"play; pause {args.silence}s and the model answers. Ctrl+C to stop.",
              flush=True)
    print(f"MIDI clock: waiting for Live's Sync output on '{args.in_port}' "
          f"(falls back to the remote-script socket)", flush=True)

    buf = NoteBuffer()
    history: list[list[dict]] = []   # beat-domain segments (user, model, ...)

    comp = {"value": args.latency_comp}  # live-tunable latency compensation

    def answer_target(phrase_beats: float) -> float:
        """Beats the answer should span, measured from the call's content end.
        With --call-bars the call's grid length is known, so the answer is
        stretched to end on a bar line even when the last note was released
        early."""
        import math
        call_len = args.call_bars * 4.0 if args.call_bars else phrase_beats
        if args.answer_bars == "match":
            bars = max(1, math.ceil((call_len - 0.25) / 4.0))
        else:
            bars = max(1, int(args.answer_bars))
        beats = min(bars, args.max_answer_bars) * 4.0
        if args.call_bars:
            beats += max(0.0, call_len - phrase_beats)
        return beats

    def to_beats(notes_secs: list[dict]) -> list[dict]:
        base = min(n["start"] for n in notes_secs)
        return [{
            "pitch": n["pitch"],
            "start_time": (n["start"] - base) / spb,
            "duration": max(0.05, (n["end"] - n["start"]) / spb),
            "velocity": n["velocity"],
        } for n in notes_secs]

    def encode_segments(segs: list[list[dict]]) -> tuple[list[int], float]:
        """Returns (prompt_ids, content_end_beats). content_end is where the
        encoded material actually ends — NOT padded to a bar. The tokenizer
        emits nothing after the last note, so treating a bar-padded value as
        the answer origin silently cropped the first beats of every answer
        (the padding bug)."""
        from symusic import Score, Track, Note as SNote
        TPQ = 480
        while True:
            score = Score(TPQ)
            tr = Track()
            cursor = 0.0
            content_end = 0.0
            for seg in segs:
                seg_end = max(n["start_time"] + n["duration"] for n in seg)
                for n in seg:
                    tr.notes.append(SNote(
                        time=int(round((cursor + n["start_time"]) * TPQ)),
                        duration=max(1, int(round(n["duration"] * TPQ))),
                        pitch=int(n["pitch"]), velocity=int(n["velocity"])))
                content_end = cursor + seg_end
                # inter-segment gap padded to bars (materialized as TimeShifts
                # between segments, so it is real to the model)
                cursor += (int(seg_end // 4) + 1) * 4.0
            score.tracks.append(tr)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp = f.name
            score.dump_midi(tmp)
            ids = g.encode_midi_file(tmp)
            Path(tmp).unlink(missing_ok=True)
            if len(ids) <= 1500 or len(segs) == 1:
                return ids, content_end
            segs = segs[1:]

    gen_lock = threading.Lock()          # serialize model access
    spec_lock = threading.Lock()
    spec = {"key": None, "plan": None, "busy": False}

    def prompt_segments(user_notes_beats: list[dict]) -> list[list[dict]]:
        if args.context == "session":
            return list(history) + [user_notes_beats]
        return [user_notes_beats]

    def generate_plan(user_notes_beats: list[dict]) -> dict:
        """Full generation with no side effects: returns the answer as a note
        plan. Used by the speculative worker (and could serve the direct path
        too, but the direct path streams for lower first-note latency)."""
        phrase_beats = max(n["start_time"] + n["duration"] for n in user_notes_beats)
        target = answer_target(phrase_beats)
        prompt_ids, prompt_beats = encode_segments(prompt_segments(user_notes_beats))
        t0 = time.perf_counter()
        notes = []
        for note in g.stream_notes(prompt_ids, tempo_bpm=60.0 / spb,
                                   max_new_tokens=int(target * 40),
                                   temperature=args.temperature):
            rel_beat = note.start / spb - prompt_beats
            if rel_beat < -1e-6:
                continue
            if rel_beat >= target:
                break
            notes.append({"pitch": note.pitch, "start_time": rel_beat,
                          "duration": max(0.05, (note.end - note.start) / spb),
                          "velocity": note.velocity})
        return {"notes": notes, "target": target, "prompt_tokens": len(prompt_ids),
                "origin": prompt_beats, "gen_s": time.perf_counter() - t0}

    cal = {"track": None}

    def _ableton(cmd, params=None, timeout=5.0):
        import json as _json
        import socket as _socket
        sk = _socket.socket()
        sk.settimeout(timeout)
        sk.connect(("localhost", 9877))
        sk.sendall(_json.dumps({"type": cmd, "params": params or {}}).encode())
        raw = b""
        while True:
            chunk = sk.recv(4194304)
            if not chunk:
                break
            raw += chunk
            try:
                resp = _json.loads(raw)
                break
            except ValueError:
                continue
        sk.close()
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("message"))
        return resp.get("result", resp)

    def _find_record_track() -> int | None:
        """The armed track whose MIDI input is our answer bus."""
        import re as _re
        m = _re.search(r"Bus \d+", args.out_port)
        needle = m.group(0) if m else args.out_port
        try:
            n = int(_ableton("get_session_info").get("track_count", 0))
            for i in range(n):
                t = _ableton("get_track_info", {"track_index": i})
                if not t.get("arm"):
                    continue
                r = _ableton("get_track_routing", {"track_index": i})
                if needle in str(r.get("input_routing_type", "")):
                    return i
        except Exception:
            pass
        return None

    def _calibrate(intended: list[float]) -> None:
        """Read back where our answer actually landed (recorded arrangement
        notes) vs where we aimed it, and trim the latency compensation."""
        try:
            if cal["track"] is None:
                cal["track"] = _find_record_track()
            if cal["track"] is None:
                return
            clips = _ableton("get_arrangement_clips",
                             {"track_index": cal["track"]}).get("clips", [])
            if not clips:
                return
            idx = len(clips) - 1
            c = clips[idx]
            notes = _ableton("get_arrangement_clip_notes",
                             {"track_index": cal["track"],
                              "arrangement_clip_index": idx}).get("notes", [])
            lo, hi = min(intended) - 0.5, max(intended) + 0.5
            landed = sorted(c["start_time"] + n["start_time"] for n in notes
                            if lo <= c["start_time"] + n["start_time"] <= hi)
            if len(landed) < 4:
                return
            errs = sorted(min((l - i for i in intended), key=abs) for l in landed)
            median_err_beats = errs[len(errs) // 2]
            if abs(median_err_beats) > 0.4:
                return  # matched the wrong material; don't learn from it
            err_s = median_err_beats * spb
            new = min(0.15, max(0.0, comp["value"] + 0.5 * err_s))
            if abs(new - comp["value"]) > 0.002:
                comp["value"] = new
                print(f"answered-calibration: landed {err_s*1000:+.0f}ms off — "
                      f"latency comp now {new*1000:.0f}ms", flush=True)
        except Exception as e:
            print(f"(calibration skipped: {e})", flush=True)

    clip_state = {"track": None, "slot": 0, "n_slots": 8, "fails": 0,
                  "write_s": 0.4}  # recent arrangement-write latency (adaptive)

    # Live's remote-script socket answers in 0.4-1.5s while the transport runs
    # (its Python threads only get time between Live's ticks), so nothing
    # time-critical may wait on it. A background poller keeps a dead-reckoned
    # transport clock instead: beat = pos + elapsed wall time / spb.
    clock = {"pos": None, "wall": None, "playing": False, "lat": 0.0,
             "recording": False}
    clock_lock = threading.Lock()

    def _clock_poll():
        while True:
            try:
                t0 = time.monotonic()
                r = _ableton("get_arrangement_info", timeout=3.0)
                t1 = time.monotonic()
                with clock_lock:
                    clock["lat"] = t1 - t0
                    clock["playing"] = bool(r.get("is_playing", False))
                    clock["recording"] = bool(r.get("record_mode", False))
                    if clock["playing"] and r.get("current_song_time") is not None:
                        # measured under light load: the delay is on the request
                        # side and the reading is current at response time
                        # (error ~0.02 beats). Under load (recording, clip
                        # writes) a reading can be SECONDS stale, so a reading
                        # behind the dead-reckoned estimate is ignored; only a
                        # big backwards jump (a seek / restart) resets.
                        pos = float(r["current_song_time"])
                        est = None
                        if clock["wall"] is not None and t1 - clock["wall"] < 4.0:
                            est = clock["pos"] + (t1 - clock["wall"]) / spb
                        if est is None or pos >= est - 0.05 or est - pos > 8.0:
                            clock["pos"], clock["wall"] = pos, t1
                        else:
                            clock["pos"], clock["wall"] = est, t1
                m = mclock.ref(max_age=1.0)
                if m is not None and clock["playing"]:
                    midi_est = m[0] + (t1 - m[1]) / spb
                    if pos - midi_est > 0.25:
                        print(f"MIDI clock behind Live by {pos - midi_est:.2f} beats "
                              f"(missed position message?) — resynced from socket",
                              flush=True)
                        mclock.resync(pos, t1)
            except Exception:
                with clock_lock:
                    clock["playing"] = False
            time.sleep(0.3)

    clock_src = {"midi": False}

    def clock_ref() -> tuple[float, float] | None:
        """(beat, wall) reference for Live's transport: the MIDI clock when it
        is running, else the socket poller. None when stopped / unknown."""
        r = mclock.ref()
        if r is not None:
            if not clock_src["midi"]:
                clock_src["midi"] = True
                print("MIDI clock locked — transport timing now sample-derived",
                      flush=True)
            return r
        if mclock.seen and mclock.playing:
            return None          # clocks stalled (transport stopping)
        with clock_lock:
            if not clock["playing"] or clock["wall"] is None:
                return None
            if time.monotonic() - clock["wall"] > 4.0:
                return None
            return clock["pos"], clock["wall"]

    def clock_now() -> float | None:
        r = clock_ref()
        if r is None:
            return None
        pos, wall = r
        return pos + (time.monotonic() - wall) / spb

    def wall_at(beat: float) -> float | None:
        r = clock_ref()
        if r is None:
            return None
        pos, wall = r
        return wall + (beat - pos) * spb

    def place(origin: float, anchor: float | None, headroom_s: float,
              allow_past: bool = False):
        """Where on Live's grid an answer goes. The answer's rel-0 belongs at
        anchor + origin (anchor: Live beat of the call's start, origin: the
        call's content end in its own beats). Snap that to --sync's grid,
        keeping the fractional remainder as a note shift so the model's
        on-grid notes land on Live's grid.

        allow_past (streaming): the grid start may already be behind the
        playhead — the model's continuation then begins mid-way (its notes
        before now are simply skipped), so a call whose last note ended
        early, or an answer triggered at the bar line, is NOT pushed a whole
        bar later; the model's beat 16 still lands on Live's bar line.
        Otherwise (arrangement writes) slide later by whole grid units until
        the write can beat the playhead.
        Returns (start_beat, shift, exact) or None (no transport clock)."""
        import math
        now_beat = clock_now()
        if now_beat is None or anchor is None or args.sync == "off":
            return None
        q = 4.0 if args.sync == "bar" else 1.0
        exact = anchor + origin
        start_beat = round(exact / q) * q
        earliest = now_beat + headroom_s / spb
        if start_beat < earliest and not allow_past:
            start_beat += math.ceil((earliest - start_beat) / q) * q
        shift = exact - start_beat
        while shift < -0.5 * q:
            shift += q
        return start_beat, shift, exact

    if args.sync != "off":
        threading.Thread(target=_clock_poll, daemon=True).start()

    def _find_model_track() -> int | None:
        try:
            n = int(_ableton("get_session_info").get("track_count", 0))
            for i in range(n):
                if _ableton("get_track_info", {"track_index": i}).get("name") == "model":
                    return i
        except Exception:
            pass
        return None

    def play_clip(plan: dict) -> bool:
        """Write the answer as a session clip on the 'model' track and fire it.
        Live plays the clip from its own timeline — sample-accurate, nothing
        to compensate. Returns False on any failure (caller falls back to
        streaming)."""
        import math
        try:
            # fired session clips only sound while Live's transport runs;
            # free-time jamming (transport stopped) streams over MIDI instead
            if not _ableton("get_arrangement_info").get("is_playing", False):
                return False
            if clip_state["track"] is None:
                clip_state["track"] = _find_model_track()
            if clip_state["track"] is None:
                return False
            origin = plan.get("origin", 0.0)
            base = math.floor(origin)  # integer beat: preserves grid phase
            notes = []
            for n in plan["notes"]:
                start = origin + n["start_time"] - base
                notes.append({"pitch": n["pitch"],
                              "start_time": round(start, 4),
                              "duration": max(0.05, round(n["duration"], 4)),
                              "velocity": n["velocity"]})
            length = max(4.0, math.ceil(max(x["start_time"] + x["duration"]
                                            for x in notes) / 4.0) * 4.0)
            tr = clip_state["track"]
            slot = clip_state["slot"]
            try:
                _ableton("delete_clip", {"track_index": tr, "clip_index": slot})
            except Exception:
                pass
            _ableton("create_clip", {"track_index": tr, "clip_index": slot,
                                     "length": length})
            _ableton("add_notes_to_clip", {"track_index": tr, "clip_index": slot,
                                           "notes": notes})
            try:
                # play once and stop — call-and-response, not a loop
                _ableton("set_clip_loop", {"track_index": tr, "clip_index": slot,
                                           "loop": False})
            except Exception:
                pass
            _ableton("fire_clip", {"track_index": tr, "clip_index": slot})
            clip_state["slot"] = (slot + 1) % clip_state["n_slots"]
            clip_state["fails"] = 0
            print(f"answer -> clip slot {slot} ({len(notes)} notes, "
                  f"{length:.0f} beats, fires on Live's launch quantization)",
                  flush=True)
            return True
        except Exception:
            clip_state["fails"] += 1
            if clip_state["fails"] == 3:
                print("clip output failing — falling back to MIDI streaming",
                      flush=True)
            return False

    def _find_you_track() -> int | None:
        """The track that sends your playing to our input bus."""
        import re as _re
        m = _re.search(r"Bus \d+", args.in_port)
        needle = m.group(0) if m else args.in_port
        try:
            n = int(_ableton("get_session_info").get("track_count", 0))
            for i in range(n):
                r = _ableton("get_track_routing", {"track_index": i})
                if needle in str(r.get("output_routing_type", "")):
                    return i
        except Exception:
            pass
        return None

    def _model_monitor(mode: str) -> None:
        """Arrangement delivery needs the model track DISARMED (arrangement
        recording would overwrite its answer clips) with Monitor Auto (In
        mutes the lane); streaming needs Monitor In and ARMED so Live's
        Record captures the answer. Switch as needed (slow socket calls, so
        only on a change, and never on the answer path's critical section)."""
        if clip_state.get("monitor") == mode:
            return
        try:
            if clip_state["track"] is None:
                clip_state["track"] = _find_model_track()
            if clip_state["track"] is None:
                print("WARNING: no track named 'model' in the Live set — "
                      "run setup_jam_set", flush=True)
                return
            tr = clip_state["track"]
            if mode == "arrange":
                _ableton("set_track_arm", {"track_index": tr, "arm": False})
                _ableton("set_track_monitoring", {"track_index": tr, "state": 1})
            else:
                _ableton("set_track_monitoring", {"track_index": tr, "state": 0})
                _ableton("set_track_arm", {"track_index": tr, "arm": True})
                # exclusive-arm can steal the arm from your track: give it back
                you = _find_you_track()
                if you is not None and you != tr:
                    _ableton("set_track_arm", {"track_index": you, "arm": True})
            clip_state["monitor"] = mode
            print(f"model track -> "
                  f"{'Monitor Auto, disarmed' if mode == 'arrange' else 'Monitor In, armed'}",
                  flush=True)
        except Exception as e:
            print(f"WARNING: could not set the model track's state ({e}); "
                  f"set it by hand: {'Monitor Auto, disarmed' if mode == 'arrange' else 'Monitor In, armed'}",
                  flush=True)

    if args.output == "stream":
        threading.Thread(target=_model_monitor, args=("stream",),
                         daemon=True).start()

    def play_arrange(plan: dict, live_start: float | None = None) -> bool:
        """Write the answer into the arrangement where the call ended —
        Live's timeline plays it sample-accurately, and the jam accumulates
        on the model track's arrangement lane. Needs a running transport.

        live_start: Live beat where the call's first note sounded (when
        known). The answer's origin then sits at live_start + call length,
        snapped to --sync's grid (bar: the answer to a 4-bar call starts on
        the next bar line), and only slides later — by whole grid units — if
        the write can't beat the playhead."""
        import math
        try:
            song_time = clock_now()
            if song_time is None:
                if args.sync == "off":
                    info = _ableton("get_arrangement_info")
                    if not info.get("is_playing", False):
                        return False
                    song_time = float(info["current_song_time"])
                else:
                    return False
            if clip_state["track"] is None:
                clip_state["track"] = _find_model_track()
            if clip_state["track"] is None:
                return False
            _model_monitor("arrange")
            origin = plan.get("origin", 0.0)
            # headroom: the write runs on Live's main thread and its latency
            # varies — lead by 1.5x the recently observed write time (min
            # 0.5s) so the clip lands ahead of the playhead
            lead_s = max(0.5, 1.5 * clip_state["write_s"])
            earliest = math.ceil(song_time + lead_s / spb)
            placed = place(origin, live_start, lead_s)
            if placed is not None:
                start_beat, shift, exact = placed
                where = (f"bar {start_beat / 4 + 1:.2f} (call ended at bar "
                         f"{exact / 4 + 1:.2f})")
            else:
                base = math.floor(origin)       # integer-beat rebase keeps grid phase
                start_beat = earliest
                shift = origin - base
                where = f"beat {start_beat}"
            notes = []
            for n in plan["notes"]:
                st = shift + n["start_time"]
                if st < -0.02:
                    continue                    # belongs before the clip start
                notes.append({"pitch": n["pitch"],
                              "start_time": round(max(0.0, st), 4),
                              "duration": max(0.05, round(n["duration"], 4)),
                              "velocity": n["velocity"]})
            if not notes:
                return False
            length = max(1.0, math.ceil(max(x["start_time"] + x["duration"]
                                            for x in notes)))
            t_w = time.monotonic()
            _ableton("create_arrangement_midi_clip",
                     {"track_index": clip_state["track"], "time": start_beat,
                      "length": length, "notes": notes}, timeout=5.0)
            took = time.monotonic() - t_w
            clip_state["write_s"] = 0.7 * clip_state["write_s"] + 0.3 * took
            now_time = clock_now()
            if now_time is not None and now_time >= start_beat:
                print(f"WARNING: arrangement write landed late (playhead "
                      f"{now_time:.2f} >= clip start {start_beat}, write took "
                      f"{took:.2f}s) — raising headroom", flush=True)
                clip_state["write_s"] += 0.3
            # if a session clip ever took this track over, hand it back to
            # the arrangement so the new clip actually sounds
            try:
                _ableton("set_back_to_arranger")
            except Exception:
                pass
            clip_state["fails"] = 0
            print(f"answer -> arrangement at {where} "
                  f"({len(notes)} notes, {length:.0f} beats)", flush=True)
            return True
        except Exception:
            clip_state["fails"] += 1
            if clip_state["fails"] == 3:
                print("arrangement output failing — falling back to MIDI "
                      "streaming", flush=True)
            return False

    def play_plan(plan: dict, phrase_t0: float | None = None,
                  anchor: float | None = None, headroom: float = 0.05) -> None:
        """Stream the answer over the MIDI bus, each note scheduled on the
        wall clock from the transport clock (MIDI clock: ~ms accurate)."""
        import mido as _m
        if args.output != "stream":
            _model_monitor("stream")
        origin = plan.get("origin", 0.0)
        now = time.monotonic()
        placed = place(origin, anchor, headroom, allow_past=True)
        if placed is not None:
            start_beat, shift, exact = placed
            start = wall_at(start_beat) - comp["value"]
            # notes up to MAX_LATE behind schedule still play (Player policy)
            first = min((shift + n["start_time"] for n in plan["notes"]
                         if start + (shift + n["start_time"]) * spb
                         >= now - Player.MAX_LATE),
                        default=None)
            with clock_lock:
                sock_est = (clock["pos"] + (now - clock["wall"]) / spb
                            if clock["playing"] and clock["wall"] else None)
            agree = (f", socket clock {sock_est - (clock_now() or 0):+.2f} beats"
                     if sock_est is not None else "")
            how = (f"live-clock: grid bar {start_beat / 4 + 1:.2f} (call ended "
                   f"at bar {exact / 4 + 1:.2f}); first note in "
                   f"{(start + first * spb - now) if first is not None else 0:.2f}s"
                   f"{agree}")
            if clock["recording"] and plan["notes"]:
                intended = [start_beat + shift + n["start_time"]
                            for n in sorted(plan["notes"],
                                            key=lambda x: x["start_time"])[:10]]
                total = (max(n["start_time"] + n["duration"]
                             for n in plan["notes"])) * spb
                threading.Timer(max(0.0, start - now) + total + 1.0,
                                _calibrate, args=(intended,)).start()
        elif phrase_t0 is not None:
            # no transport: continue the phrase's own clock — origin belongs
            # at phrase_t0 + origin beats; land on the next congruent beat
            shift = 0.0
            start = phrase_t0 + origin * spb - comp["value"]
            while start < now + headroom:
                start += spb
            how = f"phrase-clock (wait {start - now:.2f}s)"
        else:
            shift = 0.0
            start = now + headroom
            how = "unaligned"
        _onsets = sorted(n["start_time"] for n in plan["notes"])[:12]
        print(f"answer onsets(beats): {[round(o, 2) for o in _onsets]} | {how}",
              flush=True)
        for n in plan["notes"]:
            rel = shift + n["start_time"]
            if rel < -0.02:
                continue                        # belongs before the answer's grid start
            on_t = start + rel * spb
            off_t = on_t + min(n["duration"], plan["target"] - n["start_time"]) * spb
            player.schedule(on_t, _m.Message("note_on", note=n["pitch"],
                                             velocity=n["velocity"]))
            player.schedule(max(off_t, on_t + 0.05),
                            _m.Message("note_off", note=n["pitch"], velocity=0))

    def commit(user_notes_beats: list[dict], plan: dict, how: str) -> None:
        if not plan["notes"]:
            print("no notes generated — keep playing", flush=True)
            return
        if args.context == "session":
            history.append(user_notes_beats)
            history.append(plan["notes"])
        print(f"answered{how}: {len(plan['notes'])} notes / {plan['target']:.0f} beats "
              f"(gen {plan['gen_s']:.2f}s, context {plan['prompt_tokens']} tokens)",
              flush=True)

    def buffer_key():
        # note_ons only: a release after a speculation started must not
        # invalidate the draft (held notes are speculated closed at 'now')
        return (buf.n_on, buf.t0)

    def spec_worker(snapshot: list[dict], key):
        try:
            with gen_lock:
                if buffer_key() != key and spec.get("trigger") != key:
                    return  # user kept playing while we waited for the model
                plan = generate_plan(to_beats(snapshot))
            with spec_lock:
                # still current, or exactly the phrase that just triggered
                if buffer_key() == key or spec.get("trigger") == key:
                    spec.update(key=key, plan=plan)
        finally:
            spec["busy"] = False

    def answer(user_notes_beats: list[dict], key,
               phrase_t0: float | None = None,
               live_start: float | None = None) -> None:
        """Speculative hit -> play the precomputed plan instantly; miss ->
        generate now (streaming schedule as notes decode)."""
        with spec_lock:
            hit = spec["plan"] if spec["key"] == key else None
            spec.update(key=None, plan=None)
        def deliver(plan: dict) -> None:
            if clip_state["fails"] < 3:
                if args.output == "arrange" and play_arrange(plan, live_start):
                    return
                if args.output == "clip" and play_clip(plan):
                    return
            play_plan(plan, phrase_t0, live_start)

        if hit is not None:
            deliver(hit)
            commit(user_notes_beats, hit, " instantly (speculated)")
            return
        import mido as _m
        with gen_lock:
            # a speculative run may have finished while we waited for the lock
            with spec_lock:
                hit = spec["plan"] if spec["key"] == key else None
                spec.update(key=None, plan=None)
            if hit is not None:
                deliver(hit)
                commit(user_notes_beats, hit, " instantly (speculated)")
                return
            if args.output in ("arrange", "clip") and clip_state["fails"] < 3:
                plan = generate_plan(user_notes_beats)
                deliver(plan)
                commit(user_notes_beats, plan, "")
                return
            phrase_beats = max(n["start_time"] + n["duration"] for n in user_notes_beats)
            target = answer_target(phrase_beats)
            prompt_ids, prompt_beats = encode_segments(prompt_segments(user_notes_beats))
            t0 = time.perf_counter()
            _now = time.monotonic()
            shift = 0.0
            placed = place(prompt_beats, live_start, 0.15, allow_past=True)
            if placed is not None:
                start_beat, shift, exact = placed
                start = wall_at(start_beat) - comp["value"]
                print(f"streaming answer as it decodes -> bar {start_beat / 4 + 1:.2f}",
                      flush=True)
            elif phrase_t0 is not None:
                start = phrase_t0 + prompt_beats * spb - comp["value"]
                while start < _now + 0.15:
                    start += spb
            else:
                start = _now + 0.15
            resp = []
            for note in g.stream_notes(prompt_ids, tempo_bpm=60.0 / spb,
                                       max_new_tokens=int(target * 40),
                                       temperature=args.temperature):
                rel_beat = note.start / spb - prompt_beats
                if rel_beat < -1e-6:
                    continue
                if rel_beat >= target:
                    break
                if shift + rel_beat < -0.02:
                    continue                    # before the answer's grid start
                on_t = start + (shift + rel_beat) * spb
                off_t = start + (shift + min(note.end / spb - prompt_beats, target)) * spb
                player.schedule(on_t, _m.Message("note_on", note=note.pitch,
                                                 velocity=note.velocity))
                player.schedule(max(off_t, on_t + 0.05),
                                _m.Message("note_off", note=note.pitch, velocity=0))
                resp.append({"pitch": note.pitch, "start_time": rel_beat,
                             "duration": max(0.05, (note.end - note.start) / spb),
                             "velocity": note.velocity})
            plan = {"notes": resp, "target": target, "prompt_tokens": len(prompt_ids),
                    "gen_s": time.perf_counter() - t0}
        commit(user_notes_beats, plan, "")

    phrase_live: dict = {"start": None}
    # ---------- main loop ---------- #
    try:
        while True:
            for msg in inport.iter_pending():
                if msg.type in ("clock", "start", "continue", "stop", "songpos"):
                    mclock.feed(msg, time.monotonic())
                    if mclock.events:
                        for ev in mclock.events:
                            print(f"transport: {ev}", flush=True)
                        mclock.events.clear()
                    continue
                if msg.type not in ("note_on", "note_off"):
                    continue
                # echo guard: our own answer looping back through a misrouted
                # track (e.g. an armed track with input "All Ins") would
                # otherwise be captured as user playing — a feedback loop
                if player.sent_recently(msg.note):
                    continue
                if buf.t0 is None and msg.type == "note_on" and msg.velocity > 0:
                    print("hearing you...", flush=True)
                    # where on Live's timeline did this phrase start? With
                    # fixed-length calls the phrase is anchored to the nearest
                    # bar line (a pickup counts toward the coming bar).
                    b = clock_now()          # None when the transport is stopped
                    if b is not None and args.call_bars:
                        b = round(b / 4.0) * 4.0
                    elif b is not None and args.anchor == "beat":
                        b = round(b)
                    phrase_live["start"] = b
                buf.feed(msg, time.monotonic())
            now = time.monotonic()
            if args.bpm is None:
                est = mclock.spb()
                if est is not None and buf.t0 is None:
                    adopt_tempo(est)
            n_notes = len(buf.notes)
            trigger = False
            if args.call_bars and buf.t0 is not None and phrase_live["start"] is not None:
                # fixed-length call: everything is decided by the bar line
                call_end = phrase_live["start"] + args.call_bars * 4.0
                beat = clock_now()
                if beat is None:
                    beat = phrase_live["start"] + (now - buf.t0) / spb
                if (args.speculate and beat >= call_end - args.spec_lead
                        and buf.n_on >= args.min_notes and not spec["busy"]
                        and spec["key"] != buffer_key()):
                    spec["busy"] = True
                    threading.Thread(target=spec_worker,
                                     args=(buf.snapshot(now), buffer_key()),
                                     daemon=True).start()
                # fire a hair early so the answer's downbeat note is scheduled
                # in the future instead of played late (measured ~50ms late
                # when triggered exactly at the bar line)
                if beat >= call_end - 0.1:
                    trigger = buf.n_on >= args.min_notes
                    if not trigger:
                        buf.flush()              # too few notes: not a call
            else:
                # adaptive window: a phrase that stops on a whole-bar boundary
                # of its own grid is probably finished — halve the wait
                window = args.silence
                if args.adaptive_silence and buf.t0 is not None and buf.last_event:
                    end_beats = (buf.last_event - buf.t0) / spb
                    if abs(end_beats - round(end_beats / 4.0) * 4.0) < 0.25:
                        window = args.silence * 0.5
                quiet_for = (now - buf.last_event) if buf.last_event else 0.0
                silent = (buf.last_event is not None and quiet_for >= window
                          and not buf.open)
                # speculation: after a short beat of quiet, generate the answer
                # in the background; discarded automatically if more notes arrive
                if (args.speculate and not buf.open and n_notes >= args.min_notes
                        and 0.15 <= quiet_for and not spec["busy"]
                        and spec["key"] != buffer_key()):
                    spec["busy"] = True
                    threading.Thread(target=spec_worker,
                                     args=(list(buf.notes), buffer_key()),
                                     daemon=True).start()
                if silent and n_notes >= args.min_notes:
                    trigger = True
                elif silent and n_notes < args.min_notes:
                    buf.flush()  # discard stray taps
            if trigger or n_notes >= args.max_notes:
                key = buffer_key()
                spec["trigger"] = key  # let an in-flight speculation land post-flush
                phrase_t0 = buf.t0
                notes = buf.flush()
                _onsets = sorted((n["start"] - min(x["start"] for x in notes)) / spb
                                 for n in notes)[:12]
                print(f"phrase captured: {len(notes)} notes | onsets(beats): "
                      f"{[round(o, 2) for o in _onsets]}", flush=True)
                answer(to_beats(notes), key, phrase_t0, phrase_live["start"])
                spec["trigger"] = None
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("jam over.")


if __name__ == "__main__":
    main()
