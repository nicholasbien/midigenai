"""
Interactive MIDI jamming: play, pause, and the model answers — no clicks.

The modern successor to openmusenet2's ableton_bridge.py + note_player.py:
- listens to your playing on a MIDI input port (default "IAC Driver Bus 1")
- when you go silent for --silence seconds (after at least --min-notes),
  the buffered phrase becomes the prompt and generation starts
- the answer STREAMS back out on a MIDI output port (default "IAC Driver
  Bus 2") as the model emits notes — first notes sound almost immediately
- keeps a running conversation (your phrases + its answers) as context

Ableton side (the midi_test template): one track with your instrument,
"MIDI To" -> IAC Bus 1; another track "MIDI From" -> IAC Bus 2, monitor In,
with the model's instrument.

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

    def feed(self, msg, now: float):
        if msg.type == "note_on" and msg.velocity > 0:
            if self.t0 is None:
                self.t0 = now
            self.open[msg.note] = (now - self.t0, msg.velocity)
            self.last_event = now
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
        return notes


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


def main():
    import mido

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--in-port", default="IAC Driver Bus 1")
    ap.add_argument("--out-port", default="IAC Driver Bus 2")
    ap.add_argument("--bpm", type=float, default=120.0)
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
    ap.add_argument("--context", choices=["phrase", "session"], default="phrase",
                    help="prompt with just your latest phrase (default, lowest "
                         "latency) or the whole running session history")
    ap.add_argument("--latency-comp", type=float, default=0.037,
                    help="seconds to schedule answers EARLY, compensating the "
                         "socket/IAC/Live-input delivery chain (measured ~37ms "
                         "from recorded arrangement clips; re-measure with the "
                         "onset diagnostics if your buffer settings change)")
    ap.add_argument("--output", choices=["arrange", "clip", "stream"],
                    default="arrange",
                    help="arrange (default): write each answer into the "
                         "ARRANGEMENT just ahead of the playhead — the timeline "
                         "plays it sample-accurately and the jam accumulates on "
                         "the model track's lane. clip: fire session clips "
                         "instead. stream: raw MIDI over the bus. All modes "
                         "fall back to streaming when the transport is stopped "
                         "or the remote-script socket is unavailable.")
    ap.add_argument("--sync", choices=["off", "beat", "bar"], default="beat",
                    help="delay each answer's start to Live's next beat/bar "
                         "(via the AbletonMCP socket when available) so answers "
                         "land on the transport grid")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    if args.checkpoint:
        from .generate import Generator
        g = Generator(args.checkpoint, args.tokenizer)
    else:
        from .hub import load_from_hub
        g = load_from_hub()

    spb = 60.0 / args.bpm  # seconds per beat

    inport = mido.open_input(args.in_port)
    outport = mido.open_output(args.out_port)
    player = Player(outport)
    print(f"jamming: listening on '{args.in_port}', answering on '{args.out_port}' "
          f"({args.bpm:.0f} bpm, backend {g.backend})", flush=True)
    print(f"play; pause {args.silence}s and the model answers. Ctrl+C to stop.",
          flush=True)

    buf = NoteBuffer()
    history: list[list[dict]] = []   # beat-domain segments (user, model, ...)

    sync_state = {"enabled": args.sync != "off", "fails": 0,
                  "pos": None, "wall": None, "recording": False}
    comp = {"value": args.latency_comp}  # live-tunable latency compensation

    def sync_delay(phase: float = 0.0) -> float | None:
        """Seconds until Live's next beat/bar, or None (transport stopped /
        socket unavailable). Fails open and disables itself after 3 errors."""
        if not sync_state["enabled"]:
            return None
        import json as _json
        import socket as _socket
        try:
            sk = _socket.socket()
            sk.settimeout(0.2)
            sk.connect(("localhost", 9877))
            t_req = time.monotonic()
            sk.sendall(_json.dumps(
                {"type": "get_arrangement_info", "params": {}}).encode())
            raw = b""
            while True:
                chunk = sk.recv(65536)
                if not chunk:
                    break
                raw += chunk
                try:
                    resp = _json.loads(raw)
                    break
                except ValueError:
                    continue
            t_resp = time.monotonic()
            sk.close()
            res = resp.get("result", {})
            song_time = res.get("current_song_time")
            sync_state["recording"] = bool(res.get("record_mode"))
            if song_time is None or not res.get("is_playing", False):
                return None
            sync_state["fails"] = 0
            # RTT correction: song_time was sampled ~mid-round-trip; by now
            # the transport has advanced about half the RTT
            song_time += ((t_resp - t_req) / 2.0) / spb
            sync_state["pos"] = song_time
            sync_state["wall"] = t_resp
            q = 4.0 if args.sync == "bar" else 1.0
            # phase-preserving: the answer's rel-0 sits at content_end, which
            # is generally mid-grid; place it so content_end's grid phase is
            # kept — then the model's on-grid notes land on Live's grid
            target_phase = phase % q
            return ((target_phase - (song_time % q)) % q) * spb
        except Exception:
            sync_state["fails"] += 1
            if sync_state["fails"] >= 3:
                sync_state["enabled"] = False
                print("transport sync disabled (Ableton socket unavailable)",
                      flush=True)
            return None

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
        target = min((int(phrase_beats // 4) + 1) * 4.0, args.max_answer_bars * 4.0)
        prompt_ids, prompt_beats = encode_segments(prompt_segments(user_notes_beats))
        t0 = time.perf_counter()
        notes = []
        for note in g.stream_notes(prompt_ids, tempo_bpm=args.bpm,
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

    def _ableton(cmd, params=None, timeout=1.0):
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
        except Exception:
            pass

    clip_state = {"track": None, "slot": 0, "n_slots": 8, "fails": 0,
                  "write_s": 0.4}  # recent arrangement-write latency (adaptive)

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

    def play_arrange(plan: dict) -> bool:
        """Write the answer into the arrangement just ahead of the playhead —
        Live's timeline plays it sample-accurately, and the jam accumulates
        on the model track's arrangement lane. Needs a running transport."""
        import math
        try:
            info = _ableton("get_arrangement_info")
            if not info.get("is_playing", False):
                return False
            if clip_state["track"] is None:
                clip_state["track"] = _find_model_track()
            if clip_state["track"] is None:
                return False
            song_time = float(info["current_song_time"])
            origin = plan.get("origin", 0.0)
            base = math.floor(origin)  # integer-beat rebase keeps grid phase
            notes = []
            for n in plan["notes"]:
                notes.append({"pitch": n["pitch"],
                              "start_time": round(origin + n["start_time"] - base, 4),
                              "duration": max(0.05, round(n["duration"], 4)),
                              "velocity": n["velocity"]})
            length = max(1.0, math.ceil(max(x["start_time"] + x["duration"]
                                            for x in notes)))
            # headroom: the create call runs on Live's main thread and its
            # latency varies — lead by twice the recently observed write time
            # (min 0.5s) so the clip lands ahead of the playhead
            lead_s = max(0.5, 2.0 * clip_state["write_s"])
            start_beat = math.ceil(song_time + lead_s / spb)
            t_w = time.monotonic()
            _ableton("create_arrangement_midi_clip",
                     {"track_index": clip_state["track"], "time": start_beat,
                      "length": length, "notes": notes}, timeout=3.0)
            clip_state["write_s"] = 0.7 * clip_state["write_s"] + \
                0.3 * (time.monotonic() - t_w)
            now_time = _ableton("get_arrangement_info").get("current_song_time", 0)
            if now_time >= start_beat:
                print(f"WARNING: arrangement write landed late (playhead "
                      f"{now_time:.2f} >= clip start {start_beat}) — raising "
                      f"headroom", flush=True)
                clip_state["write_s"] += 0.3
            # if a session clip ever took this track over, hand it back to
            # the arrangement so the new clip actually sounds
            try:
                _ableton("set_back_to_arranger")
            except Exception:
                pass
            clip_state["fails"] = 0
            print(f"answer -> arrangement at beat {start_beat} "
                  f"({len(notes)} notes, {length:.0f} beats)", flush=True)
            return True
        except Exception:
            clip_state["fails"] += 1
            if clip_state["fails"] == 3:
                print("arrangement output failing — falling back to MIDI "
                      "streaming", flush=True)
            return False

    def play_plan(plan: dict, phrase_t0: float | None = None,
                  headroom: float = 0.05) -> None:
        import mido as _m
        origin = plan.get("origin", 0.0)
        d = sync_delay(origin)
        now = time.monotonic()
        if d is not None:
            start = now + d - comp["value"]
            while start < now + headroom:
                start += (4.0 if args.sync == "bar" else 1.0) * spb
            how = f"live-grid (wait {start - now:.2f}s)"
            if (sync_state["recording"] and sync_state["pos"] is not None
                    and plan["notes"]):
                # intended Live-beat positions of our first notes
                live_at_start = sync_state["pos"] + \
                    (start + comp["value"] - sync_state["wall"]) / spb
                intended = [live_at_start + n["start_time"]
                            for n in sorted(plan["notes"],
                                            key=lambda x: x["start_time"])[:10]]
                total = (max(n["start_time"] + n["duration"]
                             for n in plan["notes"])) * spb
                threading.Timer(max(0.0, start - now) + total + 1.0,
                                _calibrate, args=(intended,)).start()
        elif phrase_t0 is not None:
            # no transport: continue the phrase's own clock — origin belongs
            # at phrase_t0 + origin beats; land on the next congruent beat
            start = phrase_t0 + origin * spb - comp["value"]
            while start < now + headroom:
                start += spb
            how = f"phrase-clock (wait {start - now:.2f}s)"
        else:
            start = now + headroom
            how = "unaligned"
        _onsets = sorted(n["start_time"] for n in plan["notes"])[:12]
        print(f"answer onsets(beats): {[round(o, 2) for o in _onsets]} | {how}",
              flush=True)
        for n in plan["notes"]:
            on_t = start + n["start_time"] * spb
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
        return (len(buf.notes), buf.last_event)

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
               phrase_t0: float | None = None) -> None:
        """Speculative hit -> play the precomputed plan instantly; miss ->
        generate now (streaming schedule as notes decode)."""
        with spec_lock:
            hit = spec["plan"] if spec["key"] == key else None
            spec.update(key=None, plan=None)
        def deliver(plan: dict) -> None:
            if clip_state["fails"] < 3:
                if args.output == "arrange" and play_arrange(plan):
                    return
                if args.output == "clip" and play_clip(plan):
                    return
            play_plan(plan, phrase_t0)

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
            target = min((int(phrase_beats // 4) + 1) * 4.0, args.max_answer_bars * 4.0)
            prompt_ids, prompt_beats = encode_segments(prompt_segments(user_notes_beats))
            t0 = time.perf_counter()
            d = sync_delay(prompt_beats)
            _now = time.monotonic()
            if d is not None:
                start = _now + d - comp["value"]
                while start < _now + 0.15:
                    start += (4.0 if args.sync == "bar" else 1.0) * spb
            elif phrase_t0 is not None:
                start = phrase_t0 + prompt_beats * spb - comp["value"]
                while start < _now + 0.15:
                    start += spb
            else:
                start = _now + 0.15
            resp = []
            for note in g.stream_notes(prompt_ids, tempo_bpm=args.bpm,
                                       max_new_tokens=int(target * 40),
                                       temperature=args.temperature):
                rel_beat = note.start / spb - prompt_beats
                if rel_beat < -1e-6:
                    continue
                if rel_beat >= target:
                    break
                on_t = start + rel_beat * spb
                off_t = start + min(note.end / spb - prompt_beats, target) * spb
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

    # ---------- main loop ---------- #
    try:
        while True:
            for msg in inport.iter_pending():
                if msg.type not in ("note_on", "note_off"):
                    continue
                # echo guard: our own answer looping back through a misrouted
                # track (e.g. an armed track with input "All Ins") would
                # otherwise be captured as user playing — a feedback loop
                if player.sent_recently(msg.note):
                    continue
                if buf.t0 is None and msg.type == "note_on" and msg.velocity > 0:
                    print("hearing you...", flush=True)
                buf.feed(msg, time.monotonic())
            now = time.monotonic()
            n_notes = len(buf.notes)
            # adaptive window: a phrase that stops on a whole-bar boundary of
            # its own grid is probably finished — halve the wait
            window = args.silence
            if args.adaptive_silence and buf.t0 is not None and buf.last_event:
                end_beats = (buf.last_event - buf.t0) / spb
                if abs(end_beats - round(end_beats / 4.0) * 4.0) < 0.25:
                    window = args.silence * 0.5
            quiet_for = (now - buf.last_event) if buf.last_event else 0.0
            silent = (buf.last_event is not None and quiet_for >= window
                      and not buf.open)
            # speculation: after a short beat of quiet, generate the answer in
            # the background; discarded automatically if more notes arrive
            if (args.speculate and not buf.open and n_notes >= args.min_notes
                    and 0.15 <= quiet_for and not spec["busy"]
                    and spec["key"] != buffer_key()):
                spec["busy"] = True
                threading.Thread(target=spec_worker,
                                 args=(list(buf.notes), buffer_key()),
                                 daemon=True).start()
            if (silent and n_notes >= args.min_notes) or n_notes >= args.max_notes:
                key = buffer_key()
                spec["trigger"] = key  # let an in-flight speculation land post-flush
                phrase_t0 = buf.t0
                notes = buf.flush()
                _onsets = sorted((n["start"] - min(x["start"] for x in notes)) / spb
                                 for n in notes)[:12]
                print(f"phrase captured: {len(notes)} notes | onsets(beats): "
                      f"{[round(o, 2) for o in _onsets]}", flush=True)
                answer(to_beats(notes), key, phrase_t0)
                spec["trigger"] = None
            elif silent and n_notes < args.min_notes:
                buf.flush()  # discard stray taps
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("jam over.")


if __name__ == "__main__":
    main()
