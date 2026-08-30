"""
Interactive Ableton call-and-response session.

Watches Live for clips you session-record, answers each phrase with a model
continuation on a dedicated "midigenai" track, fired on Live's launch
quantization. Successor to openmusenet2/ableton_bridge.py: local model
instead of a remote flask API, Live's clip/bar clock (via the AbletonMCP
remote script) instead of wall-clock IAC stream parsing, and a running
musical context instead of one-shot prompts.

Requires Ableton Live running with the AbletonMCP control surface enabled
(https://github.com/nicholasbien/ableton-mcp-pro).

Usage:
    python -m v2.live_session                  # model from HF hub
    python -m v2.live_session --answer-bars 4  # fixed answer length
    python -m v2.live_session --instrument "query:Synths#Drift"

Flow:
  1. Session-record a phrase on any track (tip: set launch quantization —
     the Q dropdown in Live's top bar — to 1/4 so answers start on the next
     quarter note instead of the next bar).
  2. When recording stops, the model answers within ~0.5 s, matching your
     phrase length in bars (override with --answer-bars).
  3. Every exchange feeds the next prompt, so the model tracks the session.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import tempfile
import time
from pathlib import Path

DEFAULT_INSTRUMENT = "query:Synths#Electric"


class AbletonClient:
    """Minimal JSON-over-TCP client for the AbletonMCP remote script."""

    def __init__(self, host: str = "localhost", port: int = 9877, timeout: float = 30.0):
        self.sock = socket.socket()
        self.sock.connect((host, port))
        self.sock.settimeout(timeout)

    def send(self, command_type: str, params: dict | None = None):
        self.sock.sendall(json.dumps(
            {"type": command_type, "params": params or {}}).encode())
        buf = b""
        while True:
            buf += self.sock.recv(262144)
            try:
                resp = json.loads(buf)
                break
            except json.JSONDecodeError:
                continue
        if resp.get("status") == "error":
            raise RuntimeError(f"{command_type}: {resp.get('message')}")
        return resp.get("result", resp)

    def clip_notes(self, track: int, slot: int) -> list[dict]:
        return self.send("get_clip_notes",
                         {"track_index": track, "clip_index": slot}).get("notes", [])


def beats_len(notes: list[dict]) -> float:
    return max(n["start_time"] + n["duration"] for n in notes)


def bars_ceil(beats: float, beats_per_bar: float = 4.0) -> float:
    bars = int(beats // beats_per_bar) + (1 if beats % beats_per_bar > 1e-6 else 0)
    return max(1, bars) * beats_per_bar


class LiveSession:
    def __init__(self, generator, ableton: AbletonClient,
                 answer_bars: int | None = None, max_bars: int = 8,
                 instrument_uri: str = DEFAULT_INSTRUMENT, poll_s: float = 0.25):
        self.g = generator
        self.live = ableton
        self.answer_bars = answer_bars
        self.max_bars = max_bars
        self.poll_s = poll_s

        # id -> beats for TimeShift tokens ("TimeShift_a.b.res" = a + b/res beats),
        # used to stop generation once the answer reaches its beat budget
        self.shift_beats: dict[int, float] = {}
        for tok, tid in self.g.tokenizer.vocab.items():
            m = re.match(r"TimeShift_(\d+)\.(\d+)\.(\d+)", tok)
            if m:
                a, b, res = map(int, m.groups())
                self.shift_beats[tid] = a + b / res

        # response track: reuse a track named "midigenai" or create one
        n = int(self.live.send("get_session_info").get("track_count", 0))
        self.resp_track = None
        for i in range(n):
            if self.live.send("get_track_info", {"track_index": i}).get("name") == "midigenai":
                self.resp_track = i
                break
        if self.resp_track is None:
            r = self.live.send("create_midi_track", {"index": -1})
            self.resp_track = r.get("index", n)
            self.live.send("set_track_name",
                           {"track_index": self.resp_track, "name": "midigenai"})
        if not self.live.send("get_track_info",
                              {"track_index": self.resp_track}).get("devices"):
            self.live.send("load_instrument_or_effect",
                           {"track_index": self.resp_track, "uri": instrument_uri})

        # conversation state
        self.history: list[list[dict]] = []
        self.processed: dict[tuple[int, int], int] = {}
        self.was_recording: set[tuple[int, int]] = set()

    # ---------- prompt building ---------- #

    def _encode_history(self, max_prompt_tokens: int = 1500) -> tuple[list[int], float]:
        """Stitch history sequentially into one MIDI and tokenize; drops the
        oldest exchanges until the prompt fits. Returns (ids, total_beats)."""
        from symusic import Score, Track, Note as SNote
        TPQ = 480
        segs = list(self.history)
        while True:
            score = Score(TPQ)
            tr = Track()
            cursor = 0.0
            for seg in segs:
                for n in seg:
                    tr.notes.append(SNote(
                        time=int(round((cursor + n["start_time"]) * TPQ)),
                        duration=max(1, int(round(n["duration"] * TPQ))),
                        pitch=int(n["pitch"]),
                        velocity=int(n.get("velocity", 100)),
                    ))
                cursor += bars_ceil(beats_len(seg))
            score.tracks.append(tr)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp = f.name
            score.dump_midi(tmp)
            ids = self.g.encode_midi_file(tmp)
            Path(tmp).unlink(missing_ok=True)
            if len(ids) <= max_prompt_tokens or len(segs) == 1:
                return ids, cursor

            segs = segs[1:]

    # ---------- answering ---------- #

    def _generate_answer(self, prompt_ids: list[int], prompt_beats: float,
                         target_beats: float) -> list[dict]:
        """Generate until the answer covers target_beats (tracked via TimeShift
        tokens), then decode and trim. Interrupting the stream early is safe:
        the tokenizer just drops any dangling NoteOn."""
        new_ids = []
        elapsed = 0.0
        # hard cap: generous token budget for the beat budget
        max_tokens = int(target_beats * 40)
        for tid in self.g.generate_ids(prompt_ids, max_new_tokens=max_tokens):
            new_ids.append(tid)
            elapsed += self.shift_beats.get(tid, 0.0)
            if elapsed >= target_beats:
                break
        out = self.g.tokenizer.decode(list(prompt_ids) + new_ids)
        otpq = out.ticks_per_quarter
        notes = []
        for tr in out.tracks:
            if tr.is_drum:
                continue
            for n in tr.notes:
                sb = n.start / otpq
                if prompt_beats - 1e-6 <= sb < prompt_beats + target_beats:
                    end = min(sb + n.duration / otpq, prompt_beats + target_beats)
                    notes.append({
                        "pitch": int(n.pitch),
                        "start_time": sb - prompt_beats,
                        "duration": max(0.05, end - sb),
                        "velocity": int(n.velocity),
                    })
        return notes

    def _respond(self, user_notes: list[dict], resp_slot: int):
        self.history.append(user_notes)
        user_beats = bars_ceil(beats_len(user_notes))
        target = (self.answer_bars * 4.0 if self.answer_bars
                  else min(user_beats, self.max_bars * 4.0))
        prompt_ids, prompt_beats = self._encode_history()

        t0 = time.perf_counter()
        resp = self._generate_answer(prompt_ids, prompt_beats, target)
        dt = time.perf_counter() - t0
        if not resp:
            print("model had nothing to say; play a bit more", flush=True)
            self.history.pop()
            return
        self.history.append(resp)

        try:
            self.live.send("delete_clip", {"track_index": self.resp_track,
                                           "clip_index": resp_slot})
        except RuntimeError:
            pass
        self.live.send("create_clip", {"track_index": self.resp_track,
                                       "clip_index": resp_slot, "length": target})
        self.live.send("add_notes_to_clip", {"track_index": self.resp_track,
                                             "clip_index": resp_slot, "notes": resp})
        self.live.send("set_clip_name", {"track_index": self.resp_track,
                                         "clip_index": resp_slot,
                                         "name": f"answer {len(self.history) // 2}"})
        self.live.send("fire_clip", {"track_index": self.resp_track,
                                     "clip_index": resp_slot})
        print(f"answered in {dt:.2f}s: {len(resp)} notes / {target:.0f} beats "
              f"(context {len(prompt_ids)} tokens)", flush=True)

    # ---------- watching ---------- #

    def _user_tracks(self) -> list[int]:
        n = int(self.live.send("get_session_info").get("track_count", 0))
        return [i for i in range(n) if i != self.resp_track]

    def run(self):
        # ignore clips that already exist
        for ti in self._user_tracks():
            for slot in self.live.send("get_track_info",
                                       {"track_index": ti}).get("clip_slots", []):
                if slot.get("has_clip") and not (slot.get("clip") or {}).get("is_recording"):
                    idx = slot.get("index", 0)
                    self.processed[(ti, idx)] = len(self.live.clip_notes(ti, idx))
        print("watching — session-record a phrase on any track "
              "(set launch quantization to 1/4 for snappy answers)", flush=True)
        while True:
            for ti in self._user_tracks():
                tr = self.live.send("get_track_info", {"track_index": ti})
                for slot in tr.get("clip_slots", []):
                    idx = slot.get("index", 0)
                    key = (ti, idx)
                    clip = slot.get("clip") or {}
                    if not slot.get("has_clip"):
                        continue
                    if clip.get("is_recording"):
                        if key not in self.was_recording:
                            print(f"recording on track {ti} slot {idx}...", flush=True)
                        self.was_recording.add(key)
                        continue
                    just_stopped = key in self.was_recording
                    self.was_recording.discard(key)
                    if just_stopped or key not in self.processed:
                        notes = self.live.clip_notes(ti, idx)
                        if not notes or self.processed.get(key) == len(notes):
                            self.processed[key] = len(notes) if notes else 0
                            continue
                        self.processed[key] = len(notes)
                        print(f"heard you: {len(notes)} notes "
                              f"({beats_len(notes):.1f} beats) on track {ti} slot {idx}",
                              flush=True)
                        self._respond(notes, idx)
            time.sleep(self.poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default=None, help="local ckpt (default: HF hub)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--answer-bars", type=int, default=None,
                    help="fixed answer length in bars (default: match your phrase)")
    ap.add_argument("--max-bars", type=int, default=8,
                    help="cap on matched answer length (default 8)")
    ap.add_argument("--instrument", default=DEFAULT_INSTRUMENT,
                    help="browser URI for the answer track's instrument")
    ap.add_argument("--port", type=int, default=9877)
    args = ap.parse_args()

    try:
        ableton = AbletonClient(port=args.port)
    except OSError:
        raise SystemExit(
            "Can't reach Ableton on localhost:%d — is Live running with the "
            "AbletonMCP control surface enabled?" % args.port)

    if args.checkpoint:
        from .generate_v2 import V2Generator
        g = V2Generator(args.checkpoint, args.tokenizer)
    else:
        from .hub import load_v2_from_hub
        g = load_v2_from_hub()
    info = ableton.send("get_session_info")
    session = LiveSession(g, ableton, answer_bars=args.answer_bars,
                          max_bars=args.max_bars, instrument_uri=args.instrument)
    print(f"ready: tempo {info.get('tempo', 120):.0f}, backend {g.backend}, "
          f"answering on track {session.resp_track} ('midigenai')", flush=True)
    try:
        session.run()
    except KeyboardInterrupt:
        print("session over.")


if __name__ == "__main__":
    main()
