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
    buffer so slightly out-of-order arrivals still play in order."""

    LOOKAHEAD = 0.3  # seconds held back before sending

    def __init__(self, outport):
        self.outport = outport
        self.heap: list[tuple[float, int, object]] = []
        self.lock = threading.Condition()
        self.seq = 0
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
            self.outport.send(msg)


def main():
    import mido

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--in-port", default="IAC Driver Bus 1")
    ap.add_argument("--out-port", default="IAC Driver Bus 2")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--silence", type=float, default=1.5,
                    help="seconds of silence that triggers an answer")
    ap.add_argument("--min-notes", type=int, default=4)
    ap.add_argument("--max-notes", type=int, default=100,
                    help="answer immediately once this many notes are buffered")
    ap.add_argument("--max-answer-bars", type=int, default=8)
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

    def to_beats(notes_secs: list[dict]) -> list[dict]:
        base = min(n["start"] for n in notes_secs)
        return [{
            "pitch": n["pitch"],
            "start_time": (n["start"] - base) / spb,
            "duration": max(0.05, (n["end"] - n["start"]) / spb),
            "velocity": n["velocity"],
        } for n in notes_secs]

    def encode_history() -> tuple[list[int], float]:
        from symusic import Score, Track, Note as SNote
        TPQ = 480
        segs = list(history)
        while True:
            score = Score(TPQ)
            tr = Track()
            cursor = 0.0
            for seg in segs:
                seg_len = max(n["start_time"] + n["duration"] for n in seg)
                seg_len = (int(seg_len // 4) + 1) * 4.0
                for n in seg:
                    tr.notes.append(SNote(
                        time=int(round((cursor + n["start_time"]) * TPQ)),
                        duration=max(1, int(round(n["duration"] * TPQ))),
                        pitch=int(n["pitch"]), velocity=int(n["velocity"])))
                cursor += seg_len
            score.tracks.append(tr)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp = f.name
            score.dump_midi(tmp)
            ids = g.encode_midi_file(tmp)
            Path(tmp).unlink(missing_ok=True)
            if len(ids) <= 1500 or len(segs) == 1:
                return ids, cursor
            segs = segs[1:]

    def answer(user_notes_beats: list[dict]):
        import mido as _m
        history.append(user_notes_beats)
        phrase_beats = max(n["start_time"] + n["duration"] for n in user_notes_beats)
        target = min((int(phrase_beats // 4) + 1) * 4.0, args.max_answer_bars * 4.0)
        prompt_ids, prompt_beats = encode_history()

        t0 = time.perf_counter()
        start = time.monotonic() + 0.15   # tiny scheduling headroom
        resp: list[dict] = []
        n_sent = 0
        for note in g.stream_notes(prompt_ids, tempo_bpm=args.bpm,
                                   max_new_tokens=int(target * 40),
                                   temperature=args.temperature):
            nb = note.start / spb  # stream_notes gives seconds from sequence start
            if nb < prompt_beats - 1e-6:
                continue  # prompt part of the decoded sequence
            rel_beat = nb - prompt_beats
            if rel_beat >= target:
                break
            on_t = start + rel_beat * spb
            off_t = start + min((note.end / spb) - prompt_beats, target) * spb
            player.schedule(on_t, _m.Message("note_on", note=note.pitch,
                                             velocity=note.velocity))
            player.schedule(max(off_t, on_t + 0.05),
                            _m.Message("note_off", note=note.pitch, velocity=0))
            resp.append({"pitch": note.pitch, "start_time": rel_beat,
                         "duration": max(0.05, (note.end - note.start) / spb),
                         "velocity": note.velocity})
            n_sent += 1
        dt = time.perf_counter() - t0
        if resp:
            history.append(resp)
            print(f"answered: {n_sent} notes / {target:.0f} beats "
                  f"(gen {dt:.2f}s, context {len(prompt_ids)} tokens)", flush=True)
        else:
            history.pop()
            print("no notes generated — keep playing", flush=True)

    # ---------- main loop ---------- #
    try:
        while True:
            for msg in inport.iter_pending():
                if msg.type in ("note_on", "note_off"):
                    if buf.t0 is None and msg.type == "note_on" and msg.velocity > 0:
                        print("hearing you...", flush=True)
                    buf.feed(msg, time.monotonic())
            now = time.monotonic()
            n_notes = len(buf.notes)
            silent = (buf.last_event is not None
                      and now - buf.last_event >= args.silence
                      and not buf.open)
            if (silent and n_notes >= args.min_notes) or n_notes >= args.max_notes:
                notes = buf.flush()
                print(f"phrase captured: {len(notes)} notes", flush=True)
                answer(to_beats(notes))
            elif silent and buf.last_event is not None and n_notes < args.min_notes:
                buf.flush()  # discard stray taps
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("jam over.")


if __name__ == "__main__":
    main()
