"""
Autonomous jam demo: build a set with a techno / chill-rock backing and six
composed 4-bar calls, then (with --record) run a recorded pass in which
jam.py answers each call live on the 'model' lane.

    python -m midigenai.demo_song            # build into the OPEN (blank) Live set
    python -m midigenai.demo_song --record   # ...and record the pass (jam.py must be running)

Song form (56 bars @ 120): 1-4 intro | 6 x (4-bar call, 4-bar answer) from
bar 5 | 53-56 outro. Backing: 909 Core Kit, Organ House pads on Am7 | Fmaj7
| C | C (2 bars each), Dub Techno Bass. Requires the AbletonMCP remote
script (port 9877) with the routing tools, and Live's MIDI clock Sync on
IAC Bus 1 for tight answers (see docs/ABLETON_JAM.md).

Assumes the set is Live's blank template (tracks 1-MIDI, 2-MIDI, 3-Audio,
4-Audio): the two MIDI tracks become drums and organ, the audio tracks are
deleted, setup_jam_set adds you / you (sound) / model, and bass goes last.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time


def live(cmd, params=None, timeout=30):
    sk = socket.socket()
    sk.settimeout(timeout)
    sk.connect(("localhost", 9877))
    sk.sendall(json.dumps({"type": cmd, "params": params or {}}).encode())
    buf = b""
    while True:
        buf += sk.recv(1 << 20)
        try:
            r = json.loads(buf)
            break
        except ValueError:
            continue
    sk.close()
    if r.get("status") == "error":
        raise RuntimeError(f"{cmd}: {r.get('message')}")
    return r.get("result", r)


def N(p, t, d, v):
    return dict(pitch=p, start_time=round(t, 4), duration=round(d, 4), velocity=v)


BARS = 56
CALL_BARS = [5, 13, 21, 29, 37, 45]            # 1-based; each call is 4 bars, answer follows
KICK, SNARE, CLAP, CH, OH, CRASH = 36, 38, 39, 42, 46, 49
CHORDS = [[57, 60, 64, 67], [53, 57, 60, 64], [55, 60, 64, 67], [55, 60, 64, 67]]
ROOTS = [33, 29, 36, 36]
DRUM_KIT = "query:Drums#FileId_45674"                          # 909 Core Kit
ORGAN = "query:Sounds#Piano%20&%20Keys:FileId_45178"           # Organ House
BASS = "query:Sounds#Bass:FileId_47417"                        # Dub Techno Bass
YOU_SOUND = "query:Sounds#Piano%20&%20Keys:FileId_45155"       # E-Piano Classic
MODEL_SOUND = "query:Sounds#Piano%20&%20Keys:FileId_47319"     # Wurli Classic Piano


def drum_bar(b0, bar):
    intro, outro = bar < 4, bar >= 52
    n = []
    if not outro:
        if not intro:
            n += [N(KICK, b0 + beat, 0.3, 122) for beat in range(4)]
            n += [N(CLAP, b0 + beat, 0.3, 100) for beat in (1, 3)]
        for i in range(8):
            t = i * 0.5
            n.append(N(OH, b0 + t, 0.35, 84) if i % 2 else N(CH, b0 + t, 0.12, 66))
        if bar % 8 == 4:
            n.append(N(CRASH, b0, 2.0, 90))
        if bar % 8 == 3:
            n += [N(SNARE, b0 + 3.5 + i * 0.25, 0.2, 76 + i * 10) for i in range(2)]
    return n


def organ_2bars(b0, chord, i, bar):
    if bar < 4:
        return []
    n = [N(p, b0, 7.75, 60) for p in chord]
    if i % 2 == 1:
        n += [N(p + 12, b0 + 7.5, 0.3, 48) for p in chord[1:]]
    return n


def bass_bar(b0, root, bar):
    n = [N(root, b0 + i * 0.5, 0.42, 102 if i % 2 == 0 else 66) for i in range(8)]
    if bar % 4 == 3:
        n[-1] = N(root + 12, b0 + 3.5, 0.42, 74)
    return n


def seq(spec, vel=92):
    """(pitch or None, beats) pairs -> notes; None is a rest. Every call ends
    at least a beat before its bar line so speculation sees the whole call."""
    t, out = 0.0, []
    for p, d in spec:
        if p:
            out.append(N(p, t, d * 0.9, vel))
        t += d
    return out


CALLS = [  # A minor pentatonic
    seq([(81, 1.5), (79, .5), (76, 2), (None, 1), (72, 1), (74, 2), (76, 1.5), (74, .5),
         (72, 2), (None, 1), (69, 2)]),
    seq([(76, .5), (79, .5), (81, 1), (None, 2), (84, 1), (81, 1), (79, 2), (None, 1),
         (76, 1), (74, 2), (None, 2)]),
    seq([(69, 2), (72, 2), (74, 1.5), (76, .5), (74, 2), (None, 2), (72, 1), (69, 1),
         (None, 1)]),
    seq([(84, .5), (81, .5), (79, .5), (76, .5), (None, 2), (79, .5), (76, .5), (74, .5),
         (72, .5), (None, 2), (69, 3), (None, 1)]),
    seq([(76, 1), (None, 1), (76, 1), (79, 1), (81, 2), (None, 2), (79, 1), (76, 1),
         (74, 2), (None, 1), (72, 1), (None, 2)]),
    seq([(81, 3), (79, 1), (76, 3), (74, 1), (72, 2), (74, 2), (69, 2), (None, 2)]),
]


def build() -> dict:
    live("set_track_name", {"track_index": 0, "name": "drums"})
    live("set_track_name", {"track_index": 1, "name": "organ"})
    live("load_instrument_or_effect", {"track_index": 0, "uri": DRUM_KIT})
    live("load_instrument_or_effect", {"track_index": 1, "uri": ORGAN})
    live("delete_track", {"track_index": 3})
    live("delete_track", {"track_index": 2})
    out = subprocess.run([sys.executable, "-m", "midigenai.setup_jam_set",
                          "--you-instrument", YOU_SOUND, "--model-instrument", MODEL_SOUND],
                         capture_output=True, text=True)
    print(out.stdout.strip())
    you, model = 2, 4
    bass = live("create_midi_track", {"index": -1})["index"]
    live("set_track_name", {"track_index": bass, "name": "bass"})
    live("load_instrument_or_effect", {"track_index": bass, "uri": BASS})
    live("set_track_arm", {"track_index": bass, "arm": False})
    live("set_track_arm", {"track_index": model, "arm": True})   # new track steals the arm
    live("set_track_volume", {"track_index": 1, "volume": 0.72})
    live("set_track_volume", {"track_index": bass, "volume": 0.78})

    chunk = 14
    for c0 in range(0, BARS, chunk):
        d, o, b = [], [], []
        for bar in range(c0, c0 + chunk):
            rel = (bar - c0) * 4.0
            ci = (bar // 2) % 4
            d += drum_bar(rel, bar)
            b += bass_bar(rel, ROOTS[ci], bar)
            if bar % 2 == 0:
                o += organ_2bars(rel, CHORDS[ci], ci, bar)
        for tr, notes in ((0, d), (1, o), (bass, b)):
            if notes:
                live("create_arrangement_midi_clip",
                     {"track_index": tr, "time": c0 * 4.0, "length": chunk * 4.0, "notes": notes})
    for bar, notes in zip(CALL_BARS, CALLS):
        live("create_arrangement_midi_clip",
             {"track_index": you, "time": (bar - 1) * 4.0, "length": 16.0, "notes": notes})
    print(f"built: {BARS} bars, calls at bars {CALL_BARS}; tracks you={you} model={model} bass={bass}")
    return {"you": you, "model": model}


def record_pass(you: int, model: int) -> None:
    """Play bars 1-56 with arrangement Record on. The calls play from the
    'you' lane (armed tracks in Auto still send their clips out the bus), so
    'you' is disarmed for the pass or Record would overwrite them."""
    if live("get_arrangement_info")["is_playing"]:
        raise SystemExit("transport is running — stop it first")
    live("set_track_arm", {"track_index": you, "arm": False})
    live("set_track_arm", {"track_index": model, "arm": True})
    n = live("get_arrangement_clips", {"track_index": model})["arrangement_clip_count"]
    for _ in range(n):
        live("delete_arrangement_clip", {"track_index": model, "arrangement_clip_index": 0})
    live("set_song_time", {"time": 0.0})
    live("set_record_mode", {"on": True})
    live("start_playback")
    tempo = live("get_session_info").get("tempo", 120.0)
    time.sleep(BARS * 4 * 60.0 / tempo + 4.0)
    live("stop_playback")
    live("set_record_mode", {"on": False})
    live("set_track_arm", {"track_index": you, "arm": True})
    clips = live("get_arrangement_clips", {"track_index": model})["clips"]
    total = sum(len(live("get_arrangement_clip_notes",
                         {"track_index": model, "arrangement_clip_index": i})["notes"])
                for i in range(len(clips)))
    print(f"recorded pass: {len(clips)} clip(s), {total} answer notes on the model lane "
          f"(save the set: Live has no save command over the socket)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="after building, run the recorded pass (start jam.py "
                         "--call-bars 4 first)")
    ap.add_argument("--skip-build", action="store_true",
                    help="set already built: only run the recorded pass")
    args = ap.parse_args()
    idx = {"you": 2, "model": 4} if args.skip_build else build()
    if args.record:
        record_pass(idx["you"], idx["model"])


if __name__ == "__main__":
    main()
