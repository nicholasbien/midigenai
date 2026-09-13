"""
Minimal client for the AbletonMCP remote script (Live, port 9877).

Every ad-hoc script in the jam work re-implemented this; import it instead:

    from midigenai.live_client import live, tracks, track_by_name, clear_arrangement_clips

    live("get_arrangement_info")
    live("create_arrangement_midi_clip", {"track_index": 2, "time": 16.0, "length": 4.0, "notes": [...]})

Facts worth knowing (measured 2026-09-11, Live 12):
- every command takes 0.4-1.5 s (Live only gives the remote script's Python
  threads time between its ticks); never call this from anything time-critical
- a `current_song_time` reading is current when the RESPONSE arrives under
  light load, but can be seconds stale while Live records or writes clips
- `start_playback` starts from the arrangement INSERT MARKER (the last place
  clicked), not from the playhead set with `set_song_time`; to play from a
  position, start and then `set_song_time` while playing
- `set_device_parameter` takes normalized 0-1 values only; parameter names
  vary per device (Echo's mix is 'Dry Wet', Reverb's is 'Dry/Wet')
- there is no save command
- Live auto-arms a newly created track (stealing the arm from the current one)
"""

from __future__ import annotations

import json
import socket

PORT = 9877


class LiveError(RuntimeError):
    pass


def live(cmd: str, params: dict | None = None, timeout: float = 30.0, port: int = PORT):
    """Send one command, return its `result` (raises LiveError on error)."""
    sk = socket.socket()
    sk.settimeout(timeout)
    try:
        sk.connect(("localhost", port))
    except OSError as e:
        raise LiveError(f"can't reach Live on localhost:{port} — is Live running with the "
                        f"AbletonMCP control surface enabled? ({e})") from e
    sk.sendall(json.dumps({"type": cmd, "params": params or {}}).encode())
    buf = b""
    while True:
        chunk = sk.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
        try:
            r = json.loads(buf)
            break
        except ValueError:
            continue
    sk.close()
    if r.get("status") == "error":
        raise LiveError(f"{cmd}: {r.get('message')}")
    return r.get("result", r)


def tracks() -> list[dict]:
    """[{index, name, arm, input, output, devices, arrangement_clips}] for every track."""
    n = int(live("get_session_info").get("track_count", 0))
    out = []
    for i in range(n):
        t = live("get_track_info", {"track_index": i})
        try:
            r = live("get_track_routing", {"track_index": i})
        except LiveError:
            r = {}
        clips = live("get_arrangement_clips", {"track_index": i}).get("clips", [])
        out.append({"index": i, "name": t.get("name"), "arm": bool(t.get("arm")),
                    "input": r.get("input_routing_type"), "output": r.get("output_routing_type"),
                    "devices": [d["name"] for d in t.get("devices", [])],
                    "arrangement_clips": len(clips)})
    return out


def track_by_name(name: str) -> int | None:
    n = int(live("get_session_info").get("track_count", 0))
    for i in range(n):
        if live("get_track_info", {"track_index": i}).get("name") == name:
            return i
    return None


def clear_arrangement_clips(track: int, start_at: float = 0.0) -> int:
    """Delete arrangement clips on `track` starting at or after `start_at` beats."""
    clips = live("get_arrangement_clips", {"track_index": track}).get("clips", [])
    n = 0
    for i in range(len(clips) - 1, -1, -1):
        if clips[i]["start_time"] >= start_at:
            live("delete_arrangement_clip", {"track_index": track, "arrangement_clip_index": i})
            n += 1
    return n


def arrangement_notes(track: int) -> list[dict]:
    """All notes on a track's arrangement lane with ABSOLUTE beat positions
    ({pitch, start, duration, velocity, clip_start})."""
    clips = live("get_arrangement_clips", {"track_index": track}).get("clips", [])
    out = []
    for i, c in enumerate(clips):
        notes = live("get_arrangement_clip_notes",
                     {"track_index": track, "arrangement_clip_index": i}).get("notes", [])
        for n in notes:
            out.append({"pitch": n["pitch"], "start": c["start_time"] + n["start_time"],
                        "duration": n.get("duration", 0.0), "velocity": n.get("velocity", 0),
                        "clip_start": c["start_time"]})
    return sorted(out, key=lambda n: n["start"])


def set_param(track: int, device: int, name: str, value: float) -> bool:
    """Set a device parameter by name (normalized 0-1). Tries `name` and the
    slash-less spelling (Echo: 'Dry Wet', Reverb: 'Dry/Wet')."""
    params = live("get_device_parameters",
                  {"track_index": track, "device_index": device}).get("parameters", [])
    for i, prm in enumerate(params):
        if prm.get("name") in (name, name.replace("/", " ")):
            live("set_device_parameter", {"track_index": track, "device_index": device,
                                          "parameter_index": prm.get("index", i), "value": value})
            return True
    return False
