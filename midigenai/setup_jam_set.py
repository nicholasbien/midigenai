"""
One-command Ableton set setup for the jam flow (midigenai/jam.py).

Creates and configures two tracks in the open Live set:
  "you"   — your instrument, Monitor Auto, armed, MIDI To -> IAC Bus 1
  "model" — the model's instrument, Monitor In, MIDI From -> IAC Bus 2

Requires the AbletonMCP control surface WITH the routing tools
(ableton-mcp-pro PR #6: get/set_track_input_routing, set_track_output_routing,
set_track_monitoring). Without them it configures what it can and prints the
remaining manual clicks.

Usage:
    python -m midigenai.setup_jam_set
    python -m midigenai.setup_jam_set --you-instrument "query:Synths#Drift"
"""

from __future__ import annotations

import argparse
import json
import socket


class Live:
    def __init__(self, port: int = 9877):
        self.sock = socket.socket()
        self.sock.connect(("localhost", port))
        self.sock.settimeout(20)

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


def pick_routing(available: list[dict], *needles: str) -> str | None:
    for r in available:
        name = r.get("display_name", "")
        if all(n.lower() in name.lower() for n in needles):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--you-instrument", default="query:Synths#Electric")
    ap.add_argument("--model-instrument", default="query:Synths#Electric")
    ap.add_argument("--out-bus", default="Bus 1", help="IAC bus you play into")
    ap.add_argument("--in-bus", default="Bus 2", help="IAC bus answers arrive on")
    ap.add_argument("--port", type=int, default=9877)
    args = ap.parse_args()

    try:
        live = Live(args.port)
    except OSError:
        raise SystemExit("Can't reach Ableton on localhost:%d — is Live running "
                         "with the AbletonMCP control surface enabled?" % args.port)

    manual: list[str] = []

    def make_track(name: str, instrument: str) -> int:
        n = int(live.send("get_session_info").get("track_count", 0))
        idx = live.send("create_midi_track", {"index": -1}).get("index", n)
        live.send("set_track_name", {"track_index": idx, "name": name})
        live.send("load_instrument_or_effect", {"track_index": idx, "uri": instrument})
        return idx

    you = make_track("you", args.you_instrument)
    model = make_track("model", args.model_instrument)

    # routing (needs the routing tools in the remote script)
    have_routing = True
    try:
        routing = live.send("get_track_routing", {"track_index": you})
    except RuntimeError:
        have_routing = False

    if have_routing:
        out_name = pick_routing(routing.get("available_output_routing_types", []),
                                "IAC", args.out_bus)
        if out_name:
            live.send("set_track_output_routing",
                      {"track_index": you, "routing_type_name": out_name})
            print(f"'you' MIDI To -> {out_name}")
        else:
            manual.append(f"'you': set MIDI To -> IAC Driver ({args.out_bus})")

        m_routing = live.send("get_track_routing", {"track_index": model})
        in_name = pick_routing(m_routing.get("available_input_routing_types", []),
                               "IAC", args.in_bus)
        if in_name:
            live.send("set_track_input_routing",
                      {"track_index": model, "routing_type_name": in_name})
            print(f"'model' MIDI From -> {in_name}")
        else:
            manual.append(f"'model': set MIDI From -> IAC Driver ({args.in_bus})")

        live.send("set_track_monitoring", {"track_index": you, "state": 1})    # Auto
        live.send("set_track_monitoring", {"track_index": model, "state": 0})  # In
    else:
        manual += [
            f"'you': MIDI To -> IAC Driver ({args.out_bus}), Monitor Auto",
            f"'model': MIDI From -> IAC Driver ({args.in_bus}), Monitor In",
            "(routing tools not in the loaded remote script — install "
            "ableton-mcp-pro PR #6 and reload the control surface to automate this)",
        ]

    # arm LAST: Live auto-arms newly created tracks, which would otherwise
    # steal the arm from 'you' (this burned a previous session)
    live.send("set_track_arm", {"track_index": model, "arm": False})
    live.send("set_track_arm", {"track_index": you, "arm": True})

    tempo = live.send("get_session_info").get("tempo", 120)
    print(f"tracks ready: 'you' (idx {you}, armed) / 'model' (idx {model}); "
          f"session tempo {tempo:.0f}")
    if manual:
        print("manual steps remaining:")
        for m in manual:
            print("  -", m)
    print(f"now run:  python -m midigenai.jam --bpm {tempo:.0f}")


if __name__ == "__main__":
    main()
