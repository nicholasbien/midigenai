"""
One-command Ableton set setup for the jam flow (midigenai/jam.py).

Creates and configures three tracks in the open Live set (an instrument
track outputs AUDIO, so the play-in track must carry no instrument for the
IAC buses to appear in its output options — hence the split):
  "you"         — NO instrument, armed, Monitor Auto, MIDI To -> IAC Bus 1
  "you (sound)" — your instrument, MIDI From -> IAC Bus 1, Monitor In
  "model"       — the model's instrument, MIDI From -> IAC Bus 2, Monitor Auto,
                  armed (Record captures the model's streamed answers)

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


class Live:
    """Thin wrapper kept for the send() call sites; see live_client.live."""

    def __init__(self, port: int = 9877):
        from .live_client import live as _live
        self._live, self.port = _live, port
        self.send("get_session_info")           # fail fast if Live is unreachable

    def send(self, command_type: str, params: dict | None = None):
        return self._live(command_type, params, timeout=30.0, port=self.port)


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
    ap.add_argument("--you-input", default="auto",
                    help="input for the play-in track: 'auto' picks your hardware "
                         "keyboard if present, else Computer Keyboard — NEVER "
                         "'All Ins', which loops the model's own output back in")
    ap.add_argument("--out-bus", default="Bus 1", help="IAC bus you play into")
    ap.add_argument("--in-bus", default="Bus 2", help="IAC bus answers arrive on")
    ap.add_argument("--port", type=int, default=9877)
    args = ap.parse_args()

    try:
        live = Live(args.port)
    except Exception as e:
        raise SystemExit(str(e))

    manual: list[str] = []

    def make_track(name: str, instrument: str | None) -> int:
        n = int(live.send("get_session_info").get("track_count", 0))
        idx = live.send("create_midi_track", {"index": -1}).get("index", n)
        live.send("set_track_name", {"track_index": idx, "name": name})
        if instrument:
            live.send("load_instrument_or_effect",
                      {"track_index": idx, "uri": instrument})
        return idx

    you = make_track("you", None)          # no instrument: keeps MIDI outputs
    sound = make_track("you (sound)", args.you_instrument)
    model = make_track("model", args.model_instrument)

    have_routing = True
    try:
        routing = live.send("get_track_routing", {"track_index": you})
    except RuntimeError:
        have_routing = False

    if have_routing:
        def route(idx, direction, *needles):
            r = live.send("get_track_routing", {"track_index": idx})
            name = pick_routing(r.get(f"available_{direction}_routing_types", []),
                                *needles)
            if name:
                live.send(f"set_track_{direction}_routing",
                          {"track_index": idx, "routing_type_name": name})
                print(f"track {idx} {direction} -> {name}")
            else:
                manual.append(f"track {idx}: set {direction} routing to "
                              f"IAC ({needles[-1]}) manually")

        route(you, "output", "IAC", args.out_bus)
        # pin the play-in track's INPUT: with "All Ins" (Live's default) the
        # model's Bus 2 output re-enters this armed track and feeds back
        r = live.send("get_track_routing", {"track_index": you})
        avail = [t["display_name"] for t in r.get("available_input_routing_types", [])]
        if args.you_input != "auto":
            chosen = pick_routing(r.get("available_input_routing_types", []),
                                  args.you_input)
        else:
            # exclude buses, meta-inputs, and OTHER TRACKS (their names appear
            # as routable inputs — "1-MIDI" is a track, not a keyboard)
            n_tracks = int(live.send("get_session_info").get("track_count", 0))
            track_names = {live.send("get_track_info", {"track_index": i}).get("name")
                           for i in range(n_tracks)}
            hw = [n for n in avail if n not in ("All Ins", "Computer Keyboard",
                                                "No Input")
                  and "IAC" not in n and n not in track_names]
            chosen = hw[0] if hw else (
                "Computer Keyboard" if "Computer Keyboard" in avail else None)
        if chosen:
            live.send("set_track_input_routing",
                      {"track_index": you, "routing_type_name": chosen})
            print(f"'you' MIDI From -> {chosen}")
        else:
            manual.append("'you': set MIDI From to your keyboard (NOT All Ins)")
        # sound track listens to the out track directly (template style)
        route(sound, "input", "you")
        route(model, "input", "IAC", args.in_bus)
        live.send("set_track_monitoring", {"track_index": you, "state": 1})    # Auto
        live.send("set_track_monitoring", {"track_index": sound, "state": 0})  # In
        # (model's monitoring is set with its arm state below)
    else:
        manual += [
            f"'you': MIDI To -> IAC Driver ({args.out_bus}), Monitor Auto",
            f"'you (sound)': MIDI From -> IAC Driver ({args.out_bus}), Monitor In",
            f"'model': MIDI From -> IAC Driver ({args.in_bus}), Monitor In",
            "(routing tools not loaded — install ableton-mcp-pro PR #6 and "
            "restart Live to automate this)",
        ]

    # arm LAST: Live auto-arms newly created tracks, which would otherwise
    # steal the arm from 'you' (this burned a previous session).
    # 'you' AND 'model' both stay armed (Monitor Auto) so hitting Live's
    # Record captures the whole jam — your part and the model's streamed
    # answers — into the arrangement, and the take plays back afterwards
    # (Monitor In would mute the recorded clips). jam.py --output arrange flips the
    # model track to disarmed / Monitor Auto by itself (arrangement Record
    # would overwrite the clips it writes; Monitor In would mute them).
    live.send("set_track_arm", {"track_index": sound, "arm": False})
    if have_routing:
        live.send("set_track_monitoring", {"track_index": model, "state": 1})  # Auto
    live.send("set_track_arm", {"track_index": model, "arm": True})
    live.send("set_track_arm", {"track_index": you, "arm": True})

    tempo = live.send("get_session_info").get("tempo", 120)
    print(f"tracks ready: 'you' (idx {you}, armed) / 'you (sound)' (idx {sound}) "
          f"/ 'model' (idx {model}); session tempo {tempo:.0f}")
    if manual:
        print("manual steps remaining:")
        for m in manual:
            print("  -", m)
    print(f"now run:  python -m midigenai.jam --bpm {tempo:.0f}")


if __name__ == "__main__":
    main()
