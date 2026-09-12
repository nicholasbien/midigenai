"""
Diagnostics for the Ableton jam flow — the measurements that found every
timing bug in the 2026-09-11 session, as reusable commands.

    python -m midigenai.jam_diag lanes                 # tracks: arm, routing, devices, clips
    python -m midigenai.jam_diag timing model          # recorded notes vs the model's 1/8-beat grid
    python -m midigenai.jam_diag timing 4 --bpm 128 --after 32
    python -m midigenai.jam_diag socket-latency        # how slow is Live's socket right now?
    python -m midigenai.jam_diag clock-check           # are socket time readings request- or response-time?
    python -m midigenai.jam_diag midi-probe "IAC Driver Bus 1" --secs 8   # what arrives on a port (clock? notes?)
    python -m midigenai.jam_diag ports                 # MIDI ports mido can see

`timing` is the ground truth for "did the answer land where it should": it
reads the notes Live actually recorded (clip boundaries snap and lie; note
positions don't) and reports their distance from the nearest 1/8 beat — the
model's own grid — so a constant offset means delivery latency and scatter
means jitter. Model notes deliberately on 32nd positions show up as ±62 ms
at 120 bpm; look at the median and p90, not the max.
"""

from __future__ import annotations

import argparse
import statistics
import time

from .live_client import live, tracks, track_by_name, arrangement_notes


def cmd_lanes(args):
    for t in tracks():
        print(f"{t['index']:2d} {t['name']:20s} {'ARM' if t['arm'] else '   '} "
              f"in={str(t['input']):24s} out={str(t['output']):20s} "
              f"clips={t['arrangement_clips']:2d} {t['devices']}")


def cmd_timing(args):
    tr = int(args.track) if args.track.isdigit() else track_by_name(args.track)
    if tr is None:
        raise SystemExit(f"no track {args.track!r}")
    bpm = args.bpm or float(live("get_session_info").get("tempo", 120.0))
    spb = 60.0 / bpm
    notes = [n for n in arrangement_notes(tr) if n["start"] >= args.after]
    if not notes:
        print("no notes"); return
    grid = args.grid
    off = [((n["start"] - round(n["start"] * grid) / grid) * spb * 1000.0) for n in notes]
    srt = sorted(off)
    print(f"track {tr}: {len(notes)} notes from beat {args.after}, {bpm:.1f} bpm, grid 1/{grid} beat")
    print(f"  offset from grid: median {statistics.median(off):+.1f} ms, "
          f"p10 {srt[len(srt)//10]:+.1f}, p90 {srt[9*len(srt)//10]:+.1f}, "
          f"worst |{max(abs(o) for o in off):.0f}| ms")
    bars = {}
    for n, o in zip(notes, off):
        bars.setdefault(int(n["start"] // 4) + 1, []).append(o)
    print("  per bar (first note / median ms):",
          {b: (round(v[0]), round(statistics.median(v))) for b, v in list(bars.items())[:24]})


def cmd_socket_latency(args):
    for i in range(args.n):
        t0 = time.monotonic(); live("get_arrangement_info"); t1 = time.monotonic()
        print(f"get_arrangement_info: {t1 - t0:.3f}s")


def cmd_clock_check(args):
    """While the transport runs, compare each reading with dead reckoning from
    the first: err_resp ~0 means readings are current at response time."""
    info = live("get_arrangement_info")
    if not info["is_playing"]:
        raise SystemExit("start Live's transport first")
    spb = 60.0 / float(info["tempo"])
    rows = []
    t_end = time.monotonic() + args.secs
    while time.monotonic() < t_end:
        t0 = time.monotonic(); st = live("get_arrangement_info")["current_song_time"]; t1 = time.monotonic()
        rows.append((t0, t1, st))
    b0, e0, s0 = rows[0]
    print(" rtt   | beat   | err@req | err@resp")
    for t0, t1, st in rows[1:]:
        print(f"{t1-t0:5.2f}s | {st:6.2f} | {st - (s0 + (t0-b0)/spb):+7.2f} | {st - (s0 + (t1-e0)/spb):+8.2f}")


def cmd_midi_probe(args):
    import mido
    seen, first = {}, {}
    with mido.open_input(args.port) as p:
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.secs:
            for m in p.iter_pending():
                seen[m.type] = seen.get(m.type, 0) + 1
                first.setdefault(m.type, f"{time.monotonic()-t0:.2f}s " + str(m))
            time.sleep(0.005)
    print(f"{args.port}: {seen or 'nothing'}")
    for k, v in first.items():
        print(f"  first {k}: {v}")


def cmd_ports(args):
    import mido
    print("inputs :", mido.get_input_names())
    print("outputs:", mido.get_output_names())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lanes").set_defaults(fn=cmd_lanes)
    p = sub.add_parser("timing"); p.add_argument("track"); p.add_argument("--bpm", type=float)
    p.add_argument("--after", type=float, default=0.0, help="ignore notes before this beat")
    p.add_argument("--grid", type=int, default=8, help="grid divisions per beat (8 = the model's)")
    p.set_defaults(fn=cmd_timing)
    p = sub.add_parser("socket-latency"); p.add_argument("--n", type=int, default=5); p.set_defaults(fn=cmd_socket_latency)
    p = sub.add_parser("clock-check"); p.add_argument("--secs", type=float, default=10); p.set_defaults(fn=cmd_clock_check)
    p = sub.add_parser("midi-probe"); p.add_argument("port"); p.add_argument("--secs", type=float, default=6); p.set_defaults(fn=cmd_midi_probe)
    sub.add_parser("ports").set_defaults(fn=cmd_ports)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
