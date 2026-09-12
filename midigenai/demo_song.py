"""
Autonomous jam demo: build a set with a techno / chill-rock backing and six
composed 4-bar calls, then (with --record) run a recorded pass in which
jam.py answers each call live on the 'model' lane.

    python -m midigenai.demo_song                    # chill style, 4-bar calls (120 bpm)
    python -m midigenai.demo_song --style techno     # techno, 1-bar calls (128 bpm), effects
    python -m midigenai.demo_song --record           # ...and record the pass (jam.py must be running)

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


# ======================================================================
# techno style: 128 bpm, 64 bars, twenty 1-bar calls, effects on every lane
# ======================================================================
T_BARS = 64
T_BPM = 128.0
T_CALL_BARS = [9, 11, 13, 15, 17, 19, 21, 23,            # groove
               33, 35, 37, 39,                          # breakdown
               41, 43, 45, 47, 49, 51, 53, 55]          # drop
T_KIT = "query:Drums#FileId_45621"                            # AG Techno Kit
T_PERC = "query:Drums#FileId_45687"                           # Percussion Core Kit
T_STAB = "query:Sounds#Synth%20Lead:FileId_46735"             # Analog Stab Lead
T_PAD = "query:Sounds#Pad:FileId_45215"                       # Dark Corner Pad
T_BASS = "query:Sounds#Bass:FileId_45533"                     # Sub Drive
T_YOU = "query:Sounds#Synth%20Lead:FileId_44976"              # Tech Lead
T_MODEL = "query:Sounds#Synth%20Lead:FileId_80915"            # Classic Club Saw
FX = {"drumbuss": "query:AudioFx#Drum%20Buss", "sat": "query:AudioFx#Saturator",
      "reverb": "query:AudioFx#Reverb", "echo": "query:AudioFx#Echo",
      "autofilter": "query:AudioFx#Auto%20Filter", "glue": "query:AudioFx#Glue%20Compressor",
      "limiter": "query:AudioFx#Limiter"}
GM7, CM7 = [67, 70, 74], [65, 69, 72]                          # Gm / F triads (stab voicings)
G1 = 31


def t_section(bar):                      # 0-based bar -> section name
    if bar < 8: return "intro"
    if bar < 24: return "groove"
    if bar < 32: return "stabs"
    if bar < 40: return "break"
    if bar < 56: return "drop"
    return "outro"


def t_drums(b0, bar):
    sec = t_section(bar); n = []
    kick = sec in ("groove", "stabs", "drop") or (sec == "intro") or (sec == "outro" and bar < 60)
    if kick:
        n += [N(KICK, b0 + beat, 0.25, 127) for beat in range(4)]
    if sec != "break" and bar >= 2:                                   # closed hats: 16ths, velocity ramp
        for i in range(16):
            v = 40 + (i % 4) * 12 if i % 4 else 72
            if i % 4 == 2: continue                                   # open hat sits there
            n.append(N(CH, b0 + i * 0.25, 0.1, v))
        n += [N(OH, b0 + beat + 0.5, 0.3, 92) for beat in range(4)]  # offbeat open hats
    if sec in ("groove", "stabs", "drop") and bar >= 4:
        n += [N(CLAP, b0 + beat, 0.25, 104) for beat in (1, 3)]
    if sec == "drop":
        n += [N(51, b0 + i * 0.5, 0.25, 64) for i in range(8)]       # ride 8ths
    if bar in (8, 24, 32, 40, 56):
        n.append(N(CRASH, b0, 2.0, 100))
    if bar in (39, 55):                                               # snare roll into the next section
        n += [N(SNARE, b0 + 2 + i * 0.125, 0.1, 60 + i * 4) for i in range(16)]
    elif bar % 8 == 7 and sec != "outro":
        n += [N(SNARE, b0 + 3 + i * 0.25, 0.15, 70 + i * 12) for i in range(4)]
    return n


def t_perc(b0, bar):
    sec = t_section(bar)
    if sec in ("intro", "outro") or bar < 12: return []
    # sparse, syncopated: a clave-ish figure, not 16ths
    n = [N(37, b0 + 0.75, 0.1, 72), N(37, b0 + 2.5, 0.1, 64), N(37, b0 + 3.25, 0.1, 58)]
    n += [N(41, b0 + 1.5, 0.15, 80), N(43, b0 + 3.5, 0.15, 70)]
    if sec == "drop": n += [N(45, b0 + 2.25, 0.12, 60)]
    return n


def t_stab(b0, bar):
    sec = t_section(bar)
    if sec in ("intro", "groove") and bar < 16: return []
    chord = GM7 if bar % 8 < 6 else CM7
    hits = [(1.5, 0.25, 100), (3.5, 0.25, 92)]                      # "and" of 2 and of 4
    if sec == "break": hits = [(1.5, 0.9, 92)]
    return [N(p, b0 + t, d, v) for t, d, v in hits for p in chord]


def t_pad(b0, bar):
    if bar % 4 or t_section(bar) in ("drop",): return []
    chord = [p - 12 for p in (GM7 if bar % 8 < 6 else CM7)]
    return [N(p, b0, 15.9, 58) for p in chord]


def t_bass(b0, bar):
    sec = t_section(bar)
    if bar < 4 or sec == "break": return []
    root = G1 if bar % 8 < 6 else G1 + 5
    n = []
    for beat in range(4):                                             # rumble: 16ths off the kick
        for k, v in ((0.25, 96), (0.5, 76), (0.75, 88)):
            n.append(N(root, b0 + beat + k, 0.2, v))
    if bar % 4 == 3: n[-1] = N(root - 2, b0 + 3.75, 0.2, 96)         # slide note into the next 4
    return n


def t_seq(spec, vel=100):
    t, out = 0.0, []
    for p, d in spec:
        if p: out.append(N(p, t, d * 0.8, vel))
        t += d
    return out


# G minor pentatonic hooks (67 70 72 74 77): 3-5 notes, repetitive, done by beat 3.5
R = None
T_CALLS = [
    t_seq([(67, .5), (R, .5), (67, .5), (70, .5), (R, .5), (72, .5), (R, 1)]),
    t_seq([(74, .5), (R, .5), (74, .5), (R, .5), (72, .5), (70, .5), (R, 1)]),
    t_seq([(67, .25), (67, .25), (R, .5), (67, .25), (67, .25), (R, .5), (70, 1), (R, 1)]),
    t_seq([(74, .5), (72, .5), (R, 1), (74, .5), (72, .5), (R, 1)]),
    t_seq([(67, .5), (R, .5), (74, .5), (R, .5), (67, .5), (R, 1.5)]),
    t_seq([(70, .5), (70, .5), (R, .5), (72, .5), (R, .5), (70, .5), (R, 1)]),
    t_seq([(77, .5), (R, .5), (74, .5), (R, .5), (72, 1), (R, 1)]),
    t_seq([(67, .25), (R, .25), (67, .25), (R, .25), (67, .5), (70, .5), (R, 2)]),
    t_seq([(74, 1), (72, 1), (70, 1), (R, 1)]),
    t_seq([(67, .5), (R, 1), (67, .5), (R, 1), (70, .5), (R, .5)]),
    t_seq([(72, .5), (72, .5), (R, .5), (74, .5), (R, .5), (72, .5), (R, 1)]),
    t_seq([(67, .5), (70, .5), (72, .5), (R, .5), (72, .5), (R, 1.5)]),
    t_seq([(79, .5), (R, .5), (77, .5), (R, .5), (74, .5), (R, 1.5)]),
    t_seq([(67, .25), (R, .25), (74, .25), (R, .25), (67, .25), (R, .25), (74, .25), (R, .25), (67, .5), (R, 1.5)]),
    t_seq([(70, .5), (R, .5), (70, .5), (72, .5), (R, .5), (70, .5), (R, 1)]),
    t_seq([(74, .5), (74, .5), (R, .5), (77, .5), (R, .5), (74, .5), (R, 1)]),
    t_seq([(67, .5), (R, .5), (67, .5), (R, .5), (72, .5), (70, .5), (R, 1)]),
    t_seq([(77, .5), (74, .5), (R, .5), (72, .5), (R, .5), (70, .5), (R, 1)]),
    t_seq([(72, .5), (R, .5), (74, .5), (R, .5), (72, .5), (R, 1.5)]),
    t_seq([(67, .5), (R, .5), (70, .5), (R, .5), (74, 1), (R, 1)]),
]


def add_fx(track, *uris):
    for u in uris:
        live("load_instrument_or_effect", {"track_index": track, "uri": u})


def set_param(track, device, name, value):
    """Best effort: parameter names differ per device version (Echo's mix is
    'Dry Wet', Reverb's is 'Dry/Wet')."""
    try:
        params = live("get_device_parameters", {"track_index": track, "device_index": device}).get("parameters", [])
        for prm in params:
            if prm.get("name") in (name, name.replace("/", " ")):
                live("set_device_parameter", {"track_index": track, "device_index": device,
                                              "parameter_index": prm.get("index"), "value": value})
                return True
    except Exception as e:
        print(f"  (param {name} on track {track}: {e})")
    return False


def build_techno() -> dict:
    live("set_tempo", {"tempo": T_BPM})
    live("set_track_name", {"track_index": 0, "name": "drums"})
    live("set_track_name", {"track_index": 1, "name": "perc"})
    live("load_instrument_or_effect", {"track_index": 0, "uri": T_KIT})
    live("load_instrument_or_effect", {"track_index": 1, "uri": T_PERC})
    live("delete_track", {"track_index": 3})
    live("delete_track", {"track_index": 2})
    out = subprocess.run([sys.executable, "-m", "midigenai.setup_jam_set",
                          "--you-instrument", T_YOU, "--model-instrument", T_MODEL],
                         capture_output=True, text=True)
    print(out.stdout.strip())
    you, model = 2, 4
    idx = {}
    for name, uri in (("stab", T_STAB), ("pad", T_PAD), ("bass", T_BASS)):
        i = live("create_midi_track", {"index": -1})["index"]
        live("set_track_name", {"track_index": i, "name": name})
        live("load_instrument_or_effect", {"track_index": i, "uri": uri})
        live("set_track_arm", {"track_index": i, "arm": False})
        idx[name] = i
    live("set_track_arm", {"track_index": model, "arm": True})
    # effects (appended after each instrument)
    add_fx(0, FX["drumbuss"])
    add_fx(idx["bass"], FX["sat"])
    add_fx(idx["stab"], FX["autofilter"], FX["echo"], FX["reverb"])
    add_fx(idx["pad"], FX["reverb"])
    add_fx(3, FX["echo"], FX["reverb"])            # you (sound)
    add_fx(model, FX["echo"], FX["reverb"])
    add_fx(-1, FX["glue"], FX["limiter"])           # master
    for tr, dev in ((idx["stab"], 2), (idx["stab"], 3), (idx["pad"], 1), (3, 1), (3, 2), (model, 1), (model, 2)):
        set_param(tr, dev, "Dry/Wet", 0.26)                       # Echo -> 'Dry Wet', Reverb -> 'Dry/Wet'
    for tr, dev in ((idx["stab"], 2), (3, 1), (model, 1)):
        set_param(tr, dev, "Feedback", 0.4)
    live("set_track_volume", {"track_index": idx["pad"], "volume": 0.62})
    live("set_track_volume", {"track_index": idx["stab"], "volume": 0.7})
    live("set_track_volume", {"track_index": idx["bass"], "volume": 0.8})
    live("set_track_volume", {"track_index": 1, "volume": 0.6})
    live("set_track_volume", {"track_index": model, "volume": 0.75})
    live("set_track_volume", {"track_index": 3, "volume": 0.75})

    chunk = 16
    for c0 in range(0, T_BARS, chunk):
        parts = {0: [], 1: [], idx["stab"]: [], idx["pad"]: [], idx["bass"]: []}
        for bar in range(c0, c0 + chunk):
            rel = (bar - c0) * 4.0
            parts[0] += t_drums(rel, bar); parts[1] += t_perc(rel, bar)
            parts[idx["stab"]] += t_stab(rel, bar); parts[idx["pad"]] += t_pad(rel, bar)
            parts[idx["bass"]] += t_bass(rel, bar)
        for tr, notes in parts.items():
            if notes:
                live("create_arrangement_midi_clip",
                     {"track_index": tr, "time": c0 * 4.0, "length": chunk * 4.0, "notes": notes})
    for bar, notes in zip(T_CALL_BARS, T_CALLS):
        live("create_arrangement_midi_clip",
             {"track_index": you, "time": (bar - 1) * 4.0, "length": 4.0, "notes": notes})
    print(f"built techno: {T_BARS} bars @ {T_BPM:.0f}, 1-bar calls at bars {T_CALL_BARS}")
    return {"you": you, "model": model, "bars": T_BARS}


# ---------------------------------------------------------------------------
# --band: the model also plays drums (1-bar calls) and chords (4-bar calls).
# Three jam.py instances, one bus pair each:
#   melody  you        -> IAC Bus 1 -> jam --role model                     -> Bus 2 -> model
#   drums   you drums  -> IAC Bus 3 -> jam --role "model drums" --drums     -> Bus 4 -> model drums
#   chords  you chords -> IAC Bus 5 -> jam --role "model chords" --call-bars 4 -> Bus 6 -> model chords
# Live's clock (Sync) stays on Bus 1; the other two read it with --clock-port.
# ---------------------------------------------------------------------------
T_CHORD_CALL_BARS = [9, 17, 25, 33, 41, 49]      # 4-bar chord calls; answers fill the next 4


def t_drum_call(bar):
    """The 'top' of the kit for one call bar (kick + hats stay on the backing
    lane): claps, snares, rims, toms — something for the model to answer."""
    v = bar % 4
    pats = [
        [(CLAP, 1, 104), (CLAP, 3, 104), (37, 1.75, 70), (37, 2.5, 66), (SNARE, 3.75, 80)],
        [(CLAP, 1, 104), (SNARE, 2.5, 78), (CLAP, 3, 104), (37, 0.75, 64), (37, 3.25, 64), (45, 3.5, 84)],
        [(CLAP, 1, 104), (CLAP, 3, 104), (SNARE, 3.5, 70), (SNARE, 3.75, 90), (47, 2.75, 76)],
        [(CLAP, 1, 104), (37, 0.5, 60), (37, 1.5, 60), (CLAP, 3, 104), (SNARE, 2.25, 74), (45, 3.75, 88)],
    ]
    return [N(p, t, 0.2, vv) for p, t, vv in pats[v]]


def t_chord_call(k):
    """Four bars of stabs, the chord sequence the model should continue."""
    seqs = [[GM7, GM7, CM7, CM7], [GM7, CM7, GM7, GM7], [GM7, GM7, GM7, CM7]]
    chords = seqs[k % len(seqs)]
    hits = [(1.5, 0.25, 100), (3.5, 0.25, 92)]
    n = []
    for b, chord in enumerate(chords):
        for t, d, v in hits:
            if b == 3 and t == 3.5:
                continue                                  # leave the last half beat clear
            n += [N(p, b * 4 + t, d, v) for p in chord]
    return n


def route(idx, direction, name):
    live(f"set_track_{direction}_routing", {"track_index": idx, "routing_type_name": name})


def dub_chain(tr):
    """Dub techno on a chord lane: LP auto filter with slow LFO, chorus,
    dotted-8th feedback echo, long reverb. Devices are found by name."""
    add_fx(tr, "query:AudioFx#Chorus-Ensemble")
    names = [d["name"] for d in live("get_track_info", {"track_index": tr})["devices"]]
    for di, n in enumerate(names):
        if n == "Auto Filter":
            set_param(tr, di, "Frequency", 0.58); set_param(tr, di, "LFO Amount", 0.12)
        elif n == "Echo":
            # (delay time stays at its dotted-8th default: the API only takes
            # normalized 0-1 values and the '16th' params are raw)
            set_param(tr, di, "Dry Wet", 0.38); set_param(tr, di, "Feedback", 0.58)
            set_param(tr, di, "LP Freq", 0.62); set_param(tr, di, "HP Freq", 0.25)
        elif n == "Reverb":
            set_param(tr, di, "Decay Time", 0.62); set_param(tr, di, "Dry/Wet", 0.32)
            set_param(tr, di, "Room Size", 0.85)


def build_band(idx: dict) -> dict:
    """Extra lanes on top of build_techno(); needs IAC Buses 3-6."""
    you, model = idx["you"], idx["model"]
    lanes = {}
    for name, inst, out_bus, in_bus in (("you drums", None, "Bus 3", None),
                                        ("model drums", T_KIT, None, "Bus 4"),
                                        ("you chords", None, "Bus 5", None),
                                        ("model chords", T_STAB, None, "Bus 6")):
        i = live("create_midi_track", {"index": -1})["index"]
        live("set_track_name", {"track_index": i, "name": name})
        if inst:
            live("load_instrument_or_effect", {"track_index": i, "uri": inst})
        if out_bus:                                          # a call lane: no instrument, out to its bus
            route(i, "output", f"IAC Driver ({out_bus})")
            live("set_track_monitoring", {"track_index": i, "state": 1})   # Auto
            live("set_track_arm", {"track_index": i, "arm": False})
        else:                                                # an answer lane: in from its bus, armed
            route(i, "input", f"IAC Driver ({in_bus})")
            live("set_track_monitoring", {"track_index": i, "state": 1})   # Auto: hear the bus, play back the take
        lanes[name] = i
    # the call lanes carry no instrument (they send MIDI out), so give each a
    # sound lane listening to it — otherwise the composed calls are inaudible
    for name, src, inst in (("you chords (sound)", "you chords", T_STAB),
                            ("you drums (sound)", "you drums", T_KIT)):
        i = live("create_midi_track", {"index": -1})["index"]
        live("set_track_name", {"track_index": i, "name": name})
        live("load_instrument_or_effect", {"track_index": i, "uri": inst})
        route(i, "input", src)
        live("set_track_monitoring", {"track_index": i, "state": 0})       # In
        live("set_track_arm", {"track_index": i, "arm": False})
        lanes[name] = i
    add_fx(lanes["model drums"], FX["drumbuss"])
    add_fx(lanes["you drums (sound)"], FX["drumbuss"])
    for t in (lanes["model chords"], lanes["you chords (sound)"]):
        add_fx(t, FX["autofilter"], FX["echo"], FX["reverb"])
        dub_chain(t)
        live("set_track_volume", {"track_index": t, "volume": 0.7})
    # arm the three answer lanes last (a new track steals the arm)
    for t in (model, lanes["model drums"], lanes["model chords"]):
        live("set_track_arm", {"track_index": t, "arm": True})
    # the composed stab lane made way for chord call/answer: silence it
    stab = next(i for i in range(live("get_session_info")["track_count"])
                if live("get_track_info", {"track_index": i})["name"] == "stab")
    n = live("get_arrangement_clips", {"track_index": stab})["arrangement_clip_count"]
    for _ in range(n):
        live("delete_arrangement_clip", {"track_index": stab, "arrangement_clip_index": 0})
    live("set_track_name", {"track_index": stab, "name": "stab (unused)"})
    # drum calls on the same bars as the melody calls
    for bar in T_CALL_BARS:
        live("create_arrangement_midi_clip", {"track_index": lanes["you drums"], "time": (bar - 1) * 4.0,
                                              "length": 4.0, "notes": t_drum_call(bar)})
    for k, bar in enumerate(T_CHORD_CALL_BARS):
        live("create_arrangement_midi_clip", {"track_index": lanes["you chords"], "time": (bar - 1) * 4.0,
                                              "length": 16.0, "notes": t_chord_call(k)})
    print(f"band lanes: {lanes}; chord calls at bars {T_CHORD_CALL_BARS}")
    print("start the band:\n"
          "  python -m midigenai.jam --call-bars 1 --min-notes 3 --spec-lead 0.5\n"
          "  python -m midigenai.jam --call-bars 1 --min-notes 3 --spec-lead 0.5 --drums "
          "--role 'model drums' --in-port 'IAC Driver Bus 3' --out-port 'IAC Driver Bus 4' "
          "--clock-port 'IAC Driver Bus 1'\n"
          "  python -m midigenai.jam --call-bars 4 --role 'model chords' "
          "--in-port 'IAC Driver Bus 5' --out-port 'IAC Driver Bus 6' --clock-port 'IAC Driver Bus 1'")
    return {**idx, **lanes}


def record_pass(you: int, model: int, bars: int = BARS, extra_you=(), extra_model=()) -> None:
    """Play bars 1-56 with arrangement Record on. The calls play from the
    'you' lane (armed tracks in Auto still send their clips out the bus), so
    'you' is disarmed for the pass or Record would overwrite them."""
    if live("get_arrangement_info")["is_playing"]:
        raise SystemExit("transport is running — stop it first")
    for t in (you, *extra_you):
        live("set_track_arm", {"track_index": t, "arm": False})
    for t in (model, *extra_model):
        live("set_track_arm", {"track_index": t, "arm": True})
        n = live("get_arrangement_clips", {"track_index": t})["arrangement_clip_count"]
        for _ in range(n):
            live("delete_arrangement_clip", {"track_index": t, "arrangement_clip_index": 0})
    live("set_song_time", {"time": 0.0})
    live("set_record_mode", {"on": True})
    live("start_playback")
    tempo = live("get_session_info").get("tempo", 120.0)
    time.sleep(bars * 4 * 60.0 / tempo + 4.0)
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
    ap.add_argument("--style", choices=["chill", "techno"], default="chill")
    ap.add_argument("--band", action="store_true",
                    help="techno only: add drum and chord call/answer lanes (needs IAC Buses 3-6)")
    ap.add_argument("--band-only", action="store_true",
                    help="techno set already built: only add the band lanes")
    args = ap.parse_args()
    bars = T_BARS if args.style == "techno" else BARS
    if args.skip_build or args.band_only:
        idx = {"you": 2, "model": 4}
    else:
        idx = build_techno() if args.style == "techno" else build()
    if args.band or args.band_only:
        idx = build_band(idx)
    if args.record:
        extra_you = [idx[k] for k in ("you drums", "you chords") if k in idx]
        extra_model = [idx[k] for k in ("model drums", "model chords") if k in idx]
        record_pass(idx["you"], idx["model"], bars, extra_you, extra_model)


if __name__ == "__main__":
    main()
