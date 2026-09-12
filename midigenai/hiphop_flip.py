"""
Flip a solo-piano midigenai take into a lo-fi boom-bap beat.

    python -m midigenai.hiphop_flip generated/midigenai_ambient_jazz_seed1.mid \
        generated/midigenai_ambient_jazz_hiphop_flip [--bpm 82] [--analyze]

Pipeline: chop chord/melody passages out of the piano MIDI (source-bar ranges,
see --loop-*) -> re-time them 2x onto the beat grid (one source bar = two
beats, so a 2-bar chord becomes a 1-bar chord) -> arrange sections -> render
piano (+vibes doubling the top line) and two GM drum kits with fluidsynth ->
numpy DSP: synthesized 808 sub on the chord roots, tape wow/flutter,
section-dependent low-pass, saturation, plate-ish reverb, sidechain ducking on
kicks, vinyl hiss/crackle, reverse-piano risers, bus glue -> ffmpeg master
(fade, loudnorm -14 LUFS, 192k mp3).

--analyze prints the per-source-bar chord-root / melody table and exits; use it
to pick the --loop-* bar ranges for a new take. The defaults are the ranges
used for generated/midigenai_ambient_jazz_hiphop_flip.mp3 (take 1, 2026-09-12);
see docs/SONG_RECIPES.md. Stems are cached to <out>_stems.npz so --gains can be
re-mixed without re-rendering (--remix).
"""
import os, sys, json, random, subprocess, argparse

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile
from symusic import Score
import mido


ap = argparse.ArgumentParser()
ap.add_argument("inp", help="solo piano .mid (one track)")
ap.add_argument("out", help="output prefix")
ap.add_argument("--bpm", type=float, default=82.0)
ap.add_argument("--soundfont", default=os.environ.get("MIDIGENAI_SOUNDFONT", os.path.expanduser("~/FluidR3_GM/FluidR3_GM.sf2")))
ap.add_argument("--loop-a", default="0:8", help="source bars for loop A (hook), start:end")
ap.add_argument("--loop-b", default="16:24", help="source bars for loop B")
ap.add_argument("--loop-c", default="8:16", help="source bars for the bridge loop")
ap.add_argument("--outro", default="26:30+34:36", help="source bar ranges for the outro, '+'-joined")
ap.add_argument("--gains", default="{}", help='JSON gain overrides, e.g. \'{"sub":0.4,"piano":1.0}\'')
ap.add_argument("--remix", action="store_true", help="skip rendering; re-mix cached <out>_stems.npz with --gains")
ap.add_argument("--analyze", action="store_true", help="print per-bar chord roots + melody of the source and exit")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

BPM = a.bpm; TPQ = 480; SR = 44100
BEAT = TPQ; BAR = 4 * TPQ
SPB = 60 / BPM                     # seconds per beat
FS_LATENCY = 166                   # samples fluidsynth -F adds before tick 0 (measured 3.8 ms at 44.1k)
rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
OUT = a.out
SF2 = a.soundfont
SRC = a.inp

# ---------------------------------------------------------------- source chops
sc = Score(SRC); stpq = sc.ticks_per_quarter; SBAR = stpq * 4
src = [(n.start, n.end, n.pitch, n.velocity) for n in sc.tracks[0].notes]
NAMES = "C C# D D# E F F# G G# A A# B".split()

if a.analyze:
    def root(t0, t1):
        w = {}
        for s, e, p, v in src:
            ov = min(e, t1) - max(s, t0)
            if ov > 0: w[p] = w.get(p, 0) + ov
        if not w: return "-"
        thr = 0.3 * max(w.values()); return NAMES[min(p for p, x in w.items() if x >= thr) % 12]
    nb = max(e for _, e, _, _ in src) // SBAR + 1
    for bb in range(nb):
        bn = [x for x in src if bb * SBAR <= x[0] < (bb + 1) * SBAR]
        print(f"bar {bb:3d}  root {root(bb * SBAR, bb * SBAR + SBAR // 2):>2}/{root(bb * SBAR + SBAR // 2, (bb + 1) * SBAR):<2}  "
              f"notes {len(bn):2d}  melody {[p for _, _, p, _ in bn if p >= 60]}")
    sys.exit(0)

def rng_(spec):
    x, y = spec.split(":"); return int(x), int(y)
SCALE = (2 * BEAT) / SBAR          # one source bar -> two beats (2x compression)

def chop(b0, b1):
    """source bars [b0,b1) -> list of (tick, dur, pitch, vel) relative to 0"""
    out = []
    for s, e, p, v in src:
        if b0 * SBAR <= s < b1 * SBAR:
            e = min(e, b1 * SBAR)
            out.append((int((s - b0 * SBAR) * SCALE), max(60, int((e - s) * SCALE)), p, v))
    return out

# defaults (take 1): A = C | Am | F | Dm (the seed progression); B = B E G# A G F E D
# (the model's chromatic passage); C = E | F# | A | C; outro = the piece's own C F | C ending.
LOOP_A = chop(*rng_(a.loop_a))
LOOP_B = chop(*rng_(a.loop_b))
LOOP_C = chop(*rng_(a.loop_c))
OUTRO, off = [], 0
for part in a.outro.split("+"):
    b0, b1 = rng_(part)
    OUTRO += [(t + off, d, p, v) for t, d, p, v in chop(b0, b1)]
    off += (b1 - b0) * 2 * BEAT

# ---------------------------------------------------------------- arrangement
# (name, piano loop, n_bars, drum style, piano filter 0..1 (1=open), vibes?)
SECTIONS = [
    ("intro",  LOOP_A, 4, "none",     0.25, False),
    ("A1",     LOOP_A, 8, "boombap",  1.0,  False),
    ("B1",     LOOP_B, 8, "boombap+", 1.0,  True),
    ("bridge", LOOP_C, 4, "half",     0.6,  True),
    ("A2",     LOOP_A, 8, "boombap+", 1.0,  False),
    ("B2",     LOOP_B, 8, "boombap+", 1.0,  True),
    ("outro",  OUTRO,  4, "none",     0.4,  False),
]
piano, vibes = [], []            # (tick, dur, pitch, vel)
drums_ac, drums_808 = [], []     # (tick, note, vel)
sub = []                         # (tick, dur, midi_pitch)
sections = []                    # (name, start_bar, n_bars, style, filt)
bar_cursor = 0
for name, loop, nb, style, filt, use_vibes in SECTIONS:
    start = bar_cursor * BAR
    sections.append((name, bar_cursor, nb, style, filt))
    for rep in range(nb // 4):
        base = start + rep * 4 * BAR
        for t, d, p, v in loop:
            piano.append((base + t, d, p, v))
            if use_vibes and p >= 67:
                vibes.append((base + t, d, p + 12, 70 if rep == 0 else 64))  # vibes double the top line up an octave
    bar_cursor += nb
TOTAL_BARS = bar_cursor
piano.sort()

# roots per beat from the arranged piano (lowest sustained pitch)
def root_at(t0, t1):
    w = {}
    for t, d, p, v in piano:
        ov = min(t + d, t1) - max(t, t0)
        if ov > 0: w[p] = w.get(p, 0) + ov
    if not w: return None
    thr = 0.3 * max(w.values())
    return min(p for p, x in w.items() if x >= thr) % 12
beat_roots = []
last = 0
for b in range(TOTAL_BARS * 4):
    r = root_at(b * BEAT, (b + 1) * BEAT)
    if r is None: r = last
    last = r; beat_roots.append(r)
def sub_pitch(pc):  # E1..D#2 register (28..39)
    p = 28 + ((pc - 4) % 12); return p

# ---------------------------------------------------------------- drum patterns (16th grid)
SIXT = BEAT // 4
SWING = int(SIXT * 0.28)
def hum(): return rng.randint(-int(0.004 * TPQ * BPM / 60), int(0.004 * TPQ * BPM / 60))
def pos(bar, s):
    t = bar * BAR + s * SIXT
    if s % 2 == 1: t += SWING
    return t + hum()
KICK, SNARE, RIM, CLAP, CHH, OHH, CRASH, RIDE = 36, 38, 37, 39, 42, 46, 49, 51

def add_hit(bar, s, note, vel, layers=("ac", "808")):
    t = pos(bar, s)
    if "ac" in layers: drums_ac.append((t, note, vel))
    if "808" in layers: drums_808.append((t, note, vel))

for name, sb, nb, style, filt in sections:
    for i in range(nb):
        bar = sb + i
        last_of_8 = (i % 8 == 7) or (i == nb - 1 and style != "none")
        first = i == 0
        if style == "none":
            # intro/outro: only a sparse rim + hats ghost in the last 2 bars of intro to lead in
            if name == "intro" and i >= 2:
                for s in range(0, 16, 2): add_hit(bar, s, CHH, 38 + (10 if s % 4 == 0 else 0), ("ac",))
            if name == "intro" and i == 3:
                for s in (12, 13, 14, 15): add_hit(bar, s, SNARE, 45 + 12 * (s - 12), ("ac",))
            continue
        if style == "half":
            add_hit(bar, 0, KICK, 110); add_hit(bar, 8, SNARE, 105); add_hit(bar, 8, CLAP, 70, ("808",))
            if i >= 2:
                for s in range(0, 16, 4): add_hit(bar, s, CHH, 60, ("ac",))
                add_hit(bar, 11, KICK, 90)
            if first: add_hit(bar, 0, CRASH, 80, ("ac",))
            continue
        # boom-bap
        kick_pat = rng.choice([[0, 7, 10], [0, 6, 10], [0, 7, 10, 14], [0, 10, 11], [0, 3, 10]]) if i % 2 == 1 else [0, 7, 10]
        for s in kick_pat: add_hit(bar, s, KICK, 118 if s == 0 else 100)
        for s in (4, 12):
            add_hit(bar, s, SNARE, 112); add_hit(bar, s, CLAP, 75, ("808",))
        if rng.random() < 0.5: add_hit(bar, 15 if rng.random() < 0.5 else 7, SNARE, 38, ("ac",))  # ghost
        for s in range(0, 16, 2):
            add_hit(bar, s, CHH, 88 if s % 4 == 0 else 62, ("ac",))
        if style == "boombap+":
            for s in rng.sample([3, 7, 11, 15], 2): add_hit(bar, s, CHH, 48, ("808",))
            if i % 2 == 1: add_hit(bar, 14, OHH, 70, ("ac",))
        elif i % 4 == 3:
            add_hit(bar, 14, OHH, 64, ("ac",))
        if first: add_hit(bar, 0, CRASH, 90, ("ac",))
        if last_of_8:
            # snare roll fill on beat 4, kicks out
            drums_ac[:] = [h for h in drums_ac if not (bar * BAR + 12 * SIXT <= h[0] and h[1] == KICK)]
            for k, s in enumerate((12, 13, 14, 15)):
                add_hit(bar, s, SNARE, 60 + 18 * k, ("ac",))
        # sub bass follows the kicks (root of that beat); first kick long
        for s in kick_pat:
            beat = bar * 4 + s // 4
            p = sub_pitch(beat_roots[beat])
            dur = int(1.6 * BEAT) if s == 0 else int(0.45 * BEAT)
            sub.append((pos(bar, s), dur, p))
    # crash at the start of half/boombap sections handled above

# ---------------------------------------------------------------- MIDI writing + fluidsynth
def write_midi(path, tracks):
    """tracks: list of (channel, program, events[(tick, dur, pitch, vel)]) ; program None for drums"""
    m = mido.MidiFile(ticks_per_beat=TPQ)
    meta = mido.MidiTrack(); meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0)); m.tracks.append(meta)
    for ch, prog, evs in tracks:
        tr = mido.MidiTrack()
        tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
        msgs = []
        for t, d, p, v in evs:
            msgs.append((t, 1, mido.Message("note_on", channel=ch, note=p, velocity=max(1, min(127, v)))))
            msgs.append((t + d, 0, mido.Message("note_off", channel=ch, note=p, velocity=0)))
        msgs.sort(key=lambda x: (x[0], x[1])); last = 0
        for t, _, msg in msgs:
            msg.time = t - last; last = t; tr.append(msg)
        tr.append(mido.MetaMessage("end_of_track", time=0)); m.tracks.append(tr)
    m.save(path)

def render(mid, wav, gain=0.8):
    subprocess.run(["fluidsynth", "-ni", "-g", str(gain), "-r", str(SR), "-F", wav, SF2, mid],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if not a.remix:
  write_midi(f"{OUT}_piano.mid", [(0, 0, piano)])
  write_midi(f"{OUT}_vibes.mid", [(1, 11, vibes)])
  write_midi(f"{OUT}_drums_ac.mid", [(9, 16, [(t, 90, n, v) for t, n, v in drums_ac])])     # Power kit
  write_midi(f"{OUT}_drums_808.mid", [(9, 25, [(t, 90, n, v) for t, n, v in drums_808])])   # TR-808 kit
  for n in ("piano", "vibes", "drums_ac", "drums_808"):
      render(f"{OUT}_{n}.mid", f"{OUT}_{n}.wav")

# ---------------------------------------------------------------- DSP helpers
total_s = TOTAL_BARS * 4 * SPB + 3.0
N = int(total_s * SR)
def load(name):
    sr, x = wavfile.read(f"{OUT}_{name}.wav"); x = x.astype(np.float64) / 32768
    if x.ndim == 1: x = np.stack([x, x], 1)
    x = x[FS_LATENCY:]
    y = np.zeros((N, 2)); m = min(N, len(x)); y[:m] = x[:m]; return y
def tick2s(t): return t / TPQ * SPB
def butter(x, kind, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), kind); return signal.lfilter(b, a, x, axis=0)
def peak(x): return np.abs(x).max() + 1e-9
def sat(x, drive=1.0): return np.tanh(x * drive) / np.tanh(drive)

def tape_wobble(x, depth=0.0025, rate=0.55):
    t = np.arange(N) / SR
    wob = depth * np.sin(2 * np.pi * rate * t) + 0.0006 * np.sin(2 * np.pi * 7.3 * t)
    src_idx = t + wob * (1 / rate) / (2 * np.pi) * 0  # keep simple: map via phase integral below
    # variable-speed playback: position = cumulative sum of (1 + d/dt wob)
    speed = 1 + np.gradient(wob, 1 / SR) * 0.02
    posn = np.cumsum(speed) / SR * SR
    posn = np.clip(posn, 0, N - 1)
    return np.stack([np.interp(posn, np.arange(N), x[:, c]) for c in range(2)], 1)

def reverb(x, wet=0.22, predelay_ms=18, decay=0.78):
    combs = [1557, 1617, 1491, 1422, 1277, 1356]; aps = [225, 556, 441]
    y = np.zeros_like(x); pd = int(predelay_ms / 1000 * SR)
    xin = np.concatenate([np.zeros((pd, 2)), x[:-pd]])
    for i, d in enumerate(combs):
        g = decay ** (d / 1500.0) * 0.83
        b = np.zeros(d + 1); b[0] = 1; a = np.zeros(d + 1); a[0] = 1; a[d] = -g
        sig = signal.lfilter(b, a, xin if i % 2 == 0 else xin[:, ::-1], axis=0)
        y += sig / len(combs)
    for d in aps:
        b = np.zeros(d + 1); a = np.zeros(d + 1); b[0] = -0.5; b[d] = 1; a[0] = 1; a[d] = -0.5
        y = signal.lfilter(b, a, y, axis=0)
    y = butter(y, "low", 4200)
    return x * (1 - wet * 0.5) + y * wet

def env_from_hits(times_s, depth=0.45, att=0.006, rel=0.16):
    e = np.ones(N)
    for ts in times_s:
        i = int(ts * SR)
        if i >= N: continue
        a = int(att * SR); r = int(rel * SR)
        seg = np.concatenate([np.linspace(1, 1 - depth, a), 1 - depth + depth * (1 - np.exp(-np.linspace(0, 5, r)))])
        j = min(N, i + len(seg)); e[i:j] = np.minimum(e[i:j], seg[: j - i])
    return e

def section_env(values):
    """per-section value -> per-sample envelope with 1-bar crossfade"""
    e = np.zeros(N)
    for (name, sb, nb, style, filt), v in zip(sections, values):
        i0 = int(sb * 4 * SPB * SR); i1 = int((sb + nb) * 4 * SPB * SR)
        e[i0:min(i1, N)] = v
    return uniform_filter1d(e, int(SPB * 2 * SR), mode="nearest")

if not a.remix:
    # ---------------------------------------------------------------- piano channel
    p = load("piano")
    p = butter(p, "high", 110)
    p = tape_wobble(p)
    p_dark = butter(butter(p, "low", 1400), "low", 1400)
    p_open = butter(p, "low", 5200)
    f = section_env([s[4] for s in sections])[:, None]
    p = p_dark * (1 - f) + p_open * f
    p = sat(p, 1.3)
    p = reverb(p, wet=0.24)
    kick_times = [tick2s(t) for t, n, v in drums_ac if n == KICK]
    p *= env_from_hits(kick_times, depth=0.4)[:, None]
    p /= peak(p)

    # ---------------------------------------------------------------- vibes
    vb = load("vibes")
    vb = butter(vb, "high", 300); vb = reverb(vb, wet=0.4, decay=0.85)
    # slow stereo chorus via delayed wobble on one side
    vb[:, 1] = tape_wobble(vb, depth=0.004, rate=0.9)[:, 1]
    vb /= peak(vb)

    # ---------------------------------------------------------------- drums
    da = load("drums_ac")
    da = sat(da / peak(da), 2.2)                       # smack
    da = butter(da, "low", 9000)                        # lo-fi top
    d8 = load("drums_808"); d8 = d8 / peak(d8)
    d8 = butter(d8, "high", 60)
    drums = da * 0.95 + d8 * 0.55
    # parallel crush: bit-reduce a copy and blend a little
    crush = np.round(drums * 24) / 24
    drums = drums * 0.85 + crush * 0.15
    drums /= peak(drums)

    # ---------------------------------------------------------------- synth 808 sub
    sb_ = np.zeros((N, 2))
    for t, d, pitch in sub:
        f0 = 440 * 2 ** ((pitch - 69) / 12)
        dur = tick2s(d) + 0.08; n = int(dur * SR); i0 = int(tick2s(t) * SR)
        if i0 >= N: continue
        tt = np.arange(n) / SR
        glide = f0 * (1 + 0.9 * np.exp(-tt / 0.03))          # pitch drop at the onset
        phase = 2 * np.pi * np.cumsum(glide) / SR
        env = np.minimum(1, tt / 0.004) * np.exp(-tt / (dur * 0.9))
        env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
        x = np.sin(phase) * env
        x = sat(x * 1.6, 1.6) * 0.9 + 0.1 * np.sin(2 * phase) * env   # a little 2nd harmonic
        j = min(N, i0 + n); sb_[i0:j, 0] += x[: j - i0]; sb_[i0:j, 1] += x[: j - i0]
    sb_ = butter(sb_, "low", 180, order=2); sb_ /= peak(sb_)

    # ---------------------------------------------------------------- vinyl crackle + hiss
    noise = nrng.standard_normal((N, 2))
    hiss = butter(butter(noise, "low", 6000), "high", 1500) * 0.004
    pops = np.zeros((N, 2))
    for _ in range(int(total_s * 9)):
        i = nrng.integers(0, N - 400); L = nrng.integers(30, 300); amp = nrng.uniform(0.05, 0.35) * nrng.choice([-1, 1])
        pops[i:i + L, :] += amp * np.exp(-np.arange(L) / (L / 4))[:, None]
    pops = butter(pops, "low", 3000)
    vinyl = (hiss + pops * 0.9)
    vinyl *= section_env([1.6, 1.0, 1.0, 1.4, 1.0, 1.0, 1.8])[:, None]

    # ---------------------------------------------------------------- reverse-piano riser into the drops
    riser = np.zeros((N, 2))
    for name, sb, nb, style, filt in sections:
        if name in ("A1", "A2", "B2"):
            end = int(sb * 4 * SPB * SR); L = int(1.5 * SPB * SR); seg = p[end:end + L][::-1].copy()
            seg *= np.linspace(0, 1, L)[:, None] ** 2
            riser[end - L:end] += seg * 0.8

    np.savez(f"{OUT}_stems.npz", drums=drums.astype(np.float32), sub=sb_.astype(np.float32), piano=p.astype(np.float32), vibes=vb.astype(np.float32), vinyl=vinyl.astype(np.float32), riser=riser.astype(np.float32), sections=np.array([(n, sb, nb) for n, sb, nb, *_ in sections], dtype=object))
    print(f"bars={TOTAL_BARS} secs={TOTAL_BARS*4*SPB:.0f} piano_notes={len(piano)} vibes={len(vibes)} drums_ac={len(drums_ac)} drums_808={len(drums_808)} sub={len(sub)}")
    for s in sections: print("  ", s)

# ---------------------------------------------------------------- mix + master
z = np.load(f"{OUT}_stems.npz", allow_pickle=True)
g = dict(drums=0.95, sub=0.32, piano=1.05, vibes=0.55, vinyl=1.2, riser=0.6)   # balanced by measured RMS, see docs
g.update(json.loads(a.gains))
def rms_db(x, b0, b1):
    seg = x[int(b0 * 4 * SPB * SR):int(b1 * 4 * SPB * SR)].mean(1); return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
mix = sum(z[k].astype(np.float64) * g[k] for k in g)
for k in g: print(f"  {k:6s} gain={g[k]:.2f}  rms bars 4-12 = {rms_db(z[k] * g[k], 4, 12):6.1f} dB")
mix = sat(mix / (peak(mix) * 0.9), 1.15)      # bus glue
mix = mix / peak(mix) * 0.95
wavfile.write(f"{OUT}_mix.wav", SR, (mix * 32767).astype(np.int16))
secs = TOTAL_BARS * 4 * SPB
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{OUT}_mix.wav",
                "-af", f"afade=t=out:st={secs - 5:.1f}:d=5,loudnorm=I=-14:TP=-1.0:LRA=9", "-b:a", "192k", f"{OUT}.mp3"], check=True)
print(f"rendered {OUT}.mp3 ({secs + 3:.0f}s, {BPM:g} bpm, {TOTAL_BARS} bars)")
