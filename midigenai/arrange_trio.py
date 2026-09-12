"""
Add an upright bass + light brushed-style jazz kit under a solo-piano midigenai take.

    python -m midigenai.arrange_trio generated/midigenai_ambient_jazz_seed1.mid \
        generated/midigenai_ambient_jazz_trio_take1 --render [--trim-bars 56] [--fade 8]

The piano is left untouched. Chord roots are inferred per half-bar from the
lowest sustained piano pitch; the bass plays mostly a two-feel (root held for
three beats, chromatic approach note into the next chord) with occasional
root-fifth bars and gentle four-note walks; the kit is a swung ride pattern,
hi-hat pedal on 2 and 4, whisper-level kick on 1 and a sidestick on 4 every
other bar, with small timing/velocity humanisation. Bass enters at bar 3 and
drums at bar 5 by default; both stop on the piano's final bar.

--render runs fluidsynth (FluidR3_GM) + ffmpeg (fade-out, loudnorm -16 LUFS,
192k mp3) on the result. Used for generated/midigenai_ambient_jazz_trio_take*.mp3
(2026-09-12); see docs/SONG_RECIPES.md.
"""
import os, sys, random, argparse, subprocess
from symusic import Score
import mido

ap = argparse.ArgumentParser()
ap.add_argument("inp", help="solo piano .mid (one track)")
ap.add_argument("out", help="output prefix (writes <out>.mid, and <out>.mp3 with --render)")
ap.add_argument("--trim-bars", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--bass-in", type=int, default=2, help="bar index where bass enters")
ap.add_argument("--drums-in", type=int, default=4, help="bar index where drums enter")
ap.add_argument("--render", action="store_true", help="also render <out>.mp3 with fluidsynth + ffmpeg")
ap.add_argument("--fade", type=float, default=5.0, help="fade-out seconds for --render")
ap.add_argument("--soundfont", default=os.environ.get("MIDIGENAI_SOUNDFONT", os.path.expanduser("~/FluidR3_GM/FluidR3_GM.sf2")))
a = ap.parse_args()
rng = random.Random(a.seed)

sc = Score(a.inp)
src_tpq = sc.ticks_per_quarter
TPQ = 480
K = TPQ / src_tpq
qpm = sc.tempos[0].qpm if sc.tempos else 62
BAR = TPQ * 4
BEAT = TPQ

piano = []
for n in sc.tracks[0].notes:
    s, e = int(round(n.start * K)), int(round(n.end * K))
    piano.append([s, e, n.pitch, n.velocity])
piano.sort()
nbars_full = max(e for _, e, _, _ in piano) // BAR + 1
nbars = min(nbars_full, a.trim_bars) if a.trim_bars else nbars_full
end_tick = nbars * BAR
if a.trim_bars:
    piano = [[s, min(e, end_tick), p, v] for s, e, p, v in piano if s < end_tick]
last_note_bar = max(s for s, _, _, _ in piano) // BAR

# --- harmony: lowest sustained pitch class per half bar --------------------
def root_for(t0, t1):
    w = {}
    for s, e, p, v in piano:
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            w[p] = w.get(p, 0) + ov
    if not w:
        return None
    # candidates = pitches with substantial overlap, take the lowest
    thr = 0.3 * max(w.values())
    return min(p for p, x in w.items() if x >= thr) % 12

def to_bass_range(pc):
    p = 36 + ((pc - 0) % 12)      # C2..B2
    if p > 43: p -= 12            # keep within E1(28)..G2(43)
    return p

roots = []
for b in range(nbars):
    r1 = root_for(b * BAR, b * BAR + BAR // 2)
    r2 = root_for(b * BAR + BAR // 2, (b + 1) * BAR)
    if r1 is None: r1 = roots[-1][1] if roots else 0
    if r2 is None: r2 = r1
    roots.append((r1, r2))

# --- bass -------------------------------------------------------------------
bass = []  # (start, end, pitch, vel)
def jit(x, amt=12): return max(0, x + rng.randint(-amt, amt))
def bn(start, dur, pitch, vel):
    while pitch > 45: pitch -= 12
    while pitch < 28: pitch += 12
    bass.append((jit(start), jit(start) + int(dur * rng.uniform(0.9, 0.98)), pitch, max(1, min(127, vel + rng.randint(-4, 4)))))

for b in range(a.bass_in, min(nbars, last_note_bar + 1)):
    r1, r2 = roots[b]
    nxt = roots[b + 1][0] if b + 1 < nbars else r1
    t = b * BAR
    p1, p2 = to_bass_range(r1), to_bass_range(r2)
    style = rng.random()
    if r1 != r2:
        bn(t, 2 * BEAT, p1, 52); bn(t + 2 * BEAT, 2 * BEAT, p2, 48)
    elif style < 0.55:
        # two-feel: root long, approach note into next bar on beat 4
        bn(t, 3 * BEAT, p1, 54)
        pn = to_bass_range(nxt)
        appr = pn - 1 if rng.random() < 0.6 else pn + 1
        if abs(appr - p1) > 7: appr = p1 + 7  # fifth instead of a big leap
        bn(t + 3 * BEAT, BEAT, appr, 44)
    elif style < 0.8:
        # root, fifth
        bn(t, 2 * BEAT, p1, 54); bn(t + 2 * BEAT, 2 * BEAT, p1 + 7 if p1 + 7 <= 43 else p1 - 5, 46)
    else:
        # gentle walk: root, 3rd-ish(major 3rd or minor via piano? keep 5th), octave, approach
        pn = to_bass_range(nxt)
        line = [p1, p1 + 7 if p1 + 7 <= 43 else p1 - 5, p1 + 12 if p1 + 12 <= 43 else p1 + 5, pn - 1 if rng.random() < 0.5 else pn + 2]
        for i, p in enumerate(line):
            bn(t + i * BEAT, BEAT, p, 50 - 3 * i)

# --- drums (soft, brushes-like) --------------------------------------------
RIDE, HH_PEDAL, KICK, SIDESTICK, RIDE_BELL = 51, 44, 36, 37, 53
drums = []  # (start, note, vel)
def dh(start, note, vel, jitter=10):
    drums.append((max(0, start + rng.randint(-jitter, jitter)), note, max(1, min(127, vel + rng.randint(-5, 5)))))
SW = int(BEAT * 2 / 3)  # swung upbeat
for b in range(a.drums_in, min(nbars, last_note_bar + 1)):
    t = b * BAR
    last = b == min(nbars, last_note_bar + 1) - 1
    if last:
        dh(t, RIDE, 38); dh(t, KICK, 26); break
    for beat in range(4):
        tb = t + beat * BEAT
        dh(tb, RIDE, 40 if beat in (0, 2) else 34)
        if beat in (1, 3):
            dh(tb + SW, RIDE, 28)
            dh(tb, HH_PEDAL, 34)
    dh(t, KICK, 28)
    if rng.random() < 0.35: dh(t + 2 * BEAT + SW, KICK, 20)
    if b % 2 == 1 and rng.random() < 0.8: dh(t + 3 * BEAT, SIDESTICK, 26)
    if b % 8 == 7: dh(t + 3 * BEAT + SW, SIDESTICK, 30)

# --- write ------------------------------------------------------------------
mid = mido.MidiFile(ticks_per_beat=TPQ)
def track(name, ch, prog, events):
    tr = mido.MidiTrack(); tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if ch != 9: tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
    evs = sorted(events, key=lambda x: (x[0], x[1]))
    last = 0
    for tick, _, msg in evs:
        msg.time = tick - last; last = tick; tr.append(msg)
    tr.append(mido.MetaMessage("end_of_track", time=0)); return tr

meta = mido.MidiTrack()
meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(qpm), time=0))
meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
meta.append(mido.MetaMessage("end_of_track", time=0)); mid.tracks.append(meta)

def note_events(notes, ch):
    ev = []
    for s, e, p, v in notes:
        ev.append((s, 1, mido.Message("note_on", channel=ch, note=p, velocity=v)))
        ev.append((max(e, s + 1), 0, mido.Message("note_off", channel=ch, note=p, velocity=0)))
    return ev
mid.tracks.append(track("Acoustic Grand Piano", 0, 0, note_events(piano, 0)))
mid.tracks.append(track("Acoustic Bass", 1, 32, note_events(bass, 1)))
dev = []
for s, n, v in drums:
    dev.append((s, 1, mido.Message("note_on", channel=9, note=n, velocity=v)))
    dev.append((s + 60, 0, mido.Message("note_off", channel=9, note=n, velocity=0)))
mid.tracks.append(track("Drums", 9, 0, dev))
mid.save(a.out + ".mid")
secs = end_tick / TPQ * 60 / qpm
print(f"{a.out}.mid: bars={nbars} (of {nbars_full}) secs={secs:.0f} piano={len(piano)} bass={len(bass)} drums={len(drums)} roots={[r[0] for r in roots[:12]]}")

if a.render:
    wav = a.out + ".wav"
    subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", "44100", "-F", wav, a.soundfont, a.out + ".mid"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
                               capture_output=True, text=True).stdout.strip())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", f"afade=t=out:st={max(0, dur - a.fade)}:d={a.fade},loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-b:a", "192k", a.out + ".mp3"], check=True)
    os.unlink(wav)
    print(f"rendered {a.out}.mp3 ({dur:.0f}s)")
