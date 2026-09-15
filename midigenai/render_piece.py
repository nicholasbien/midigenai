"""
Compose-and-render: a hand-written seed in a style -> model continuation over
several seeds -> pick the best by simple health checks (length, no early EOS,
no empty bars, density kept to the end) -> fluidsynth (FluidR3_GM) + ffmpeg mp3.

    python -m midigenai.render_piece --version v3 --seeds 1,2,3,4,5 --temperature 0.9 \
        --out generated/midigenai_ambient_jazz

The seed is the ambient-jazz one used for generated/midigenai_ambient_jazz.mp3
(2026-09-12); add styles by adding CHORDS/MELODY sets. Not a jam tool: see
docs/ABLETON_JAM.md for the live flow.
"""
import sys, argparse, subprocess, statistics
from pathlib import Path
from symusic import Score, Track, Note, Tempo

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="v3")
ap.add_argument("--seeds", default="1,2,3,4,5")
ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--top-k", type=int, default=40)
ap.add_argument("--max-new-tokens", type=int, default=2600)
ap.add_argument("--bpm", type=float, default=62.0)
ap.add_argument("--out", default="generated/midigenai_ambient_jazz")
ap.add_argument("--min-seconds", type=float, default=90.0)
ap.add_argument("--max-seconds", type=float, default=180.0)
ap.add_argument("--soundfont", default="/usr/local/share/soundfonts/FluidR3_GM.sf2")
args = ap.parse_args()

TPQ = 480
def N(p, start, dur, vel): return Note(time=int(round(start * TPQ)), duration=int(round(dur * TPQ)), pitch=p, velocity=vel)

# seed: 8 bars, 2 bars per chord — Cmaj9 | Am11 | Fmaj9(#11) | Dm9 ... G13sus -> lush, soft, sustained
CHORDS = [
    [36, 48, 55, 59, 62, 64],        # Cmaj9: C1 C2 G B D E
    [33, 45, 52, 55, 59, 62],        # Am11: A1 A2 E G B D
    [29, 41, 48, 52, 55, 59],        # Fmaj9#11: F1 F2 C E G B
    [26, 38, 45, 48, 52, 57],        # Dm9: D1 D2 A C E A
]
MELODY = [  # (pitch, start beat, dur) soft, sparse, mostly long tones with a few passing notes
    (76, 1.0, 2.5), (79, 4.0, 1.5), (74, 6.0, 1.5),
    (72, 8.5, 3.0), (71, 12.0, 2.0), (74, 14.5, 1.0),
    (76, 16.5, 2.5), (81, 19.0, 1.0), (79, 20.5, 2.5), (74, 23.0, 1.0),
    (72, 24.5, 3.5), (69, 28.5, 1.5), (67, 30.0, 2.0),
]
def seed_score():
    sc = Score(TPQ); sc.tempos = [Tempo(time=0, qpm=args.bpm)]
    tr = Track(program=0, is_drum=False, name="piano")
    for i, ch in enumerate(CHORDS):
        b0 = i * 8.0
        for k, p in enumerate(ch):
            tr.notes.append(N(p, b0 + 0.04 * k, 7.6, 46 + (4 if p >= 55 else 0)))    # gentle roll, whole 2 bars
        for k, p in enumerate(ch[2:]):                                             # re-voice the top on bar 2
            tr.notes.append(N(p, b0 + 4.0 + 0.05 * k, 3.6, 40))
    for p, s, d in MELODY:
        tr.notes.append(N(p, s, d, 58))
    sc.tracks.append(tr)
    return sc

from midigenai.hub import load_from_hub
g = load_from_hub(version=args.version)
seed_path = Path(args.out + "_seed.mid"); seed_score().dump_midi(seed_path)
prompt_ids = g.encode_midi_file(seed_path)
print(f"checkpoint {args.version} ({g.backend}); seed prompt {len(prompt_ids)} tokens", flush=True)

def analyse(score, n_new):
    spb = 60.0 / args.bpm
    notes = [n for t in score.tracks for n in t.notes]
    if not notes: return None
    end_beats = max(n.start + n.duration for n in notes) / score.ticks_per_quarter
    secs = end_beats * spb
    # voices: notes per bar in the last quarter vs first quarter (dropped voices -> thin tail)
    bars = {}
    for n in notes: bars.setdefault(int(n.start / score.ticks_per_quarter // 4), 0); bars[int(n.start / score.ticks_per_quarter // 4)] += 1
    nb = max(bars) + 1 if bars else 0
    head = statistics.mean(bars.get(b, 0) for b in range(8, min(nb, 8 + max(4, nb // 4)))) if nb > 12 else 0
    tail = statistics.mean(bars.get(b, 0) for b in range(max(8, nb - max(4, nb // 4)), nb)) if nb > 12 else 0
    empty = sum(1 for b in range(8, nb) if bars.get(b, 0) == 0)
    pitches = [n.pitch for n in notes]
    return dict(secs=secs, bars=nb, notes=len(notes), head=head, tail=tail, empty=empty,
                lo=min(pitches), hi=max(pitches), n_new=n_new, tracks=len(score.tracks))

results = []
for seed in [int(x) for x in args.seeds.split(",")]:
    try:
        import mlx.core as mx; mx.random.seed(seed)
    except Exception: pass
    import torch; torch.manual_seed(seed)
    new_ids = list(g.generate_ids(prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k))
    ended = g.eos_id is not None and new_ids and new_ids[-1] == g.eos_id
    score = g.tokenizer.decode(list(prompt_ids) + new_ids)
    score.tempos = [Tempo(time=0, qpm=args.bpm)]
    a = analyse(score, len(new_ids)); a["seed"] = seed; a["eos"] = ended
    p = Path(f"{args.out}_seed{seed}.mid"); score.dump_midi(p); a["path"] = str(p)
    ok = a["secs"] >= args.min_seconds and a["empty"] <= 1 and a["tail"] >= 0.5 * a["head"] and not (ended and a["secs"] < args.min_seconds)
    a["ok"] = ok
    print(f"seed {seed}: {a['n_new']} tok, {a['secs']:.0f}s, {a['bars']} bars, {a['notes']} notes, "
          f"density head {a['head']:.1f}/tail {a['tail']:.1f} per bar, empty bars {a['empty']}, "
          f"range {a['lo']}-{a['hi']}, eos={ended} -> {'OK' if ok else 'reject'}", flush=True)
    results.append(a)

good = [r for r in results if r["ok"]] or results
best = max(good, key=lambda r: (r["ok"], -abs(r["secs"] - 140), r["tail"]))
print("best:", best["seed"], best["path"])
# trim to max_seconds on a bar boundary if needed, render
sc = Score(best["path"])
spb = 60.0 / args.bpm
limit_ticks = int(args.max_seconds / spb) // 4 * 4 * sc.ticks_per_quarter
for t in sc.tracks:
    t.notes = [n for n in t.notes if n.start < limit_ticks]
    for n in t.notes: n.duration = min(n.duration, limit_ticks - n.start)
mid = Path(args.out + ".mid"); sc.dump_midi(mid)
wav = Path(args.out + ".wav")
subprocess.run(["fluidsynth", "-ni", "-g", "0.7", "-r", "44100", "-F", str(wav), args.soundfont, str(mid)], check=True, capture_output=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", "afade=t=out:st=" + str(max(0, best["secs"] - 6)) + ":d=6", "-b:a", "192k", args.out + ".mp3"], check=True)
wav.unlink(missing_ok=True)
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.out + ".mp3"], capture_output=True, text=True).stdout.strip()
print(f"rendered {args.out}.mp3 ({float(dur):.0f}s) from seed {best['seed']}, temperature {args.temperature}, top-k {args.top_k}")
