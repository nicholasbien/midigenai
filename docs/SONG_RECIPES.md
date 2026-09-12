# Song recipes: ambient jazz → trio → hip hop flip

How the `generated/midigenai_ambient_jazz*` pieces (2026-09-12) were made, end to
end, so the same flow can be re-run on a new take. Three CLIs, each one step:

| step | tool | in → out |
|---|---|---|
| 1. compose | `python -m midigenai.render_piece` | hand-written seed → model continuation over N seeds → best one as `.mid` + `.mp3` |
| 2. trio | `python -m midigenai.arrange_trio` | solo-piano `.mid` → piano + upright bass + brushed kit `.mid`/`.mp3` |
| 3. flip | `python -m midigenai.hiphop_flip` | solo-piano `.mid` → lo-fi boom-bap beat `.mp3` (+ stems) |

Everything downstream of the model is seeded, so a command line reproduces a
file: the trio mp3s come back byte-identical; the hip hop flip's mix wav
matches to 1 LSB on a couple of dozen samples and `loudnorm` is repeatable,
but ffmpeg's mp3 encoder is not bit-repeatable even on identical input, so
compare flips by decoding (they match to ≤ 1 LSB, ≈ −86 dB), not by md5.

## Prerequisites

- The midigenai env (`setup.sh`); `requirements.txt` now includes `mido` and
  `scipy` for the two arrangement tools (`numpy` and `symusic` were already there).
- `fluidsynth` and `ffmpeg` on PATH (`brew install fluid-synth ffmpeg`).
- A GM soundfont. All three tools default to `~/FluidR3_GM/FluidR3_GM.sf2`
  (`render_piece` defaults to `/usr/local/share/soundfonts/FluidR3_GM.sf2`); override
  with `--soundfont` or `MIDIGENAI_SOUNDFONT`. The drum-kit program numbers in
  step 3 (Power kit 16, TR-808 kit 25) are FluidR3_GM's.
- Step 1 needs the `v3` checkpoint (pulled from the hub on first use); on Apple
  silicon it runs on MLX.

## 1. Compose the piano piece

```bash
python -m midigenai.render_piece --version v3 --seeds 1,2,3,4,5 \
    --temperature 0.9 --top-k 40 --bpm 62 \
    --out generated/midigenai_ambient_jazz --soundfont ~/FluidR3_GM/FluidR3_GM.sf2
```

What it does:

1. Writes an 8-bar seed prompt (`<out>_seed.mid`): two bars per chord of
   Cmaj9 | Am11 | Fmaj9#11 | Dm9 as soft sustained six-note voicings
   (gently rolled, re-voiced on the second bar) under a sparse long-tone
   melody, at 62 bpm. The voicings are 7.6 beats long so their note-offs
   overhang the bar line — a prompt whose voices all stop at the same instant
   makes the model emit EOS as its first token (see `PLAN.md`).
2. Generates a continuation per seed (`mx.random.seed` / `torch.manual_seed`),
   up to 2600 new tokens, and writes `<out>_seed<N>.mid` for each.
3. Health-checks each candidate: ≥ 90 s long, ≤ 1 empty bar after the
   prompt, note density in the last quarter ≥ half of the density just after
   the prompt (voices didn't thin out), not an early EOS. The winner is the
   healthy take closest to 140 s, trimmed on a bar boundary to `--max-seconds`.
4. Renders it with fluidsynth (gain 0.7) + ffmpeg (6 s fade, 192 kbps).

Result for the ambient-jazz seed: seed 1 won — 684 new tokens, natural EOS at
2:20, 36 bars, solo piano, velocities 39–59, 6.8 → 5.9 notes/bar front to
back. Seed 4 was rejected for empty bars, seeds 3 and 5 ran 5–6 min, seed 2
thinned out at the end.

Keep all the `_seed<N>.mid` files: the later steps pick from them.

## 2. Look at the candidates before arranging

Two checks that mattered, both cheap:

- **Per-bar note density** (`hiphop_flip --analyze` prints roots, note count and
  melody per bar): trailing empty bars mean "trim here" (seed 4 → bar 70), a
  6-minute take means "trim at a phrase boundary" (seed 3 → bar 56, i.e. 3:37).
- **Loop detection**: hash each bar's `(onset, pitch)` set and look for a
  repeating period. Seed 5 turned out to be the 8-bar seed repeated eight times
  — a valid continuation, not a piece. Roughly:

  ```python
  from symusic import Score
  sc = Score("generated/midigenai_ambient_jazz_seed5.mid"); bar = sc.ticks_per_quarter * 4
  sig = {}
  for n in sc.tracks[0].notes: sig.setdefault(n.start // bar, []).append((n.start % bar, n.pitch))
  keys = [tuple(sorted(sig.get(b, []))) for b in range(max(sig) + 1)]
  print(len(keys), "bars,", len(set(keys)), "unique")   # 84 bars, 26 unique -> loop
  ```

Note the model's MIDI comes out at 8 ticks per quarter (the tokenizer's grid);
both arrangement tools rescale to 480 before adding swing or humanisation.

## 3. Trio: add upright bass and brushed drums

```bash
for s in 1 2; do
  python -m midigenai.arrange_trio generated/midigenai_ambient_jazz_seed$s.mid \
      generated/midigenai_ambient_jazz_trio_take$s --seed $((s*11)) --fade 5 --render
done
python -m midigenai.arrange_trio generated/midigenai_ambient_jazz_seed3.mid \
    generated/midigenai_ambient_jazz_trio_take3 --seed 33 --trim-bars 56 --fade 8 --render
python -m midigenai.arrange_trio generated/midigenai_ambient_jazz_seed4.mid \
    generated/midigenai_ambient_jazz_trio_take4 --seed 44 --trim-bars 70 --fade 8 --render
```

The piano is untouched. Per half-bar the tool takes the lowest piano pitch
that sounds for ≥ 30 % of the longest-sounding one as the chord root, then:

- **Bass** (GM 32 acoustic bass, E1–A2): if the root changes mid-bar, two half
  notes; otherwise 55 % two-feel (root held three beats, chromatic approach
  note into the next bar's root on beat 4, capped to a fifth if the leap is
  big), 25 % root–fifth, 20 % a four-note walk root / fifth / octave /
  approach. Note lengths 90–98 %, ±12-tick timing and ±4 velocity jitter.
- **Drums** (channel 10): ride on every beat with a swung upbeat (2/3 beat) on
  2 and 4, hi-hat pedal on 2 and 4, kick on 1 at velocity ~28 (plus a 35 %
  chance of a soft one on the "and" of 3), sidestick on 4 every other bar and
  on the "and" of 4 every 8th bar. Velocities 20–45, ±10 ticks jitter.
- Bass enters at bar 3, drums at bar 5 (`--bass-in` / `--drums-in`); both stop
  on the piano's final bar, where the kit plays a single ride + kick.
- `--render`: fluidsynth gain 0.6 → ffmpeg fade-out (`--fade`) + `loudnorm`
  to −16 LUFS, 192 kbps.

Sanity checks used instead of listening: track programs/ranges via symusic,
`ffmpeg -af volumedetect` (mean ≈ −18 dB, peak ≈ −1.5 dB after loudnorm).

## 4. Hip hop flip

```bash
python -m midigenai.hiphop_flip generated/midigenai_ambient_jazz_seed1.mid \
    generated/midigenai_ambient_jazz_hiphop_flip --bpm 82
```

Takes ~2–3 min (the reverb and wobble run in numpy over ~6M samples). Output is
`<out>.mp3` plus the arranged MIDIs (`<out>_piano.mid`, `_vibes.mid`,
`_drums_ac.mid`, `_drums_808.mid`), the raw fluidsynth stems and a
`<out>_stems.npz` cache of the processed stems.

### 4a. Choosing the samples

Run `--analyze` first; it prints the chord root per half-bar and the melody
notes per bar. For take 1 that gave four usable passages, which are the
defaults:

| flag | source bars | what it is | used for |
|---|---|---|---|
| `--loop-a 0:8` | 0–7 | Cmaj9 · Am11 · Fmaj9#11 · Dm9 (the seed progression) | hook, A sections |
| `--loop-b 16:24` | 16–23 | B → E → G# → A → G → F → E → D, the model's own chromatic passage | B sections |
| `--loop-c 8:16` | 8–15 | E · F# · A · C | bridge |
| `--outro 26:30+34:36` | 26–29 then 34–35 | the piece's own C → F → C ending | outro |

Chops are re-timed 2× (one source bar = two beats at the new tempo) so a
two-bar chord in the original becomes a one-bar chord in the beat, and the
sparse long-tone melody becomes ordinary chop-length phrases.

### 4b. Arrangement (44 bars at 82 bpm ≈ 2:09 + tail)

| section | bars | loop | drums | piano filter |
|---|---|---|---|---|
| intro | 4 | A | none; hats from bar 3, 4-note snare pickup in bar 4 | dark (1.4 kHz) |
| A1 | 8 | A×2 | boom-bap | open (5.2 kHz) |
| B1 | 8 | B×2 | boom-bap+ (extra swung 16th hats, open hat every 2nd bar), vibes on | open |
| bridge | 4 | C | half-time (kick 1, snare 3), hats + extra kick from bar 3 | 60 % |
| A2 | 8 | A×2 | boom-bap+ | open |
| B2 | 8 | B×2 | boom-bap+, vibes on | open |
| outro | 4 | outro | none | 40 %, 5 s fade |

Boom-bap bar: kick on 1 plus a per-bar pattern drawn from
`[0,7,10] [0,6,10] [0,7,10,14] [0,10,11] [0,3,10]` (16th positions, odd bars
only), snare + 808 clap on 2 and 4, 50 % chance of a ghost snare on the last
16th or the "a" of 2, closed hats on every 8th (accented on the beat). Every
8th bar the kick drops out of beat 4 for a crescendo snare roll; crash on each
section's downbeat. Odd 16ths are swung late by 28 % of a 16th; every hit gets
±4 ms humanisation. Reverse-piano risers (1.5 beats, squared fade-in) lead
into A1, A2 and B2.

### 4c. Sounds and effects

- **Drums**: two fluidsynth renders of the same pattern subsets — Power kit
  (program 16: kick, snare, hats, open hat, crash) driven into `tanh` with
  2.2× drive and low-passed at 9 kHz, layered at 0.55 with the TR-808 kit
  (program 25: kick, snare, clap, extra hats) high-passed at 60 Hz. 15 % of a
  24-step bit-crushed copy blended in. The kit change is real: kick+snare
  spectral centroid 3550 Hz (Power) vs 314 Hz (808).
- **Sub bass**: synthesized per kick — sine at the beat's chord root in E1–D#2,
  pitch starts 1.9× and decays to the root in 30 ms, 4 ms attack, exponential
  decay, `tanh` saturation plus 10 % second harmonic, low-passed at 180 Hz. The
  kick on 1 gets a 1.6-beat note, the others 0.45 beats.
- **Piano**: high-pass 110 Hz (room for the sub) → tape wow/flutter (0.25 %
  at 0.55 Hz + 0.06 % at 7.3 Hz, variable-speed resampling) → crossfade
  between a dark (1.4 kHz double low-pass) and open (5.2 kHz) version driven by
  the section table above (1-bar-smoothed) → soft saturation → Schroeder
  reverb (6 combs / 3 allpasses, 18 ms predelay, 24 % wet, 4.2 kHz damping)
  → sidechain ducking on every kick (−40 %, 6 ms attack, 160 ms release).
- **Vibraphone** (program 11): doubles piano notes ≥ G4 an octave up in B1,
  bridge and B2; high-passed 300 Hz, 40 % reverb, slow stereo chorus on the
  right channel.
- **Vinyl**: band-limited hiss at −48 dB plus ~9 pops/s of random-length
  decaying impulses low-passed at 3 kHz; 1.6× in the intro, 1.8× in the outro.
- **Bus**: sum → `tanh` glue at 1.15 drive → peak 0.95 → ffmpeg
  `afade` (5 s) + `loudnorm` I=−14 TP=−1 LRA=9 → 192 kbps mp3.

### 4d. Balancing without ears

Stems are peak-normalised, so the mix gains were set from measured RMS over
bars 4–12 (first full A section) rather than by listening:

| stem | gain | RMS bars 4–12 |
|---|---|---|
| drums | 0.95 | −17.3 dB |
| sub | 0.32 | −20.3 dB (was −14.5 at the first guess of 0.62 — sustained sines read hot) |
| piano | 1.05 | −20.6 dB |
| vibes | 0.55 | ≈ −26 dB in B1 |
| vinyl | 1.2 | −36 dB |
| riser | 0.6 | — |

The tool prints this table every run. To re-balance without re-rendering:

```bash
python -m midigenai.hiphop_flip <same in> <same out> --remix --gains '{"sub":0.4,"piano":1.0}'
```

### 4e. Alignment gotcha

fluidsynth `-F` puts tick 0 at sample 166 (3.8 ms at 44.1 kHz, measured with a
two-click MIDI). The numpy-synthesized sub is placed at exact tick times, so
the rendered stems are shifted back by `FS_LATENCY` samples before mixing —
otherwise kick and sub flam.

## Attaching to todolist

The app rejects `.mid` uploads (`audio/midi` isn't allowlisted); attach the
mp3, or wrap the MIDI in an HTML page with a base64 download link.
