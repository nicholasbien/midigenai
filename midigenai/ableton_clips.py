"""Extract MIDI clips from Ableton Live project files as prompt seeds.

Live stores a set's MIDI clips inside the .als (gzipped XML) — not as .mid
files — so 1,600 projects on disk yield one exported MIDI file but tens of
thousands of clips. As prompts these are the best source there is: the
downstream use case IS jamming in Live, and nothing here has ever been in a
training corpus.

Dedup is by note content. Live's Backup/ folders are auto-saves of the same
set, and a clip copied across scenes is the same clip; hashing the sorted
note tuple collapses all of that. Filters are musical, not cosmetic: at
least `min_notes` enabled notes, spanning at least `min_bars` bars.

    python -m midigenai.ableton_clips --root ~/Music/Ableton --out evals/prompts_ableton
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TPQ = 480


def _tempo(root) -> float:
    t = root.find(".//Tempo/Manual")
    try:
        return float(t.get("Value")) if t is not None else 120.0
    except (TypeError, ValueError):
        return 120.0


def clips_of(path: Path):
    """Yield (notes, tempo) per MIDI clip; notes are (start_beat, dur_beat, pitch, vel)."""
    raw = path.read_bytes()
    xml = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    root = ET.fromstring(xml)
    tempo = _tempo(root)
    for clip in root.iter("MidiClip"):
        notes = []
        for kt in clip.iter("KeyTrack"):
            key = kt.find("MidiKey")
            if key is None:
                continue
            pitch = int(key.get("Value"))
            for ev in kt.iter("MidiNoteEvent"):
                if ev.get("IsEnabled", "true") != "true":
                    continue
                t, d, v = float(ev.get("Time")), float(ev.get("Duration")), float(ev.get("Velocity", 100))
                if d <= 0:
                    continue
                notes.append((t, d, pitch, int(max(1, min(127, v)))))
        if notes:
            yield notes, tempo


def write_clip(notes, tempo: float, out: Path) -> None:
    from symusic import Note, Score, Tempo, TimeSignature, Track
    sc = Score(TPQ)
    sc.tempos.append(Tempo(0, tempo))
    sc.time_signatures.append(TimeSignature(0, 4, 4))
    t0 = min(n[0] for n in notes)                      # clips can start mid-arrangement
    tr = Track(program=0, is_drum=False)
    for s, d, p, v in sorted(notes):
        tr.notes.append(Note(int(round((s - t0) * TPQ)), max(1, int(round(d * TPQ))), p, v))
    sc.tracks.append(tr)
    sc.dump_midi(str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/Music/Ableton"))
    ap.add_argument("--out", type=Path, default=Path("evals/prompts_ableton"))
    ap.add_argument("--min-notes", type=int, default=16)
    ap.add_argument("--min-bars", type=float, default=2.0)
    ap.add_argument("--min-pitches", type=int, default=5,
                    help="fewer distinct pitches than this goes to <out>_drums/")
    ap.add_argument("--limit", type=int, default=0, help="projects to scan (0 = all)")
    a = ap.parse_args()

    projects = sorted(glob.glob(os.path.join(a.root, "**", "*.als"), recursive=True))
    if a.limit:
        projects = projects[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_clips = n_kept = n_bad = n_drum = 0
    t0 = time.time()
    for i, p in enumerate(projects, 1):
        try:
            for notes, tempo in clips_of(Path(p)):
                n_clips += 1
                if len(notes) < a.min_notes:
                    continue
                span = max(s + d for s, d, *_ in notes) - min(s for s, *_ in notes)
                if span < a.min_bars * 4:
                    continue
                key = hashlib.sha1(repr(sorted((round(s, 3), round(d, 3), pch, v)
                                               for s, d, pch, v in notes)).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                # Drum-rack clips are pitched notes on a handful of pads; they
                # are real jam prompts but the wrong distribution for a
                # melodic judge, so they get their own directory instead of
                # silently being half the set (cf. 48% drum conditions).
                pitches = {pch for _, _, pch, _ in notes}
                if len(pitches) < a.min_pitches:
                    drums_dir = a.out.parent / (a.out.name + "_drums")
                    drums_dir.mkdir(parents=True, exist_ok=True)
                    write_clip(notes, tempo, drums_dir / f"val_abletondrum_{key[:16]}.mid")
                    n_drum += 1
                    continue
                write_clip(notes, tempo, a.out / f"val_ableton_{key[:16]}.mid")
                n_kept += 1
        except Exception:
            n_bad += 1
        if i % 200 == 0:
            print(f"[ableton] {i}/{len(projects)} projects  clips {n_clips}  unique kept {n_kept}  "
                  f"drum-ish {n_drum}  unreadable {n_bad}  {time.time()-t0:.0f}s", flush=True)
    print(f"[ableton] done: {len(projects)} projects, {n_clips} clips seen, "
          f"{n_kept} unique clips kept (>= {a.min_notes} notes, >= {a.min_bars} bars, "
          f">= {a.min_pitches} pitches), {n_drum} drum-ish to {a.out.name}_drums/, "
          f"{n_bad} unreadable -> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
