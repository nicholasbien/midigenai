"""Extract MIDI clips from Ableton Live project files as prompt seeds.

Live stores a set's MIDI clips inside the .als (gzipped XML) — not as .mid
files — so 1,600 projects on disk yield one exported MIDI file but tens of
thousands of clips. As prompts these are the best source there is: the
downstream use case IS jamming in Live, and nothing here has ever been in a
training corpus.

Three things decide whether a clip is a valid seed:

  * it was not written by the model. Jam sessions record the model's
    answers on Live tracks named "model" / "model drums" / "model chords",
    and create projects and tracks whose names carry the tool ("claude_dnb",
    "25m_pilot_barker"). A model-written clip used as a seed would be
    on-policy output fed back as a prompt — circular — so any clip whose
    project path, track name or clip name matches EXCLUDE is dropped. IAC
    routing would be a name-independent signal but Live does not persist it
    in the .als, so names are the whole rule.
  * whether it is a kit. Live's drum rack is a `DrumGroupDevice` in the
    track's device chain, whatever the track is called; that test comes from
    data/ableton.py. Distinct-pitch count is only the fallback when a track
    carries no device info — on its own it misfiles bass lines and pedal
    pads as drums and misses a busy six-pad kit. Kits go to a sibling
    _drums/ directory rather than silently being half the set.
  * dedup by note content. Backup/ folders are auto-saves of the same set,
    and a clip copied across scenes is the same clip; hashing the sorted
    note tuple collapses all of that (12 projects: 1,613 clips -> 49).

Filters are musical, not cosmetic: at least `min_notes` enabled notes,
spanning at least `min_bars` bars.

    python -m midigenai.ableton_clips --root ~/Music/Ableton --out evals/prompts_ableton
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TPQ = 480

# what the other session applied to its 341 clips (58 excluded), plus the
# jam harness's own answer lanes
EXCLUDE = re.compile(r"claude|pilot|_gen|mcp|midigen|openmuse|\bmodel\b", re.I)


def _tempo(root) -> float:
    t = root.find(".//Tempo/Manual")
    try:
        return float(t.get("Value")) if t is not None else 120.0
    except (TypeError, ValueError):
        return 120.0


def _text(node, path: str, default: str = "") -> str:
    el = node.find(path)
    v = el.get("Value") if el is not None else None
    return v if v is not None else default


def _is_drum_track(tr, notes_pitches: set[int], min_pitches: int) -> bool:
    """Device chain first (data/ableton.py's test); pitch count only as a
    fallback when the track carries no device information at all."""
    devices = [d.tag for d in tr.findall(".//DeviceChain//Devices/*")]
    if devices:
        return any("DrumGroupDevice" in d for d in devices)
    return len(notes_pitches) < min_pitches


def clips_of(path: Path, min_pitches: int):
    """Yield (notes, tempo, is_drum, excluded_reason) per MIDI clip, walking
    tracks so each clip knows its track's name and device chain."""
    raw = path.read_bytes()
    xml = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    root = ET.fromstring(xml)
    tempo = _tempo(root)
    proj_hit = bool(EXCLUDE.search(str(path)))
    for tr in root.findall(".//MidiTrack"):
        tname = _text(tr, ".//Name/EffectiveName") or _text(tr, ".//Name/UserName")
        track_hit = bool(EXCLUDE.search(tname))
        for clip in tr.iter("MidiClip"):
            cname = _text(clip, "Name")
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
            if not notes:
                continue
            reason = ("project" if proj_hit else "track" if track_hit
                      else "clip" if EXCLUDE.search(cname) else None)
            is_drum = _is_drum_track(tr, {n[2] for n in notes}, min_pitches)
            yield notes, tempo, is_drum, reason


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
                    help="drum fallback when a track has no device chain: fewer "
                         "distinct pitches than this counts as a kit")
    ap.add_argument("--limit", type=int, default=0, help="projects to scan (0 = all)")
    a = ap.parse_args()

    projects = sorted(glob.glob(os.path.join(a.root, "**", "*.als"), recursive=True))
    if a.limit:
        projects = projects[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_clips = n_kept = n_bad = n_drum = n_excl = 0
    excl_by: dict[str, int] = {}
    t0 = time.time()
    for i, p in enumerate(projects, 1):
        try:
            for notes, tempo, is_drum, reason in clips_of(Path(p), a.min_pitches):
                n_clips += 1
                if reason:
                    n_excl += 1; excl_by[reason] = excl_by.get(reason, 0) + 1
                    continue
                if len(notes) < a.min_notes:
                    continue
                span = max(s_ + d for s_, d, *_ in notes) - min(s_ for s_, *_ in notes)
                if span < a.min_bars * 4:
                    continue
                key = hashlib.sha1(repr(sorted((round(s_, 3), round(d, 3), pch, v)
                                               for s_, d, pch, v in notes)).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                if is_drum:
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
            print(f"[ableton] {i}/{len(projects)} projects  clips {n_clips}  kept {n_kept}  "
                  f"drums {n_drum}  excluded {n_excl}  unreadable {n_bad}  {time.time()-t0:.0f}s", flush=True)
    print(f"[ableton] done: {len(projects)} projects, {n_clips} clips seen, "
          f"{n_excl} model-written excluded {excl_by}, "
          f"{n_kept} unique pitched clips kept (>= {a.min_notes} notes, >= {a.min_bars} bars), "
          f"{n_drum} kits to {a.out.name}_drums/, {n_bad} unreadable -> {a.out}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
