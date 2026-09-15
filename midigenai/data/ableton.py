"""
Extract MIDI from Ableton Live project files.

Your own productions are the one source where exact MIDI already exists —
transcribing your own audio throws away information you still have. A `.als`
file is gzipped XML: every MIDI clip carries its notes as `MidiNoteEvent`
elements grouped under a `KeyTrack` per pitch, positioned in beats on the
clip's own internal timeline, and the clip sits at `CurrentStart` beats on
the arrangement.

The part that is easy to get wrong: a clip's note buffer is usually longer
than the clip plays. Live sounds only the region between `LoopStart` and
`LoopEnd`, mapped onto `CurrentStart`..`CurrentEnd` and repeated when
`LoopOn` is set. 67% of clips in one real project carried notes outside that
region; emitting them all produced a wall of overlapping takes.

    python -m midigenai.data.ableton --root ~/Music --out ~/midigenai_data/raw/ableton

Drum tracks are marked from the device chain, not the name: a track holding a
`DrumGroupDevice` is a kit, whatever the user called it. Live also hosts
one-shot samplers (a single `OriginalSimpler` triggered at C3 for a clap or a
909 hat), which are percussion played from one key — those are marked too.
"""

from __future__ import annotations

import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path

TPQ = 480
DEFAULT_TEMPO = 120.0


def _val(node, path: str, default=None):
    e = node.find(path)
    return e.get("Value") if e is not None else default


def project_tempo(root) -> float:
    for path in (".//MasterTrack//Tempo/Manual", ".//Tempo/Manual",
                 ".//MasterTrack//Tempo/FloatEvent"):
        v = _val(root, path)
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    return DEFAULT_TEMPO


PERC_NAME_HINTS = ("kick", "snare", "clap", "hat", "hihat", "tom", "perc",
                   "crash", "ride", "rim", "shaker", "cymbal", "808", "909",
                   "kit", "drum")


def _is_drum_track(tr, name: str) -> bool:
    """A kit, by device chain first and name only as a fallback.

    Two shapes appear in real projects: a `DrumGroupDevice` (Live's drum
    rack), and a one-shot sampler triggered from a single key - a 909 clap
    on C3 is percussion even though its note data looks like one repeated
    pitch. Both would otherwise decode as a pitched instrument.
    """
    devices = [d.tag for d in tr.findall(".//DeviceChain//Devices/*")]
    if any("DrumGroupDevice" in d for d in devices):
        return True
    lname = (name or "").lower()
    if any(k in lname for k in PERC_NAME_HINTS):
        pitches = {k.get("Value") for k in
                   tr.findall(".//ArrangerAutomation//KeyTrack/MidiKey")}
        # a one-shot sampler plays a single key; a melodic part does not
        if len(pitches) <= 2:
            return True
    return False


def extract(als_path: Path, with_audio_count: bool = False):
    """Return a symusic Score, or None if the project has no MIDI notes.

    `with_audio_count` also returns how many audio clips the arrangement has:
    a project that is mostly recorded or sampled audio yields only the parts
    that happened to be MIDI, which is worth knowing before training on it.
    """
    from symusic import Score, Tempo, Track
    from symusic.core import NoteTick

    with gzip.open(als_path) as f:
        root = ET.parse(f).getroot()
    score = Score(TPQ)
    score.tempos.append(Tempo(time=0, qpm=project_tempo(root)))
    total = 0
    for tr in root.findall(".//MidiTrack"):
        if (_val(tr, ".//Mixer/Speaker/Manual", "true") or "true").lower() == "false":
            continue                                    # track muted in the mix
        name = _val(tr, ".//Name/EffectiveName", "") or ""
        track = Track(name=name[:60], is_drum=_is_drum_track(tr, name))
        for clip in tr.findall(".//ArrangerAutomation//MidiClip"):
            if (_val(clip, "Disabled", "false") or "false").lower() == "true":
                continue
            try:
                clip_start = float(_val(clip, "CurrentStart", "0") or 0)
                clip_end = float(_val(clip, "CurrentEnd", "0") or 0)
                lo = float(_val(clip, "Loop/LoopStart", "0") or 0)
                hi = float(_val(clip, "Loop/LoopEnd", "0") or 0)
            except ValueError:
                continue
            region = hi - lo
            span = clip_end - clip_start
            if region <= 0 or span <= 0:
                continue
            loop_on = (_val(clip, "Loop/LoopOn", "false") or "false").lower() == "true"
            reps = max(1, int(span / region) + 1) if loop_on else 1
            for key in clip.findall(".//KeyTrack"):
                pitch = _val(key, "MidiKey")
                if pitch is None:
                    continue
                pitch = max(0, min(127, int(pitch)))
                for ev in key.findall(".//MidiNoteEvent"):
                    if (ev.get("IsEnabled") or "true").lower() == "false":
                        continue
                    try:
                        t = float(ev.get("Time", "0"))
                        d = float(ev.get("Duration", "0"))
                        v = int(float(ev.get("Velocity", "80")))
                    except ValueError:
                        continue
                    if d <= 0 or not (lo <= t < hi):
                        continue                        # outside what Live plays
                    for k in range(reps):
                        at = clip_start + (t - lo) + k * region
                        if at >= clip_end:
                            break
                        dur = min(d, clip_end - at)     # a clip cuts its own tail
                        if dur <= 0:
                            continue
                        track.notes.append(NoteTick(
                            time=int(round(at * TPQ)),
                            duration=max(1, int(round(dur * TPQ))),
                            pitch=pitch,
                            velocity=max(1, min(127, v))))
                        total += 1
        if len(track.notes):
            score.tracks.append(track)
    out = score if total else None
    if with_audio_count:
        n_audio = len(root.findall(".//AudioTrack//ArrangerAutomation//AudioClip"))
        return out, n_audio
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, nargs="+", required=True,
                   help="directories to search for .als files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-notes", type=int, default=64)
    p.add_argument("--skip-backups", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for r in args.root:
        files += [f for f in r.rglob("*.als")
                  if not (args.skip_backups and "Backup" in f.parts)]
    print(f"[ableton] {len(files)} project files")
    kept = skipped = failed = 0
    notes_total = 0
    manifest = []
    for f in sorted(files):
        try:
            sc, n_audio = extract(f, with_audio_count=True)
        except Exception:
            failed += 1
            continue
        if sc is None:
            skipped += 1
            continue
        n = sum(len(t.notes) for t in sc.tracks)
        if n < args.min_notes:
            skipped += 1
            continue
        stem = f"{f.parent.name}_{f.stem}".replace(" ", "_").replace("/", "_")[:80]
        dst = args.out / f"{stem}.mid"
        try:
            sc.dump_midi(dst)
        except Exception:
            failed += 1
            continue
        kept += 1
        notes_total += n
        manifest.append({"path": str(dst), "source_als": str(f),
                         "n_notes": n, "n_tracks": len(sc.tracks),
                         "n_drum_tracks": sum(1 for t in sc.tracks if t.is_drum),
                         "n_audio_clips": n_audio})
    (args.out / "ableton_manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"[ableton] wrote {kept} files ({notes_total:,} notes), "
          f"skipped {skipped} (no/few notes), {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()
