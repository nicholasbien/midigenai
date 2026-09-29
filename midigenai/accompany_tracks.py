"""Which track of an upload accompaniment plays along with.

Accompaniment answers one part, so a multitrack upload is narrowed to a single
condition track. The website lets the user pick that track, which means the
index it shows and the index the server selects must come from the same code:
this module is called by both the Flask `/api/tracks` route (no GPU, no torch)
and `MidiGen.accompany_batch` on Modal.

Indices are positions in the *windowed* score, the same list `track_names` is
built from in every response. A bad index is an error naming the valid range,
never a silent fallback to the densest track.
"""

from __future__ import annotations

from io import BytesIO

from symusic import Score

from midigenai.data.v4_docs import _window, bar_edges, trim_leading
from midigenai.tokenizer import normalize_drums


class TrackChoiceError(ValueError):
    """The requested track cannot be used; the message says why and what can."""


def prepare_window(midi_bytes: bytes, bars: int):
    """(window, bars_used, bars_available) for the first `bars` bars."""
    score = Score.from_midi(BytesIO(midi_bytes).read())
    normalize_drums(score, "upload.mid")
    score = trim_leading(score)
    edges = bar_edges(score)
    available = max(0, len(edges) - 1)
    if available < 1:
        raise TrackChoiceError("upload has no complete bar to accompany")
    bars = max(1, min(bars, available))
    return _window(score, edges[0], edges[bars]), bars, available


def densest(window) -> int:
    if not window.tracks:
        raise TrackChoiceError("upload has no tracks")
    return max(range(len(window.tracks)), key=lambda i: len(window.tracks[i].notes))


def choose_track(window, track: int | None) -> int:
    """The condition track index: `track` if valid, else the densest."""
    n = len(window.tracks)
    if track is None:
        return densest(window)
    if not 0 <= track < n:
        raise TrackChoiceError(f"track={track} is out of range; this upload has tracks 0-{n - 1}")
    if not len(window.tracks[track].notes):
        playable = [i for i, t in enumerate(window.tracks) if len(t.notes)]
        raise TrackChoiceError(
            f"track {track} has no notes in the first bars; tracks with notes: {playable}")
    return track


def parse_track_spec(raw) -> list[int] | str | None:
    """`track=` value -> list of indices, "all", or None (not given / auto).

    Accepts an int, a list of ints, or a string: "2", "0,2", "all", "auto".
    Raises TrackChoiceError on anything else, naming the value."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    s = str(raw).strip().lower()
    if s in ("", "auto"):
        return None
    if s == "all":
        return "all"
    try:
        out = [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise TrackChoiceError(f"track must be an index, a comma list like 0,2, or 'all'; got {raw!r}")
    if not out:
        raise TrackChoiceError(f"track={raw!r} names no tracks")
    return out


def choose_tracks(window, spec) -> list[int]:
    """Condition track indices for a parsed `track=` spec.

    None -> [densest]; "all" -> every track with notes; a list -> exactly
    those, each validated like choose_track, duplicates removed, order kept.
    """
    if spec is None:
        return [densest(window)]
    if spec == "all":
        live = [i for i, t in enumerate(window.tracks) if len(t.notes)]
        if not live:
            raise TrackChoiceError("upload has no notes in the first bars")
        return live
    seen = []
    for i in spec:
        choose_track(window, i)
        if i not in seen:
            seen.append(i)
    return seen


def continuation_source(midi_bytes: bytes, raw) -> tuple[bytes, list[int], list[str]]:
    """(bytes to continue, prompt track indices, their names) for `track=`.

    Continuation was trained on two shapes: the whole mix, and a single track
    on its own (v4_docs' solo views). So omitted / "auto" / "all" -> the file
    unchanged, one index -> that track alone with the file's tempo and meter.
    A subset of several tracks was never a continuation document, so it is a
    400 rather than something quietly attempted. Indices are file track
    positions, the same ones /api/tracks lists.
    """
    score = Score.from_midi(BytesIO(midi_bytes).read())
    n = len(score.tracks)
    live = [i for i, t in enumerate(score.tracks) if len(t.notes)]
    spec = parse_track_spec(raw)
    if spec is None or spec == "all" or (isinstance(spec, list) and sorted(set(spec)) == live):
        return midi_bytes, live, [score.tracks[i].name or "" for i in live]
    if len(set(spec)) > 1:
        raise TrackChoiceError(
            "continuation takes the whole mix (omit track, or track=all) or one track; "
            f"a subset like {raw!r} was not a trained continuation shape")
    i = spec[0]
    if not 0 <= i < n:
        raise TrackChoiceError(f"track={i} is out of range; this upload has tracks 0-{n - 1}")
    if not len(score.tracks[i].notes):
        raise TrackChoiceError(f"track {i} has no notes; tracks with notes: {live}")
    solo = Score(score.ticks_per_quarter)
    for ts in score.time_signatures: solo.time_signatures.append(ts)
    for te in score.tempos: solo.tempos.append(te)
    for ks in score.key_signatures: solo.key_signatures.append(ks)
    solo.tracks.append(score.tracks[i])
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".mid"); os.close(fd)
    try:
        solo.dump_midi(path)
        with open(path, "rb") as f:
            return f.read(), [i], [score.tracks[i].name or ""]
    finally:
        os.unlink(path)


def summarize(midi_bytes: bytes, bars: int = 8) -> dict:
    """What a track picker needs, with no generation.

    trackNoteCounts are inside the accompaniment window (`bars`);
    fileNoteCounts are over the whole file, which is what continuation uses."""
    window, bars, available = prepare_window(midi_bytes, bars)
    whole = Score.from_midi(BytesIO(midi_bytes).read())
    return {
        "fileNoteCounts": [len(t.notes) for t in whole.tracks],
        "trackNames": [t.name or "" for t in window.tracks],
        "trackNoteCounts": [len(t.notes) for t in window.tracks],
        "trackPrograms": [(-1 if t.is_drum else t.program) for t in window.tracks],
        "defaultTrack": densest(window),
        "bars": bars,
        "barsAvailable": available,
    }
