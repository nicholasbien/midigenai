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


def summarize(midi_bytes: bytes, bars: int = 8) -> dict:
    """What a track picker needs, with no generation."""
    window, bars, available = prepare_window(midi_bytes, bars)
    return {
        "trackNames": [t.name or "" for t in window.tracks],
        "trackNoteCounts": [len(t.notes) for t in window.tracks],
        "trackPrograms": [(-1 if t.is_drum else t.program) for t in window.tracks],
        "defaultTrack": densest(window),
        "bars": bars,
        "barsAvailable": available,
    }
