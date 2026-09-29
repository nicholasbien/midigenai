"""Track choice for accompaniment: one code path for the picker and the server."""
import pytest
from symusic import Note, Score, Track

from midigenai.accompany_tracks import TrackChoiceError, choose_track, summarize


def _upload(tmp_path, counts, empty=()):
    s = Score(480)
    for i, n in enumerate(counts):
        tr = Track(name=f"t{i}", program=i * 8)
        if i not in empty:
            for k in range(n):
                tr.notes.append(Note(k * 240, 200, 60 + i, 80))
        s.tracks.append(tr)
    p = tmp_path / "u.mid"; s.dump_midi(str(p)); return p.read_bytes()


def test_summary_lists_tracks_and_default_is_densest(tmp_path):
    out = summarize(_upload(tmp_path, [16, 32, 8]), bars=8)
    assert out["trackNoteCounts"][1] == max(out["trackNoteCounts"])
    assert out["defaultTrack"] == 1
    assert len(out["trackNames"]) == len(out["trackNoteCounts"]) == len(out["trackPrograms"])


def test_explicit_track_is_used_and_bad_index_is_loud(tmp_path):
    from midigenai.accompany_tracks import prepare_window
    win, _, _ = prepare_window(_upload(tmp_path, [16, 32, 8]), 8)
    assert choose_track(win, 2) == 2
    assert choose_track(win, None) == 1
    with pytest.raises(TrackChoiceError, match="out of range"):
        choose_track(win, len(win.tracks))
    with pytest.raises(TrackChoiceError, match="out of range"):
        choose_track(win, -1)
