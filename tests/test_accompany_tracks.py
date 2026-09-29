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


def test_track_spec_parses_lists_all_and_auto():
    from midigenai.accompany_tracks import parse_track_spec
    assert parse_track_spec(None) is None and parse_track_spec("auto") is None and parse_track_spec("") is None
    assert parse_track_spec("2") == [2] and parse_track_spec(2) == [2]
    assert parse_track_spec("0, 2") == [0, 2] and parse_track_spec([1, 0]) == [1, 0]
    assert parse_track_spec("ALL") == "all"
    with pytest.raises(TrackChoiceError):
        parse_track_spec("bass")
    with pytest.raises(TrackChoiceError):
        parse_track_spec(",")


def test_choose_tracks_validates_every_index(tmp_path):
    from midigenai.accompany_tracks import choose_tracks, prepare_window
    win, _, _ = prepare_window(_upload(tmp_path, [16, 32, 8]), 8)
    assert choose_tracks(win, None) == [1]
    assert choose_tracks(win, [2, 0, 2]) == [2, 0]
    assert choose_tracks(win, "all") == [0, 1, 2]
    with pytest.raises(TrackChoiceError, match="out of range"):
        choose_tracks(win, [0, 9])


def test_continuation_source_whole_mix_or_one_track(tmp_path):
    from symusic import Score
    from midigenai.accompany_tracks import continuation_source
    data = _upload(tmp_path, [16, 32, 8])
    b, tr, names = continuation_source(data, None)
    assert b == data and tr == [0, 1, 2]
    b, tr, names = continuation_source(data, "all")
    assert b == data and tr == [0, 1, 2]
    b, tr, names = continuation_source(data, "2")
    s = Score.from_midi(b)
    assert tr == [2] and len(s.tracks) == 1 and len(s.tracks[0].notes) == 8
    with pytest.raises(TrackChoiceError, match="not a trained"):
        continuation_source(data, "0,2")
    with pytest.raises(TrackChoiceError, match="out of range"):
        continuation_source(data, "7")
