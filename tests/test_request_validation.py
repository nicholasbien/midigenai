"""Bad requests are 400s that name the problem, refused before Modal is called.

Each of these used to be either a 500 from deep inside generation (top_k=0,
an upload that isn't MIDI) or a silent substitution (an unknown model answered
by the default under the caller's label). Modal is stubbed out; `calls`
records whether a request got as far as inference.
"""
import io
import tempfile
from pathlib import Path

import pytest
from symusic import Note, Score, TimeSignature, Track

TPQ = 480


def _midi() -> bytes:
    s = Score(TPQ)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(name="piano", program=0)
    for i in range(16):
        t.notes.append(Note(i * TPQ, TPQ // 2, 60 + i % 12, 80))
    s.tracks.append(t)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        s.dump_midi(f.name)
        return Path(f.name).read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import midigenai.web_server as ws
    monkeypatch.setattr(ws, "GENERATED_FOLDER", str(tmp_path))
    calls = []

    class _Gen:
        class generate_batch:
            @staticmethod
            def remote(midi_bytes, **kw):
                calls.append(kw)
                return {"midis": [midi_bytes, midi_bytes]}

        class accompany_batch:
            @staticmethod
            def remote(midi_bytes, **kw):
                calls.append(kw)
                return {"midis": [midi_bytes, midi_bytes], "bars": 4,
                        "bars_available": 4, "window_seconds": 8.0,
                        "condition_track": 0, "condition_track_name": "piano",
                        "track_names": ["piano"], "generated_notes": [1, 1]}

    monkeypatch.setattr(ws, "_generator", lambda version: _Gen())
    ws.app.config["TESTING"] = True
    return ws.app.test_client(), calls


def _upload(c, route, query, data=None):
    return c.post(f"{route}?{query}", data={
        "midiFile": (io.BytesIO(_midi() if data is None else data), "song.mid")},
        content_type="multipart/form-data")


def test_defaults_when_sampling_params_are_omitted(client):
    c, calls = client
    r = _upload(c, "/api/upload_midi_v2", "model=v5-rl")
    assert r.status_code == 200, r.json
    assert calls[0]["top_k"] == 50 and calls[0]["temperature"] == 1.2


def test_valid_params_pass_through(client):
    c, calls = client
    r = _upload(c, "/api/upload_midi_v2", "model=v4&temperature=0.8&top_k=10")
    assert r.status_code == 200, r.json
    assert r.json["model"] == "v4"
    assert calls[0]["top_k"] == 10 and calls[0]["temperature"] == 0.8


@pytest.mark.parametrize("route", ["/api/upload_midi_v2", "/api/accompany"])
@pytest.mark.parametrize("query,needle", [
    ("top_k=0", "top_k must be at least 1"),
    ("top_k=-5", "top_k must be at least 1"),
    ("top_k=abc", "top_k must be a number"),
    ("temperature=0", "temperature must be in (0, 2]"),
    ("temperature=3", "temperature must be in (0, 2]"),
    ("temperature=nan", "temperature must be in (0, 2]"),
    ("temperature=hot", "temperature must be a number"),
    ("model=v9", "unknown model 'v9'"),
    ("model=v1", "unknown model 'v1'"),
])
def test_bad_params_are_refused_before_modal(client, route, query, needle):
    c, calls = client
    r = _upload(c, route, query)
    assert r.status_code == 400 and needle in r.json["error"], r.json
    assert not calls


def test_unknown_model_lists_what_is_served(client):
    c, _ = client
    r = _upload(c, "/api/upload_midi_v2", "model=v9")
    assert "v5-rl" in r.json["error"] and "v2" in r.json["error"]


def test_bad_params_on_the_preset_route(client):
    c, calls = client
    r = c.get("/api/generate_from_selected_v2/anything.mid?top_k=0")
    assert r.status_code == 400 and "top_k" in r.json["error"]
    assert not calls


@pytest.mark.parametrize("route", ["/api/upload_midi_v2", "/api/accompany",
                                   "/api/upload_midi_ab"])
@pytest.mark.parametrize("data", [b"not a midi\n", b"MThd" + b"\0" * 40])
def test_an_upload_that_is_not_midi_is_a_400(client, route, data):
    c, calls = client
    r = _upload(c, route, "model=v5-rl", data=data)
    assert r.status_code == 400, r.data
    assert "could not read the upload as MIDI" in r.json["error"]
    assert not calls


def test_empty_model_means_the_default(client):
    c, _ = client
    r = _upload(c, "/api/upload_midi_v2", "model=")
    assert r.status_code == 200 and r.json["model"] == "v5-rl"
