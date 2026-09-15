"""The control surface as the site exposes it: what reaches the model, what
comes back, and what a typo gets you."""
import importlib
import io
import types

import pytest
from symusic import Note, Score, Track


def _midi(tmp_path) -> bytes:
    s = Score(480)
    t = Track(program=0)
    for i in range(16):
        t.notes.append(Note(i * 480, 240, 60 + i % 12, 80))
    s.tracks.append(t)
    f = tmp_path / "riff.mid"
    s.dump_midi(str(f))
    return f.read_bytes()


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("MIDIGENAI_DATA_DIR", str(tmp_path / "data"))
    import midigenai.web_server as ws
    return importlib.reload(ws)


@pytest.fixture
def seen(server, monkeypatch):
    """Capture what the web layer would send to the serving app."""
    calls = []

    def fake_generate(midi_bytes, temperature, top_k, max_new_tokens, n_samples,
                      version, controls=None):
        calls.append({"controls": controls, "version": version})
        return {"midis": [b"MThd1", b"MThd2"], "midi": b"MThd1",
                "header": ["Inst_Piano", "Density_2"], "tempo_bpm": 120.0,
                "prompt_tokens": 10, "prompt_tokens_total": 10,
                "prompt_truncated": False, "prompt_end_seconds": 1.0,
                "generated_tokens": [4, 4]}

    monkeypatch.setattr(server, "_generate_n", fake_generate)
    return calls


def _post(client, url, midi):
    return client.post(url, data={"midiFile": (io.BytesIO(midi), "riff.mid")},
                       content_type="multipart/form-data")


def test_health_publishes_the_control_vocabulary(server):
    body = server.app.test_client().get("/").get_json()
    assert body["controls"]["density"] == [0, 1, 2, 3]
    assert "Drums" in body["controls"]["instruments"]
    assert "density" in body["controls"]["buckets"]


def test_controls_reach_the_model_and_the_header_comes_back(server, seen, tmp_path):
    client = server.app.test_client()
    body = _post(client,
                 "/api/upload_midi?density=3&poly=1&range=2&instruments=Piano,Drums"
                 "&genre=jazz", _midi(tmp_path)).get_json()
    assert seen[0]["controls"] == {"density": 3, "poly": 1, "pitch_range": 2,
                                   "instruments": ["Piano", "Drums"],
                                   "genres": ["jazz"]}
    assert body["header"] == ["Inst_Piano", "Density_2"]


def test_an_uncontrolled_request_sends_no_controls_at_all(server, seen, tmp_path):
    """An older serving deployment rejects the argument; a plain request
    should not care which service deployed first."""
    _post(server.app.test_client(), "/api/upload_midi", _midi(tmp_path))
    assert seen[0]["controls"] == {}


def test_a_typo_is_a_400_that_says_what_was_allowed(server, seen, tmp_path):
    client = server.app.test_client()
    r = _post(client, "/api/upload_midi?instruments=Kazoo", _midi(tmp_path))
    assert r.status_code == 400
    body = r.get_json()
    assert "Kazoo" in body["error"] and "Drums" in body["controls"]["instruments"]
    r = _post(client, "/api/upload_midi?density=9", _midi(tmp_path))
    assert r.status_code == 400 and "0..3" in r.get_json()["error"]
    assert seen == []                       # nothing reached the GPU


def test_controls_on_a_model_that_has_no_header_are_a_400(server, monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise ValueError("attribute controls need a v4 checkpoint; "
                         "'v3' has no attribute header")
    monkeypatch.setattr(server, "_generate_n", refuse)
    r = _post(server.app.test_client(), "/api/upload_midi?model=v3&density=1",
              _midi(tmp_path))
    assert r.status_code == 400 and "v4 checkpoint" in r.get_json()["error"]


def test_infill_route_returns_the_span_it_rewrote(server, monkeypatch, tmp_path):
    calls = []

    def fake_infill(midi_bytes, at_bar, bars, temperature, top_k, n_samples,
                    controls=None):
        calls.append({"at_bar": at_bar, "bars": bars, "controls": controls})
        return {"midis": [b"MThd1", b"MThd2"], "midi": b"MThd1",
                "at_bar": at_bar, "bars": bars, "bars_available": 8,
                "span_start_seconds": 4.0, "span_seconds": 4.0,
                "generated_notes": [12, 9], "header": ["Density_1"],
                "tempo_bpm": 120.0, "span_bars_generated": [2, 2]}

    stub = types.SimpleNamespace(
        infill_batch=types.SimpleNamespace(remote=fake_infill))
    monkeypatch.setattr(server, "_generator", lambda version: stub)

    r = _post(server.app.test_client(), "/api/infill?at_bar=2&bars=2&density=1",
              _midi(tmp_path))
    body = r.get_json()
    assert r.status_code == 200
    assert calls[0] == {"at_bar": 2, "bars": 2, "controls": {"density": 1}}
    assert body["atBar"] == 2 and body["bars"] == 2 and body["barsAvailable"] == 8
    assert body["spanStartSeconds"] == 4.0
    assert body["midiUrl1"] != body["midiUrl2"]


def test_infill_on_a_v3_model_is_a_400(server, monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise ValueError("infilling needs a v4 checkpoint; 'v3' is not one")
    stub = types.SimpleNamespace(
        infill_batch=types.SimpleNamespace(remote=refuse))
    monkeypatch.setattr(server, "_generator", lambda version: stub)
    r = _post(server.app.test_client(), "/api/infill?model=v3", _midi(tmp_path))
    assert r.status_code == 400 and "v4" in r.get_json()["error"]
