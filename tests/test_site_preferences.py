"""The site's preference feed: a vote has to name the sample it was cast on,
and the result has to be readable by the reward pipeline."""
import importlib
import json
from pathlib import Path

import pytest
from symusic import Note, Score, Track


def _midi_bytes(tmp_path: Path, pitch=60) -> bytes:
    s = Score(480)
    t = Track(program=0)
    for i in range(8):
        t.notes.append(Note(i * 480, 240, pitch + i, 80))
    s.tracks.append(t)
    f = tmp_path / f"p{pitch}.mid"
    s.dump_midi(str(f))
    return f.read_bytes()


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("MIDIGENAI_DATA_DIR", str(tmp_path / "data"))
    import midigenai.web_server as ws
    ws = importlib.reload(ws)
    ws.app.config["TESTING"] = True
    return ws


def _fake_result(tmp_path, with_pair_data=True) -> dict:
    full = [_midi_bytes(tmp_path, 60), _midi_bytes(tmp_path, 72)]
    out = {"midis": full, "midi": full[0], "tempo_bpm": 120.0,
           "prompt_tokens": 40, "prompt_tokens_total": 40,
           "prompt_truncated": False, "prompt_end_seconds": 4.0,
           "generated_tokens": [10, 10]}
    if with_pair_data:
        out.update({"cont_midis": [_midi_bytes(tmp_path, 64),
                                   _midi_bytes(tmp_path, 76)],
                    "prompt_ids": [1, 2, 3], "cont_ids": [[4, 5], [6, 7]]})
    return out


def _upload(client, midi: bytes, route="/api/upload_midi_ab"):
    return client.post(route, data={"midiFile": (__import__("io").BytesIO(midi),
                                                 "riff.mid")},
                       content_type="multipart/form-data")


def test_vote_resolves_through_the_recorded_display_order(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_generate_n",
                        lambda *a, **k: _fake_result(tmp_path))
    # force the shuffle so option 1 is sample b
    monkeypatch.setattr(server.random, "random", lambda: 0.0)
    client = server.app.test_client()
    rid = _upload(client, _midi_bytes(tmp_path)).get_json()["requestId"]

    meta = json.loads((Path(server.PAIRS_FOLDER) / f"{rid}.json").read_text())
    assert meta["shown"] == ["b", "a"]

    r = client.post("/api/submit_preference_ab",
                    data={"requestId": rid, "preferredMidi": "option1"})
    assert r.get_json()["paired"] is True
    row = json.loads(Path(server.LABELS_PATH).read_text().splitlines()[-1])
    assert row["preferred"] == "b"          # option 1 held sample b
    assert row["left_is"] == "b" and row["right_is"] == "a"
    assert row["pair_id"] == rid


def test_pairs_are_readable_by_reward_align(server, tmp_path, monkeypatch):
    from midigenai.reward_align import load_label_app_pairs

    monkeypatch.setattr(server, "_generate_n",
                        lambda *a, **k: _fake_result(tmp_path))
    monkeypatch.setattr(server.random, "random", lambda: 1.0)   # no shuffle
    client = server.app.test_client()
    rid = _upload(client, _midi_bytes(tmp_path)).get_json()["requestId"]
    client.post("/api/submit_preference_ab",
                data={"requestId": rid, "preferredMidi": "option2"})

    pairs, votes = load_label_app_pairs(Path(server.LABELS_PATH))
    assert len(pairs) == 1
    winner, loser, group, meta_path, win = pairs[0]
    assert win == "b" and winner.name == f"{rid}_b.mid" and loser.exists()
    assert votes[rid] == ["b"]
    # drift features need the continuation ids to be in the meta
    assert json.loads(meta_path.read_text())["cont_b_ids"] == [6, 7]


def test_plain_generate_route_also_writes_a_pair(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_generate_n",
                        lambda *a, **k: _fake_result(tmp_path))
    client = server.app.test_client()
    body = _upload(client, _midi_bytes(tmp_path), route="/api/upload_midi").get_json()
    rid = body["requestId"]
    assert (Path(server.PAIRS_FOLDER) / f"{rid}_prompt.mid").exists()

    r = client.post("/api/submit_preference",
                    data={"requestId": rid, "preferredMidi": "option2",
                          "selectedFileName": "riff.mid"})
    assert r.get_json()["paired"] is True
    row = json.loads(Path(server.LABELS_PATH).read_text().splitlines()[-1])
    assert row["preferred"] == "b"
    # the legacy CSV keeps its shape for the existing frontend
    assert Path(server.DATA_DIR, "responses.csv").read_text().strip().endswith(",1")


def test_vote_without_a_pair_is_logged_not_lost(server, tmp_path, monkeypatch):
    client = server.app.test_client()
    r = client.post("/api/submit_preference",
                    data={"preferredMidi": "option1", "selectedFileName": "x.mid"})
    assert r.status_code == 200 and r.get_json()["paired"] is False
    event = json.loads(Path(server.EVENTS_PATH).read_text().splitlines()[-1])
    assert event["kind"] == "vote_orphan"


def test_old_serving_deployment_logs_no_half_pair(server, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_generate_n",
                        lambda *a, **k: _fake_result(tmp_path, with_pair_data=False))
    client = server.app.test_client()
    rid = _upload(client, _midi_bytes(tmp_path)).get_json()["requestId"]
    assert not (Path(server.PAIRS_FOLDER) / f"{rid}_a.mid").exists()
    event = json.loads(Path(server.EVENTS_PATH).read_text().splitlines()[-1])
    assert event["kind"] == "pair_skipped"


def test_storage_id_survives_a_restart_and_shows_up_in_health(server, tmp_path):
    first = server.app.test_client().get("/").get_json()["storage"]
    reloaded = importlib.reload(server)
    second = reloaded.app.test_client().get("/").get_json()["storage"]
    assert first["id"] == second["id"] and first["id"] is not None
    assert second["dir"] == str(tmp_path / "data")
