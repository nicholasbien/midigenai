"""Corpus quality labeling: excerpting, features, log round-trip, predictor."""
import json
import math
import random
import types
from pathlib import Path

import numpy as np
import pytest
from symusic import Note, Score, Tempo, TimeSignature, Track

from midigenai import quality_predictor as qp
from midigenai import rate_corpus_app as rc


def synth_score(bars=64, bpm=120.0, tpq=480, seed=0, with_drums=True) -> Score:
    """4/4 at `bpm`; a walking bass line + chords + hats, one bar = 4 beats."""
    rng = random.Random(seed)
    s = Score(tpq)
    s.tempos.append(Tempo(time=0, qpm=bpm))
    s.time_signatures.append(TimeSignature(time=0, numerator=4, denominator=4))
    piano = Track(program=0, is_drum=False)
    bass = Track(program=33, is_drum=False)
    for bar in range(bars):
        t0 = bar * 4 * tpq
        root = 48 + [0, 5, 7, 2][bar % 4]
        for beat in range(4):
            t = t0 + beat * tpq
            bass.notes.append(Note(time=t, duration=tpq, pitch=root - 12 + rng.choice([0, 7]),
                                   velocity=80))
            for iv in (0, 4, 7):
                piano.notes.append(Note(time=t, duration=tpq // 2, pitch=root + 12 + iv,
                                        velocity=70 + rng.randrange(20)))
    s.tracks.append(piano)
    s.tracks.append(bass)
    if with_drums:
        drums = Track(program=0, is_drum=True)
        for bar in range(bars):
            for eighth in range(8):
                drums.notes.append(Note(time=bar * 4 * tpq + eighth * tpq // 2,
                                        duration=tpq // 4, pitch=42, velocity=90))
        s.tracks.append(drums)
    return s


def test_window_is_bar_aligned_and_about_30s():
    score = synth_score(bars=64, bpm=120.0)          # 2 s per bar -> 15 bars
    bars = set(int(b) for b in rc.bar_boundaries(score))
    to_sec = rc.tick_to_seconds_fn(score)
    rng = random.Random(1)
    for _ in range(10):
        start, end = rc.choose_window(score, rng)
        assert start in bars and end in bars
        assert start % (4 * 480) == 0 and end % (4 * 480) == 0
        dur = to_sec(end) - to_sec(start)
        assert abs(dur - 30.0) <= 1.0, dur           # within half a bar
        assert (end - start) == 15 * 4 * 480


def test_window_respects_tempo_changes():
    score = synth_score(bars=64, bpm=60.0)          # 4 s per bar -> 7 or 8 bars
    to_sec = rc.tick_to_seconds_fn(score)
    start, end = rc.choose_window(score, random.Random(0))
    assert abs((to_sec(end) - to_sec(start)) - 30.0) <= 2.0


def test_short_file_uses_whole_file():
    score = synth_score(bars=12, bpm=120.0)          # 24 s total
    start, end = rc.choose_window(score, random.Random(0))
    assert start == 0 and end == int(score.end())


def test_extract_pins_tempo_and_shifts_to_zero():
    score = synth_score(bars=64, bpm=97.0)
    start, end = rc.choose_window(score, random.Random(3))
    ex = rc.extract_excerpt(score, start, end)
    assert ex.tempos[0].time == 0 and abs(ex.tempos[0].qpm - 97.0) < 1e-3
    assert ex.time_signatures[0].time == 0 and ex.time_signatures[0].numerator == 4
    assert min(n.start for t in ex.tracks for n in t.notes) >= 0
    assert max(n.end for t in ex.tracks for n in t.notes) <= end - start


def test_features_have_no_nans():
    score = synth_score()
    start, end = rc.choose_window(score, random.Random(0))
    ex = rc.extract_excerpt(score, start, end)
    feats = rc.compute_features(ex, {"n_tracks": 3, "n_notes": 1000,
                                     "duration_seconds": 128.0})
    for k, v in feats.items():
        assert isinstance(v, (int, float)), k
        assert math.isfinite(v), k
    assert feats["has_drums"] == 1.0
    assert feats["n_programs"] == 2
    assert feats["n_tracks"] == 3
    assert abs(feats["tempo_bpm"] - 120.0) < 1e-6
    assert abs(feats["duration_seconds"] - 30.0) < 1e-3
    assert 1.0 <= feats["polyphony_rate"] <= 4.0
    assert feats["file_duration_seconds"] == 128.0


def test_rating_log_round_trips(tmp_path):
    log = tmp_path / "sub" / "ratings.jsonl"
    rec = {"ts": rc.utcnow(), "excerpt_id": "abc", "path": "/raw/lakh/x.mid",
           "source": "lakh", "rating": 4, "flags": [], "is_repeat": False,
           "features": {"n_notes": 10, "has_drums": 1.0}}
    rc.append_record(log, rec)
    rc.append_record(log, {**rec, "rating": None, "skipped": True})
    rc.append_record(log, {**rec, "rating": 5, "is_repeat": True})
    back = rc.load_ratings(log)
    assert len(back) == 3 and back[0] == rec
    stats = rc.compute_stats(back)
    assert stats["rated"] == 2 and stats["skipped"] == 1 and stats["repeats"] == 1
    sc = stats["self_consistency"]
    assert sc["pairs"] == 1 and sc["exact_agreement"] == 0.0 and sc["mean_abs_diff"] == 1.0
    assert stats["per_source"]["lakh"]["mean"] == 4.5


def test_source_of():
    assert rc.source_of("/Users/x/midigenai_data/raw/aria/pruned/a.mid") == "aria"
    assert rc.source_of("/data/raw/gigamidi/foo/b.mid") == "gigamidi"
    assert rc.source_of("/elsewhere/c.mid") == "unknown"


def test_spearman_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 200).astype(float)          # heavy ties, like ratings
    b = a + rng.normal(0, 1.5, 200)
    assert abs(qp.spearman(a, b) - scipy.spearmanr(a, b).correlation) < 1e-9


def synth_records(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sources = ["lakh", "aria", "lamd", "maestro"]
    recs = []
    for i in range(n):
        f1, f2 = rng.normal(), rng.normal()
        feats = {"scale_consistency": f1, "repetition_rate": f2,
                 "pitch_range": rng.normal(), "n_tracks": int(rng.integers(1, 9)),
                 "has_drums": bool(rng.integers(0, 2)),
                 "file_duration_seconds": None if i % 7 == 0 else float(rng.uniform(30, 300))}
        rating = int(np.clip(np.round(3 + 0.9 * f1 - 0.7 * f2 + rng.normal(0, 0.4)), 1, 5))
        recs.append({"excerpt_id": f"e{i}", "path": f"/raw/{sources[i % 4]}/{i // 2}.mid",
                     "source": sources[i % 4], "rating": rating, "features": feats,
                     "is_repeat": False})
    # some repeats and skips, which fit must average / drop
    for i in range(0, 30):
        recs.append({**recs[i], "rating": min(5, recs[i]["rating"] + (i % 2)),
                     "is_repeat": True})
    recs.append({**recs[0], "rating": None, "skipped": True})
    return recs


def test_fit_recovers_linear_signal(tmp_path):
    recs = synth_records()
    model, report = qp.fit_predictor(recs, k=5, seed=0, try_gbr=False)
    assert report["n_excerpts"] == 300 and report["n_ratings"] == 330
    assert report["ridge_cv"]["spearman"] > 0.8
    coef = report["coefficients"]
    assert coef["scale_consistency"] > 0 and coef["repetition_rate"] < 0
    assert abs(coef["scale_consistency"]) > abs(coef["pitch_range"])
    assert set(report["per_source"]) == {"lakh", "aria", "lamd", "maestro"}
    assert report["self_consistency"]["pairs"] == 30
    path = tmp_path / "predictor.json"
    model.save(path)
    again = qp.RidgeModel.load(path)
    X = qp.feature_matrix([r["features"] for r in recs[:5]], model.names)
    assert np.allclose(model.predict(X), again.predict(X))
    assert 1.0 <= again.predict_features({"scale_consistency": 0.0}) <= 5.0


def test_score_writes_quartile_buckets(tmp_path):
    # a model over the real feature names, so score() can featurize MIDI
    names = ["tempo_bpm", "n_notes", "has_drums"]
    model = qp.RidgeModel(names, medians=[120, 500, 0.5], means=[120, 500, 0.5],
                          stds=[30, 200, 0.5], coef=[0.6, 0.1, 0.1],
                          intercept=3.0, alpha=1.0)
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as f:
        for i in range(12):
            p = tmp_path / f"f{i}.mid"
            synth_score(bars=8 + 4 * i, bpm=60 + 10 * i, seed=i,
                        with_drums=bool(i % 2)).dump_midi(p)
            f.write(json.dumps({"path": str(p), "n_notes": 1, "n_tracks": 3,
                                "duration_seconds": 40.0}) + "\n")
        f.write(json.dumps({"path": str(tmp_path / "missing.mid")}) + "\n")
    summary = qp.score_manifest(model, manifest, tmp_path / "scores", workers=1,
                                max_seconds=60.0)
    assert summary["scored"] == 12 and summary["errors"] == 1
    rows = [json.loads(l) for l in (tmp_path / "scores.jsonl").open()]
    assert len(rows) == 12
    assert {r["q_bucket"] for r in rows} == {0, 1, 2, 3}
    assert all(1.0 <= r["quality"] <= 5.0 for r in rows)
    assert all(r["source"] == "unknown" for r in rows)


def test_seconds_to_tick_bounds_long_files():
    score = synth_score(bars=64, bpm=120.0)          # 128 s
    cut = qp.seconds_to_tick(score, 60.0)
    assert abs(rc.tick_to_seconds_fn(score)(cut) - 60.0) < 0.01
    feats = qp.file_features(str(score_path(score)), 60.0)
    assert abs(feats["duration_seconds"] - 60.0) < 0.01
    assert feats["file_duration_seconds"] == pytest.approx(128.0)


def score_path(score, _cache={}):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "long.mid"
    score.dump_midi(p)
    return p


def test_app_is_blind_and_logs_features(tmp_path):
    pytest.importorskip("flask")
    raw = tmp_path / "raw"
    files = []
    for src in ("lakh", "aria"):
        d = raw / src
        d.mkdir(parents=True)
        for i in range(3):
            p = d / f"{i}.mid"
            synth_score(bars=40, bpm=100 + 10 * i, seed=i).dump_midi(p)
            files.append(p)
    (raw / "lakh" / "broken.mid").write_bytes(b"not a midi file")
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as f:
        for p in files + [raw / "lakh" / "broken.mid"]:
            f.write(json.dumps({"path": str(p), "n_notes": 100, "n_tracks": 3,
                                "duration_seconds": 96.0}) + "\n")
        f.write(json.dumps({"path": str(raw / "aria" / "short.mid"), "n_notes": 5,
                            "n_tracks": 1, "duration_seconds": 5.0}) + "\n")
    args = types.SimpleNamespace(
        manifests=[str(manifest)], source_weights=None, seed=0, rater="test",
        repeat_rate=0.0, min_before_repeat=1, out=str(tmp_path / "quality"),
        pool_per_source=100, queue_size=2, next_timeout=30.0)
    app = rc.build_app(args)
    client = app.test_client()

    r = client.get("/api/next").get_json()
    assert r["status"] == "ok"
    item = r["item"]
    assert set(item) == {"item_id", "url", "eid"}            # nothing identifying
    assert "raw" not in item["url"] and item["url"].endswith(".mid")
    assert client.get(item["url"]).status_code == 200

    ok = client.post("/api/rate", json={"item_id": item["item_id"], "action": "rate",
                                        "rating": 4, "session_id": "s1"})
    assert ok.status_code == 200
    item2 = client.get("/api/next").get_json()["item"]
    client.post("/api/rate", json={"item_id": item2["item_id"], "action": "junk"})
    item3 = client.get("/api/next").get_json()["item"]
    client.post("/api/rate", json={"item_id": item3["item_id"], "action": "skip"})
    # rating the same item twice is rejected
    assert client.post("/api/rate", json={"item_id": item["item_id"], "action": "rate",
                                          "rating": 2}).status_code == 400

    recs = rc.load_ratings(tmp_path / "quality" / "ratings.jsonl")
    assert [r["rating"] for r in recs] == [4, 1, None]
    assert recs[1]["flags"] == ["junk"] and recs[2]["skipped"]
    first = recs[0]
    assert first["rater"] == "test" and first["source"] in ("lakh", "aria")
    assert Path(first["excerpt_file"]).exists()
    assert first["end_tick"] > first["start_tick"]
    assert abs((first["end_seconds"] - first["start_seconds"]) - 30) <= 2
    for k in ("pitch_class_entropy", "repetition_rate", "n_programs", "has_drums",
              "notes_per_second", "tempo_bpm", "pitch_range", "file_n_tracks",
              "file_duration_seconds"):
        assert k in first["features"] and first["features"][k] is not None

    stats = client.get("/api/stats").get_json()
    assert stats["rated"] == 2 and stats["skipped"] == 1
    assert client.get("/stats").status_code == 200

    # blind repeats: with repeat_rate=1 every next item is a rated excerpt
    args.repeat_rate = 1.0
    app2 = rc.build_app(args)
    c2 = app2.test_client()
    rep = c2.get("/api/next").get_json()["item"]
    c2.post("/api/rate", json={"item_id": rep["item_id"], "action": "rate", "rating": 5})
    last = rc.load_ratings(tmp_path / "quality" / "ratings.jsonl")[-1]
    assert last["is_repeat"] is True
    assert last["excerpt_id"] in {r["excerpt_id"] for r in recs if r["rating"]}
    assert rc.compute_stats(rc.load_ratings(tmp_path / "quality" / "ratings.jsonl")
                            )["self_consistency"]["pairs"] == 1
    for a in (app, app2):
        a.config["RATE_FACTORY"].stop()
