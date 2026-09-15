"""
Blind quality rating of TRAINING-CORPUS excerpts (not model outputs).

Serves ~30 s bar-aligned excerpts sampled from the manifest, stratified by
source so aria (70% of files) doesn't dominate, and records a 1-5 rating per
excerpt. The rater never sees the filename, source or path. 10% of served
items are blind repeats of already-rated excerpts so self-consistency can be
measured. Ratings + per-excerpt features feed midigenai.quality_predictor,
which fits a simple regressor used to downweight low-quality files when the
training corpus is built.

Run:
    python -m midigenai.rate_corpus_app --port 7795
    # open http://localhost:7795  (stats at /stats)

Keys: 1-5 rate (5 = excellent, 1 = junk) · j = junk/broken · s = skip
      r = replay · space = play/pause

Output (default --out evals/quality):
    ratings.jsonl          append-only, one line per event (rating or skip)
    excerpts/<sha1>.mid    the exact excerpt that was rated, re-listenable
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import queue
import random
import re
import sys
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from midigenai import eval as ev

SOURCES = ("lakh", "lamd", "aria", "gigamidi", "maestro", "pop909", "giantmidi")
_SOURCE_RE = re.compile(r"/raw/([^/]+)/")

DEFAULT_MANIFESTS = [
    "~/midigenai_data/manifest_all_dedup.jsonl",
    "~/midigenai_data/manifest_gigamidi_dedup.jsonl",
]
TARGET_SECONDS = 30.0
MIN_FILE_SECONDS = 20.0
MIN_EXCERPT_NOTES = 8       # windows with fewer notes are silence/tails: retry
WINDOW_TRIES = 4


# data-issue flags (keys d / i): metadata defects, rated separately from the
# music itself so the quality predictor learns craft, not tagging accidents
RATER_FLAGS = {"drums_as_piano",      # 'd': a drum part playing as pitched piano
               "missing_instruments"} # 'i': everything on program 0 / clearly untagged


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_of(path: str) -> str:
    m = _SOURCE_RE.search(path)
    src = m.group(1) if m else "unknown"
    return src if src in SOURCES else "unknown"


# ------------------------------- corpus pool ------------------------------- #

def load_pool(manifests: list[Path], seed: int, per_source: int = 4000,
              min_seconds: float = MIN_FILE_SECONDS) -> dict[str, list[dict]]:
    """Reservoir-sample up to `per_source` eligible rows per source.

    Deterministic given the seed and the manifest order; memory stays bounded
    for the ~2M-row manifests. Rows shorter than `min_seconds` are skipped.
    """
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    seen: Counter = Counter()
    for mpath in manifests:
        mpath = Path(mpath).expanduser()
        if not mpath.exists():
            print(f"[rate] manifest not found, skipping: {mpath}")
            continue
        with mpath.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if float(row.get("duration_seconds") or 0) < min_seconds:
                    continue
                src = source_of(row["path"])
                seen[src] += 1
                slim = {"path": row["path"],
                        "n_notes": row.get("n_notes"),
                        "n_tracks": row.get("n_tracks"),
                        "duration_seconds": row.get("duration_seconds")}
                pool = pools[src]
                if len(pool) < per_source:
                    pool.append(slim)
                else:
                    j = rng.randrange(seen[src])
                    if j < per_source:
                        pool[j] = slim
    for src, pool in pools.items():
        rng.shuffle(pool)
        print(f"[rate] {src}: {seen[src]} eligible files, pool {len(pool)}")
    return dict(pools)


def parse_source_weights(spec: str | None, sources: list[str]) -> dict[str, float]:
    weights = {s: 1.0 for s in sources}          # equal weight per source
    if spec:
        for kv in spec.split(","):
            k, v = kv.split(":")
            weights[k.strip()] = float(v)
    return {s: w for s, w in weights.items() if s in sources and w > 0}


# ------------------------------ excerpting --------------------------------- #

def tick_to_seconds_fn(score):
    """Piecewise-linear tick->seconds map honoring every tempo change."""
    tpq = max(int(score.ticks_per_quarter), 1)
    tempos = sorted(((int(t.time), float(t.qpm)) for t in score.tempos
                     if t.qpm and t.qpm > 0), key=lambda x: x[0])
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, tempos[0][1] if tempos else 120.0))
    times = np.array([t for t, _ in tempos], dtype=np.float64)
    qpms = np.array([q for _, q in tempos], dtype=np.float64)
    spt = 60.0 / (qpms * tpq)                      # seconds per tick per segment
    cum = np.concatenate([[0.0], np.cumsum(np.diff(times) * spt[:-1])])

    def f(tick) -> float:
        i = int(np.searchsorted(times, tick, side="right")) - 1
        i = max(i, 0)
        return float(cum[i] + (tick - times[i]) * spt[i])
    return f


def tempo_at(score, tick: int) -> float:
    best = 120.0
    for t in sorted(score.tempos, key=lambda t: t.time):
        if t.time <= tick and t.qpm > 0:
            best = float(t.qpm)
        elif t.time > tick:
            break
    return best


def time_signature_at(score, tick: int):
    best = (4, 4)
    for ts in sorted(score.time_signatures, key=lambda t: t.time):
        if ts.time <= tick:
            best = (int(ts.numerator), int(ts.denominator))
        else:
            break
    return best


def bar_boundaries(score) -> np.ndarray:
    """Downbeat ticks plus the score end as the final boundary. Falls back to
    4/4 bars from tpq when the file has no usable time signature."""
    end = int(score.end())
    db = np.asarray(score.get_downbeats(), dtype=np.int64)
    if len(db) < 2:
        bar = 4 * max(int(score.ticks_per_quarter), 1)
        db = np.arange(0, end + 1, bar, dtype=np.int64)
    db = db[db < end]
    return np.unique(np.concatenate([db, [end]]))


def choose_window(score, rng: random.Random,
                  target_seconds: float = TARGET_SECONDS) -> tuple[int, int]:
    """Random bar-aligned (start_tick, end_tick) whose bar count is the one
    closest to `target_seconds` at the file's (possibly changing) tempo."""
    bars = bar_boundaries(score)
    if len(bars) < 2:
        return 0, int(score.end())
    to_sec = tick_to_seconds_fn(score)
    secs = np.array([to_sec(int(b)) for b in bars])
    total = float(secs[-1])
    if total <= target_seconds + 2.0:
        return int(bars[0]), int(bars[-1])
    starts = [i for i in range(len(bars) - 1) if total - secs[i] >= target_seconds]
    i = rng.choice(starts) if starts else 0
    js = np.arange(i + 1, len(bars))
    j = int(js[np.argmin(np.abs(secs[js] - secs[i] - target_seconds))])
    return int(bars[i]), int(bars[j])


def extract_excerpt(score, start: int, end: int):
    """Clip [start, end) to a standalone Score starting at tick 0 with the
    tempo / time signature in effect at `start` pinned at time 0."""
    from symusic import Tempo, TimeSignature
    qpm = tempo_at(score, start)
    num, den = time_signature_at(score, start)
    ex = score.clip(start, end, clip_end=True).shift_time(-start)
    ex.tempos = [t for t in ex.tempos if t.time > 0]
    ex.tempos.insert(0, Tempo(time=0, qpm=qpm))
    ex.time_signatures = [t for t in ex.time_signatures if t.time > 0]
    ex.time_signatures.insert(0, TimeSignature(time=0, numerator=num, denominator=den))
    for track in ex.tracks:
        track.controls = [c for c in track.controls if c.time >= 0]
        track.pitch_bends = [c for c in track.pitch_bends if c.time >= 0]
    return ex


def model_view(excerpt, tokenizer):
    """What the model learns from this excerpt: the excerpt tokenized and
    decoded again (onsets/durations on the tokenizer's 1/8-beat grid,
    velocities in 32 bins, mislabeled drum tracks promoted), with the
    excerpt's tempo re-applied for playback since training strips tempo.
    Returns (roundtrip Score, token names)."""
    from symusic import Tempo

    from midigenai.tokenizer import normalize_drums
    ex = excerpt.copy()
    normalize_drums(ex, "")
    toks = tokenizer(ex)
    names = list(toks.tokens) if hasattr(toks, "tokens") else []
    rt = tokenizer.decode(toks.ids)
    qpm = excerpt.tempos[0].qpm if len(excerpt.tempos) else 120.0
    rt.tempos = [Tempo(time=0, qpm=qpm)]
    return rt, names


def excerpt_id(path: str, start: int, end: int) -> str:
    return hashlib.sha1(f"{path}|{start}|{end}".encode()).hexdigest()


# ------------------------------- features ---------------------------------- #

def polyphony_rate(score) -> float:
    """Mean number of pitched notes sounding at each note onset (incl. the
    onset's own note). Vectorized so whole-file scoring stays cheap; matches
    eval.polyphony_rate up to how exact-tie onsets are counted."""
    notes = list(ev._all_notes(score))
    if not notes:
        return 0.0
    starts = np.array([n.start for n in notes], dtype=np.int64)
    ends = starts + np.array([n.duration for n in notes], dtype=np.int64)
    s_sorted = np.sort(starts)
    e_sorted = np.sort(ends)
    started = np.searchsorted(s_sorted, starts, side="right")
    ended = np.searchsorted(e_sorted, starts, side="right")
    return float(np.mean(np.maximum(started - ended, 1)))


def _finite(x) -> float:
    x = float(x)
    return x if math.isfinite(x) else 0.0


def compute_features(score, file_meta: dict | None = None) -> dict:
    """Feature dict for an excerpt (or whole file). Every value is a finite
    number so the predictor can consume it directly."""
    file_meta = file_meta or {}
    tpq = max(int(score.ticks_per_quarter), 1)
    to_sec = tick_to_seconds_fn(score)
    end_tick = int(score.end())
    duration = max(to_sec(end_tick), 1e-6)
    tracks = [t for t in score.tracks if len(t.notes)]
    all_notes = sum(len(t.notes) for t in tracks)
    pitched = sum(len(t.notes) for t in tracks if not t.is_drum)
    drum_notes = all_notes - pitched
    programs = {int(t.program) for t in tracks if not t.is_drum}
    velocities = [n.velocity for t in tracks for n in t.notes]
    durations_beats = [n.duration / tpq for t in tracks if not t.is_drum for n in t.notes]
    feats = {
        # eval.py metrics (pitched notes only, drums excluded)
        "pitch_class_entropy": ev.pitch_class_entropy(score),
        "scale_consistency": ev.scale_consistency(score),
        "polyphony_rate": polyphony_rate(score),
        "note_density_hz": ev.note_density_hz(score),
        "pitch_range": ev.pitch_range(score),
        "repetition_rate": ev.repetition_rate(score, n=4),
        "ioi_entropy": ev.ioi_entropy(score),
        # excerpt-level extras
        "n_notes": all_notes,
        "n_pitched_notes": pitched,
        "n_tracks": len(tracks),
        "n_programs": len(programs),
        "has_drums": float(drum_notes > 0),
        "drum_fraction": drum_notes / all_notes if all_notes else 0.0,
        "notes_per_second": all_notes / duration,
        "tempo_bpm": tempo_at(score, 0),
        "duration_seconds": duration,
        "velocity_mean": float(np.mean(velocities)) if velocities else 0.0,
        "velocity_std": float(np.std(velocities)) if velocities else 0.0,
        "mean_note_beats": float(np.mean(durations_beats)) if durations_beats else 0.0,
        # whole-file context from the manifest
        "file_n_tracks": file_meta.get("n_tracks"),
        "file_n_notes": file_meta.get("n_notes"),
        "file_duration_seconds": file_meta.get("duration_seconds"),
    }
    return {k: (_finite(v) if v is not None else None) for k, v in feats.items()}


# -------------------------------- logging ---------------------------------- #

def append_record(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_ratings(log_path: Path) -> list[dict]:
    if not Path(log_path).exists():
        return []
    out = []
    with Path(log_path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def self_consistency(records: list[dict]) -> dict:
    """Agreement between repeated ratings of the same excerpt: exact-match
    rate and mean |difference| over consecutive rating pairs."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r.get("rating") is not None:
            by_key[r["excerpt_id"]].append(int(r["rating"]))
    exact, absdiff, pairs = 0, 0.0, 0
    for ratings in by_key.values():
        for a, b in zip(ratings, ratings[1:]):
            pairs += 1
            exact += int(a == b)
            absdiff += abs(a - b)
    return {"pairs": pairs,
            "exact_agreement": exact / pairs if pairs else None,
            "mean_abs_diff": absdiff / pairs if pairs else None}


def compute_stats(records: list[dict]) -> dict:
    rated = [r for r in records if r.get("rating") is not None]
    per_source: dict[str, dict] = {}
    for src, group in _group_by(rated, "source").items():
        vals = [int(r["rating"]) for r in group]
        per_source[src] = {"n": len(vals), "mean": sum(vals) / len(vals),
                           "hist": {k: vals.count(k) for k in range(1, 6)}}
    return {"total_events": len(records), "rated": len(rated),
            "skipped": sum(1 for r in records if r.get("rating") is None),
            "flagged_junk": sum(1 for r in rated if "junk" in (r.get("flags") or [])),
            "repeats": sum(1 for r in rated if r.get("is_repeat")),
            "unique_excerpts": len({r["excerpt_id"] for r in rated}),
            "per_source": dict(sorted(per_source.items())),
            "self_consistency": self_consistency(rated)}


def _group_by(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r.get(key, "unknown")].append(r)
    return out


def format_stats(stats: dict) -> str:
    lines = [f"ratings: {stats['rated']} ({stats['unique_excerpts']} unique excerpts, "
             f"{stats['repeats']} repeats, {stats['skipped']} skips, "
             f"{stats['flagged_junk']} junk flags)"]
    for src, d in stats["per_source"].items():
        hist = " ".join(f"{k}:{v}" for k, v in d["hist"].items())
        lines.append(f"  {src:<10} n={d['n']:<4} mean={d['mean']:.2f}   [{hist}]")
    sc = stats["self_consistency"]
    if sc["pairs"]:
        lines.append(f"self-consistency on {sc['pairs']} repeat pairs: exact "
                     f"{sc['exact_agreement']:.0%}, mean |diff| {sc['mean_abs_diff']:.2f}")
    else:
        lines.append("self-consistency: no repeats yet")
    return "\n".join(lines)


# ------------------------------ item factory ------------------------------- #

class ExcerptFactory:
    """Pre-renders excerpts in a background thread so the rater never waits."""

    def __init__(self, pools: dict[str, list[dict]], weights: dict[str, float],
                 excerpts_dir: Path, seed: int, queue_size: int = 5,
                 as_model: bool = False, exclude_paths: set[str] | None = None):
        # files already rated in earlier sessions are never served as new
        # items again (blind repeats go through reextract instead), so a
        # restart with the same seed picks up where the rater left off
        self.exclude_paths = set(exclude_paths or ())
        self.as_model = as_model
        self.tokenizer = None
        if as_model:
            from midigenai.tokenizer import build_tokenizer
            self.tokenizer = build_tokenizer()
        self.pools = {s: list(p) for s, p in pools.items() if s in weights}
        self.cursor = {s: 0 for s in self.pools}
        self.weights = weights
        self.excerpts_dir = excerpts_dir
        self.excerpts_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=queue_size)
        self.skipped_unparseable = 0
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _next_row(self) -> dict | None:
        live = [s for s in self.pools if self.cursor[s] < len(self.pools[s])]
        if not live:
            return None
        src = self.rng.choices(live, weights=[self.weights[s] for s in live])[0]
        row = self.pools[src][self.cursor[src]]
        self.cursor[src] += 1
        if row["path"] in self.exclude_paths:
            return self._next_row()
        return row

    def make_item(self, row: dict) -> dict | None:
        from symusic import Score
        try:
            score = Score(row["path"])
        except Exception as e:
            self.skipped_unparseable += 1
            print(f"[rate] unparseable, skipping: {row['path']} ({type(e).__name__})")
            return None
        if score.end() <= 0 or not any(len(t.notes) for t in score.tracks):
            return None
        for _ in range(WINDOW_TRIES):
            start, end = choose_window(score, self.rng)
            excerpt = extract_excerpt(score, start, end)
            if sum(len(t.notes) for t in excerpt.tracks) >= MIN_EXCERPT_NOTES:
                break
        else:
            return None
        return self.build_item(row, score, excerpt, start, end)

    def build_item(self, row: dict, score, excerpt, start: int, end: int) -> dict:
        to_sec = tick_to_seconds_fn(score)
        eid = excerpt_id(row["path"], start, end)
        midi_path = self.excerpts_dir / f"{eid}.mid"
        if not midi_path.exists():
            excerpt.dump_midi(midi_path)
        extra = self._model_extra(eid, excerpt)
        return {
            **extra,
            "item_id": uuid.uuid4().hex[:12],
            "excerpt_id": eid,
            "path": row["path"],
            "source": source_of(row["path"]),
            "start_tick": start, "end_tick": end,
            "start_seconds": round(to_sec(start), 3),
            "end_seconds": round(to_sec(end), 3),
            "n_bars": int(np.sum((bar_boundaries(score) >= start)
                                 & (bar_boundaries(score) < end))),
            "features": compute_features(excerpt, row),
            "excerpt_file": str(midi_path),
            "is_repeat": False,
        }

    def _model_extra(self, eid: str, excerpt) -> dict:
        """as-model view: roundtrip MIDI next to the excerpt + token names."""
        if not self.as_model:
            return {}
        model_path = self.excerpts_dir / f"{eid}.model.mid"
        rt, names = model_view(excerpt, self.tokenizer)
        rt.dump_midi(model_path)
        return {"as_model": True, "model_file": str(model_path),
                "tokens": names, "n_tokens": len(names)}

    def reextract(self, record: dict) -> dict | None:
        """Rebuild a previously rated excerpt (for repeats) from its log row."""
        from symusic import Score
        eid = record["excerpt_id"]
        midi_path = self.excerpts_dir / f"{eid}.mid"
        try:
            if not midi_path.exists() or self.as_model:
                score = Score(record["path"])
                excerpt = extract_excerpt(score, record["start_tick"], record["end_tick"])
                if not midi_path.exists():
                    excerpt.dump_midi(midi_path)
                extra = self._model_extra(eid, excerpt)
            else:
                extra = {}
        except Exception as e:
            print(f"[rate] cannot rebuild repeat {eid}: {e}")
            return None
        item = {k: record[k] for k in ("excerpt_id", "path", "source", "start_tick",
                                       "end_tick", "start_seconds", "end_seconds")
                if k in record}
        item.update({**extra, "item_id": uuid.uuid4().hex[:12], "n_bars": record.get("n_bars"),
                     "features": record.get("features") or {},
                     "excerpt_file": str(midi_path), "is_repeat": True})
        return item

    def _run(self):
        while not self._stop.is_set():
            row = self._next_row()
            if row is None:
                print("[rate] pool exhausted; restart with another --seed")
                return
            try:
                item = self.make_item(row)
            except Exception as e:   # keep the worker alive on odd files
                print(f"[rate] excerpt error on {row['path']}: {type(e).__name__}: {e}")
                continue
            if item is not None:
                self.queue.put(item)

    def stop(self):
        self._stop.set()


# ---------------------------------- app ------------------------------------ #

def build_app(args):
    from flask import Flask, jsonify, request, send_from_directory

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ratings.jsonl"
    excerpts_dir = out_dir / "excerpts"

    pools = load_pool([Path(m) for m in args.manifests], args.seed,
                      per_source=args.pool_per_source)
    if not pools:
        raise SystemExit("no eligible files in the given manifests")
    weights = parse_source_weights(args.source_weights, sorted(pools))
    print(f"[rate] source weights: {weights}")
    already = {r["path"] for r in load_ratings(log_path) if r.get("path")}
    if already:
        print(f"[rate] {len(already)} files already rated/seen; they will not be served again")
    factory = ExcerptFactory(pools, weights, excerpts_dir, args.seed,
                             queue_size=args.queue_size,
                             as_model=getattr(args, "as_model", False),
                             exclude_paths=already)

    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    pending: dict[str, dict] = {}           # item_id -> full item (server-side only)
    # rated excerpts eligible for blind repeats: latest log record per excerpt
    rated: dict[str, dict] = {}
    for r in load_ratings(log_path):
        if r.get("rating") is not None and r.get("excerpt_id"):
            rated[r["excerpt_id"]] = r
    repeated_this_session: set[str] = set()
    repeat_rng = random.Random(args.seed + 1)
    lock = threading.Lock()

    def make_repeat() -> dict | None:
        with lock:
            cands = [r for eid, r in rated.items() if eid not in repeated_this_session]
        if len(cands) < args.min_before_repeat:
            return None
        rec = repeat_rng.choice(cands)
        item = factory.reextract(rec)
        if item is not None:
            repeated_this_session.add(rec["excerpt_id"])
        return item

    def public(item: dict) -> dict:
        # BLIND: the browser only ever sees an opaque id and the audio url
        # (plus, in --as-model mode, the tokenizer roundtrip and its tokens,
        # which carry no identifying information either)
        out = {"item_id": item["item_id"], "url": f"/midi/{item['excerpt_id']}.mid",
               "eid": item["excerpt_id"][:8]}
        if item.get("as_model"):
            out["model_url"] = f"/midi/{item['excerpt_id']}.model.mid"
            out["tokens"] = " ".join(item.get("tokens") or [])
            out["n_tokens"] = item.get("n_tokens")
        return out

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "rate.html")

    @app.route("/api/next")
    def next_item():
        item = None
        if repeat_rng.random() < args.repeat_rate:
            item = make_repeat()
        if item is None:
            try:
                item = factory.queue.get(timeout=args.next_timeout)
            except queue.Empty:
                return jsonify({"status": "preparing"}), 202
        with lock:
            pending[item["item_id"]] = item
            served[item["excerpt_id"][:8]] = item
            if len(pending) > 200:       # forget stale unrated items
                for k in list(pending)[:-100]:
                    pending.pop(k, None)
        return jsonify({"status": "ok", "item": public(item),
                        "queued": factory.queue.qsize()})

    # excerpts served this process, by short id, so the rater can revisit one
    served: dict[str, dict] = {}

    @app.route("/api/item/<eid>")
    def get_item(eid):
        with lock:
            base = served.get(eid)
        if base is None:
            rec = next((r for e, r in rated.items() if e.startswith(eid)), None)
            base = factory.reextract(rec) if rec else None
            if base is None:
                return jsonify({"error": "unknown item"}), 404
        item = {**base, "item_id": uuid.uuid4().hex[:12], "is_repeat": False,
                "revision": True}
        with lock:
            pending[item["item_id"]] = item
        return jsonify({"status": "ok", "item": public(item)})

    @app.route("/api/rate", methods=["POST"])
    def rate():
        data = request.get_json(force=True)
        with lock:
            item = pending.pop(data.get("item_id", ""), None)
        if item is None:
            return jsonify({"error": "unknown or already-rated item"}), 400
        action = data.get("action")
        rating, flags = None, []
        if action == "rate":
            rating = int(data.get("rating"))
            if not 1 <= rating <= 5:
                return jsonify({"error": "rating must be 1-5"}), 400
        elif action == "junk":
            rating, flags = 1, ["junk"]
        elif action != "skip":
            return jsonify({"error": f"bad action {action!r}"}), 400
        # data-issue flags toggled by the rater (kept separate from the
        # rating so they can drive pipeline fixes, e.g. drum promotion)
        flags += [f for f in (data.get("flags") or []) if f in RATER_FLAGS and f not in flags]
        record = {
            "ts": utcnow(),
            "session_id": data.get("session_id", ""),
            "rater": args.rater,
            "item_id": item["item_id"],
            "excerpt_id": item["excerpt_id"],
            "path": item["path"],
            "source": item["source"],
            "start_tick": item["start_tick"], "end_tick": item["end_tick"],
            "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"],
            "n_bars": item.get("n_bars"),
            "rating": rating,
            "flags": flags,
            "is_repeat": bool(item.get("is_repeat")),
            "as_model": bool(item.get("as_model")),
            "revision": bool(item.get("revision")),
            "skipped": rating is None,
            "listen_seconds": data.get("listen_seconds"),
            "features": item["features"],
            "excerpt_file": item["excerpt_file"],
        }
        append_record(log_path, record)
        if rating is not None:
            with lock:
                rated.setdefault(item["excerpt_id"], record)
        return jsonify({"ok": True})

    @app.route("/api/stats")
    def stats_api():
        stats = compute_stats(load_ratings(log_path))
        stats["queued"] = factory.queue.qsize()
        stats["skipped_unparseable"] = factory.skipped_unparseable
        return jsonify(stats)

    @app.route("/stats")
    def stats_page():
        stats = compute_stats(load_ratings(log_path))
        body = format_stats(stats).replace("\n", "<br>")
        return (f"<html><body style='font:14px/1.6 ui-monospace,monospace;"
                f"padding:24px'><h2>corpus quality ratings</h2>{body}"
                f"<p><a href='/'>back to rating</a></p></body></html>")

    @app.route("/midi/<path:name>")
    def serve_midi(name):
        full = (excerpts_dir / name).resolve()
        if excerpts_dir.resolve() not in full.parents or full.suffix.lower() != ".mid":
            return "forbidden", 403
        return send_from_directory(excerpts_dir, name)

    app.config["RATE_LOG"] = log_path
    app.config["RATE_FACTORY"] = factory
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifests", nargs="*", default=None,
                   help="manifest JSONL files (default: manifest_all_dedup + "
                        "manifest_gigamidi_dedup under ~/midigenai_data)")
    p.add_argument("--source-weights", default=None,
                   help="e.g. lakh:2,aria:1 (default: equal weight per source)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rater", default="nicholas")
    p.add_argument("--repeat-rate", type=float, default=0.1,
                   help="probability of blindly re-serving a rated excerpt")
    p.add_argument("--min-before-repeat", type=int, default=10)
    p.add_argument("--out", default="evals/quality")
    p.add_argument("--pool-per-source", type=int, default=4000)
    p.add_argument("--queue-size", type=int, default=5)
    p.add_argument("--next-timeout", type=float, default=20.0)
    p.add_argument("--port", type=int, default=7795)
    p.add_argument("--as-model", action="store_true",
                   help="play the excerpt as the model sees it: tokenized and "
                        "decoded (1/8-beat grid, 32 velocity bins, tempo re-applied) "
                        "and show the token stream; 'o' toggles the original")
    args = p.parse_args(argv)
    if not args.manifests:
        args.manifests = [m for m in (Path(x).expanduser() for x in DEFAULT_MANIFESTS)
                          if m.exists()]
        if not args.manifests:
            raise SystemExit("no default manifests found; pass --manifests")

    app = build_app(args)
    print(f"[rate] open http://localhost:{args.port}   (stats: /stats)")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + format_stats(compute_stats(load_ratings(app.config["RATE_LOG"]))))
        app.config["RATE_FACTORY"].stop()


if __name__ == "__main__":
    main()
