"""Blind re-labeling of already-voted pairs: the self-consistency ceiling.

Every judge number in `evals/reward/` is scored against a labeler whose own
repeat-agreement is essentially unmeasured — there are 4 repeated pairs in
the entire preference corpus. Without that ceiling an agreement of 0.74 is
uninterpretable: it is either a judge at the limit of the signal or a judge
with 15 points of headroom.

`label_app --dup-rate` only re-serves pairs voted *within the same live
session*, so it cannot measure the ceiling on the historical pairs the judge
is actually scored on. This serves those pairs from disk instead: blind (the
UI is never told it is a repeat), with the sides re-randomised independently
of how they were shown the first time, and with the vote written to a
separate file so the original labels are never touched.

    python -m midigenai.relabel_app select -n 60 --out evals/ceiling
    python -m midigenai.relabel_app serve  --out evals/ceiling
    python -m midigenai.relabel_app score  --out evals/ceiling
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The four label sets and where their pair MIDI actually lives. The pairs/
# dirs are gitignored, so these are local working copies rather than repo
# paths -- hence home-relative defaults plus an override, instead of one
# machine's absolute layout baked into a committed module.
#
#   MIDIGENAI_SETS="v4=/some/where,v1=/else"    overrides individual sets
#   MIDIGENAI_WORKTREE=~/src/midigenai-v4       moves the v4-era sets together
_V4_WORKTREE = os.environ.get("MIDIGENAI_WORKTREE", "~/midigenai-v4")
_DEFAULT_SETS = {
    "v1": "~/midigenai/evals/labeling",
    "v3_same": f"{_V4_WORKTREE}/evals/labeling_v3_same",
    "v4": f"{_V4_WORKTREE}/evals/labeling_v4",
    "v4_final": f"{_V4_WORKTREE}/evals/labeling_v4_final",
}


def _resolve_sets(defaults: dict) -> dict:
    """Expand `~` and apply MIDIGENAI_SETS overrides (`name=path`, comma-separated)."""
    out = {k: os.path.expanduser(v) for k, v in defaults.items()}
    for item in os.environ.get("MIDIGENAI_SETS", "").split(","):
        name, _, path = item.partition("=")
        if name.strip() and path.strip():
            out[name.strip()] = os.path.expanduser(path.strip())
    return out


DEFAULT_SETS = _resolve_sets(_DEFAULT_SETS)
PROMPT_TAIL_BEATS = 8.0  # label_app's default; only used when no timeline exists


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def decided(set_dir: Path) -> dict[str, str]:
    """pair_id -> the side ("a"/"b") the labeler picked, last vote wins."""
    out: dict[str, str] = {}
    labels = set_dir / "labels.jsonl"
    if not labels.exists():
        return out
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("preferred") in ("a", "b"):
            out[r["pair_id"]] = r["preferred"]
    return out


def playable(pairs_dir: Path, pid: str) -> bool:
    return all((pairs_dir / f"{pid}_{s}.mid").exists() for s in ("prompt", "a", "b"))


def select_dir(args) -> None:
    """Manifest for a directory of pre-generated pairs (e.g. pairgen output).

    Unlike the ceiling manifest these have no prior vote to agree with, so
    there is no `original`; they are fresh labels. `--dup` adds that many
    pairs a second time, which the server re-serves blind with the sides
    randomised again — the same self-consistency instrument label_app gets
    from --dup-rate, and the thing whose absence left the GRPO A/B without a
    ceiling to read against.
    """
    pairs_dir = Path(args.pairs).resolve()
    ids = sorted({f.name[:-len("_prompt.mid")]
                  for f in pairs_dir.glob("*_prompt.mid")
                  if (pairs_dir / f"{f.name[:-len('_prompt.mid')]}_a.mid").exists()})
    if not ids:
        raise SystemExit(f"no complete pairs in {pairs_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    picked = ids[:args.n]
    rows = [{"set": args.name, "pairs_dir": str(pairs_dir), "pair_id": pid,
             "original": None} for pid in picked]
    for pid in rng.sample(picked, min(args.dup, len(picked))):
        rows.append({"set": args.name, "pairs_dir": str(pairs_dir),
                     "pair_id": pid, "original": None, "is_repeat": True})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["idx"] = i                      # duplicates need distinct identities
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[select] {len(picked)} pairs + {len(rows) - len(picked)} blind repeats "
          f"-> {out_dir / 'manifest.jsonl'}")


def select(args) -> None:
    sets = {k: Path(v) for k, v in DEFAULT_SETS.items()}
    pool = []
    for name, d in sets.items():
        pairs_dir = d / "pairs"
        got = [(name, str(pairs_dir), pid, win)
               for pid, win in decided(d).items() if playable(pairs_dir, pid)]
        print(f"[select] {name}: {len(got)} decided pairs with playable MIDI")
        pool.append(got)

    # stratified in proportion to each set's size, so the ceiling is measured
    # on the same mix of models the judge is scored on
    total = sum(len(g) for g in pool)
    rng = random.Random(args.seed)
    picked = []
    for got in pool:
        if not got:
            continue
        k = max(1, round(args.n * len(got) / total))
        picked += rng.sample(got, min(k, len(got)))
    rng.shuffle(picked)
    picked = picked[:args.n]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as f:
        for name, pairs_dir, pid, win in picked:
            f.write(json.dumps({"set": name, "pairs_dir": pairs_dir,
                                "pair_id": pid, "original": win}) + "\n")
    by_set: dict[str, int] = {}
    for name, *_ in picked:
        by_set[name] = by_set.get(name, 0) + 1
    print(f"[select] wrote {len(picked)} pairs to {manifest}: {by_set}")


# ---------------------------------------------------------------- serving

def _roll_of(path: Path, tempo: float) -> dict | None:
    """Note list for one MIDI file, in the shape the UI's roll expects."""
    from symusic import Score
    try:
        sc = Score(str(path))
    except Exception:
        return None
    tpq = max(sc.ticks_per_quarter, 1)
    spt = 60.0 / (tempo * tpq)
    notes = [{"s": round(n.start * spt, 3),
              "e": round((n.start + n.duration) * spt, 3),
              "p": int(n.pitch), "v": int(n.velocity),
              "d": bool(t.is_drum), "prompt": True}
             for t in sc.tracks for n in t.notes]
    return {"notes": notes, "prompt_end_s": 0.0} if notes else None


def build_roll(pairs_dir: Path, pid: str, side: str, cache: Path) -> tuple[dict, str]:
    """(roll JSON, url path of the MIDI to play) for one side.

    Prefers the `_timeline.mid` written at generation time (prompt tail +
    continuation). The older label sets have no timeline files, so there the
    timeline is rebuilt from the prompt's last few beats plus the
    continuation — the same construction label_app used.
    """
    from symusic import Score, Tempo

    tl_path = pairs_dir / f"{pid}_{side}_timeline.mid"
    cont_path = pairs_dir / f"{pid}_{side}.mid"
    meta_path = pairs_dir / f"{pid}.json"
    tempo, mode = 120.0, "continue"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        tempo = float(meta.get("tempo_bpm") or 120.0)
        mode = meta.get("mode", "continue")

    if mode == "accompany":
        # The side file is the condition already mixed with its accompaniment,
        # so there is nothing to prepend and no instant where "the model takes
        # over" — both parts sound from the first beat. prompt_ticks 0 leaves
        # the roll unshaded and drops the handoff line.
        tl = Score(str(cont_path))
        tpq = max(tl.ticks_per_quarter, 1)
        prompt_ticks = 0
        url_name = f"{pid}_{side}.mid"
        served_from = pairs_dir
    elif tl_path.exists():
        tl = Score(str(tl_path))
        tpq = max(tl.ticks_per_quarter, 1)
        cont_end = max((n.start + n.duration
                        for t in Score(str(cont_path)).tracks for n in t.notes),
                       default=0)
        tl_end = max((n.start + n.duration for t in tl.tracks for n in t.notes),
                     default=0)
        prompt_ticks = max(0, tl_end - cont_end)
        url_name = f"{pid}_{side}_timeline.mid"
        served_from = pairs_dir
    else:
        prompt = Score(str(pairs_dir / f"{pid}_prompt.mid"))
        cont = Score(str(cont_path))
        tpq = max(prompt.ticks_per_quarter, 1)
        p_end = max((n.start + n.duration for t in prompt.tracks for n in t.notes),
                    default=0)
        tail0 = max(0, p_end - int(PROMPT_TAIL_BEATS * tpq))
        prompt_ticks = p_end - tail0
        tl = prompt.copy()
        cont_by_prog = {(t.program, t.is_drum): t for t in cont.tracks}
        for t in tl.tracks:
            kept = [n for n in t.notes if n.start >= tail0]
            for n in kept:
                n.start -= tail0
            ct = cont_by_prog.get((t.program, t.is_drum))
            if ct is not None:
                for n in ct.notes:
                    m = n.copy()
                    m.start = n.start + prompt_ticks
                    kept.append(m)
            t.notes = kept
        cache.mkdir(parents=True, exist_ok=True)
        url_name = f"{pid}_{side}_rebuilt.mid"
        tl.tempos = [Tempo(time=0, qpm=tempo)]
        tl.dump_midi(str(cache / url_name))
        served_from = cache

    spt = 60.0 / (tempo * tpq)
    notes = []
    for t in tl.tracks:
        for n in t.notes:
            notes.append({"s": round(n.start * spt, 3),
                          "e": round((n.start + n.duration) * spt, 3),
                          "p": int(n.pitch), "v": int(n.velocity),
                          "d": bool(t.is_drum), "prompt": n.start < prompt_ticks})
    roll = {"notes": notes, "prompt_end_s": round(prompt_ticks * spt, 3)}
    return roll, f"{'pairs' if served_from is pairs_dir else 'cache'}/{url_name}"


class Source:
    """One pre-generated set served blind: its manifest, what is left to
    vote on, and where the votes go. A server holds several and the page
    switches between them with `?source=<name>`."""

    def __init__(self, out: str, seed: int):
        self.out_dir = Path(out).resolve()
        self.id = self.out_dir.name
        self.cache_dir = self.out_dir / "timeline_cache"
        self.labels_path = self.out_dir / "labels.jsonl"
        self.manifest = [json.loads(l) for l in (self.out_dir / "manifest.jsonl").read_text().splitlines()
                         if l.strip()]
        for i, m in enumerate(self.manifest):
            m.setdefault("idx", i)
        already = set()
        if self.labels_path.exists():
            already = {json.loads(l).get("idx", json.loads(l)["pair_id"])
                       for l in self.labels_path.read_text().splitlines() if l.strip()}
        self.todo = [m for m in self.manifest if m["idx"] not in already]
        random.Random(seed).shuffle(self.todo)
        # the sets this manifest actually references — a directory of
        # generated pairs is not one of DEFAULT_SETS, and validating against
        # that list is what made every MIDI request 403
        self.set_dirs = {m["set"]: Path(m["pairs_dir"]).resolve() for m in self.manifest}
        self.mode = self._mode()
        print(f"[relabel] {self.id}: {len(self.todo)} pairs left of {len(self.manifest)} ({self.mode})")

    def _mode(self) -> str:
        for m in self.manifest[:5]:
            mp = Path(m["pairs_dir"]) / f"{m['pair_id']}.json"
            try:
                if json.loads(mp.read_text()).get("mode") == "accompany":
                    return "accompaniment"
            except Exception:
                continue
        return "continuation"

    def voted(self) -> set:
        if not self.labels_path.exists():
            return set()
        return {json.loads(l).get("idx", json.loads(l)["pair_id"])
                for l in self.labels_path.read_text().splitlines() if l.strip()}

    def skip_voted(self) -> None:
        """Drop pairs voted since startup — by another server on the same
        set, or another tab — so nothing is served twice."""
        done = self.voted()
        while self.todo and self.todo[0]["idx"] in done:
            self.todo.pop(0)

    def counts(self) -> dict:
        total = decided = 0
        if self.labels_path.exists():
            for l in self.labels_path.read_text().splitlines():
                if not l.strip():
                    continue
                total += 1
                if json.loads(l).get("choice") in ("left", "right"):
                    decided += 1
        return {"total_labels": total, "decided": decided, "queued": len(self.todo)}

    def entry(self) -> dict:
        c = self.counts()
        return {"id": self.id, "mode": self.mode, **c,
                "label": f"{self.id} · {self.mode} · {c['decided']} decided, {c['queued']} left"}


SETS_CONFIG = "labeling_sets.json"


def sets_config(sets_dir: Path | None) -> dict:
    """<sets_dir>/labeling_sets.json: {"order": [basename, ...], "hidden":
    [basename, ...]}. Listed sets come first in that order and the first
    unfinished one is the default; hidden sets are served if asked for by
    name but never listed. Missing or broken file: no order, nothing hidden."""
    if sets_dir is None:
        return {"order": [], "hidden": []}
    try:
        cfg = json.loads((sets_dir / SETS_CONFIG).read_text())
        return {"order": list(cfg.get("order", [])), "hidden": list(cfg.get("hidden", []))}
    except (FileNotFoundError, ValueError):
        return {"order": [], "hidden": []}


def discover_sets(sets_dir: Path) -> list[Path]:
    """Every labeling_*/ under `sets_dir` with a manifest: a set another
    session just wrote appears here without anyone restarting anything."""
    return sorted((d for d in sets_dir.glob("labeling_*") if (d / "manifest.jsonl").exists()),
                  key=lambda d: (d / "manifest.jsonl").stat().st_mtime, reverse=True)


def build_app(args):
    from flask import Flask, Response, jsonify, request, send_from_directory

    from midigenai.label_app import retract_vote, sources_payload

    sets_dir = Path(args.sets_dir).resolve() if getattr(args, "sets_dir", None) else None
    outs = list(args.out or [])
    sources: dict[str, Source] = {}

    def add_source(out) -> None:
        try:
            src = Source(out, args.seed)
        except Exception as e:
            print(f"[relabel] cannot serve {out}: {type(e).__name__}: {e}")
            return
        sources[src.id] = src

    last_scan = 0.0

    def rescan(force: bool = False) -> None:
        # the hub picks up sets written after it started; cheap (a glob), so
        # it runs on every /api/sources and before every /api/next
        nonlocal last_scan
        if sets_dir is None or (not force and time.time() - last_scan < args.rescan):
            return
        last_scan = time.time()
        for d in discover_sets(sets_dir):
            if d.name not in sources:
                add_source(d)

    for out in outs:
        add_source(out)
    rescan(force=True)
    if not sources:
        raise SystemExit("nothing to serve: pass --out <dir> or --sets-dir <dir with labeling_*/manifest.jsonl>")

    def ordered() -> list[Source]:
        # sets with work left first — those named in labeling_sets.json in
        # that order, then the rest newest-manifest-first — finished ones
        # last; hidden sets never listed
        cfg = sets_config(sets_dir)
        rank = {name: i for i, name in enumerate(cfg["order"])}
        shown = [s for s in sources.values() if s.id not in cfg["hidden"]]
        return sorted(shown, key=lambda s: (not s.todo, rank.get(s.id, len(rank)),
                                            -(s.out_dir / "manifest.jsonl").stat().st_mtime))

    def pick() -> Source:
        # a set named in the query or the vote body; the first one otherwise,
        # so a single-set server behaves exactly as before
        name = request.args.get("source") or (
            (request.get_json(silent=True) or {}).get("source") if request.method == "POST" else None)
        rescan()
        return sources.get(name) or ordered()[0]

    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "label.html")

    def hub_sources(current: str):
        rescan()
        return jsonify(sources_payload([s.entry() for s in ordered()], current,
                                       request.host.split(":")[0], proxy_live=args.live))

    @app.route("/api/sources")
    def api_sources():
        return hub_sources(pick().id)

    # ---- live-generation servers (label_app) behind this one's URL, so a
    # single tunnel reaches them: /live/<port>/<anything> -> 127.0.0.1:<port>.
    # The page uses paths relative to its own URL, so it works unchanged
    # under the prefix; only the source list is answered here, with hub URLs.
    @app.route("/live/<int:port>")
    def live_root(port):
        from flask import redirect
        return redirect(f"/live/{port}/")

    @app.route("/live/<int:port>/", defaults={"rest": ""}, methods=["GET", "POST"])
    @app.route("/live/<int:port>/<path:rest>", methods=["GET", "POST"])
    def live_proxy(port, rest):
        if not args.live:
            return "live proxy off", 404
        if rest == "api/sources":
            return hub_sources(f"live:{port}")
        url = f"http://127.0.0.1:{port}/{rest}"
        if request.query_string:
            url += "?" + request.query_string.decode()
        req = urllib.request.Request(url, method=request.method,
                                     data=request.get_data() if request.method == "POST" else None)
        if request.content_type:
            req.add_header("Content-Type", request.content_type)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return Response(r.read(), status=r.status,
                                content_type=r.headers.get("Content-Type", "application/octet-stream"))
        except urllib.error.HTTPError as e:
            return Response(e.read(), status=e.code,
                            content_type=e.headers.get("Content-Type", "text/plain"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return jsonify({"error": f"live server :{port} unreachable: {e}"}), 502

    @app.route("/api/next")
    def next_pair():
        src = pick()
        src.skip_voted()
        todo = src.todo
        if not todo:
            return jsonify({"status": "done", "source": src.id}), 200
        m = todo[0]
        pairs_dir = Path(m["pairs_dir"])
        pid = m["pair_id"]
        try:
            rolls, urls = {}, {}
            for side in ("a", "b"):
                rolls[side], urls[side] = build_roll(pairs_dir, pid, side, src.cache_dir)
            mp = pairs_dir / f"{pid}.json"
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            mode = meta.get("mode", "continue")
            prompt_roll = None
            if mode == "accompany":
                # the condition on its own, so it can be seen as well as heard
                prompt_roll = _roll_of(pairs_dir / f"{pid}_prompt.mid",
                                       float(meta.get("tempo_bpm") or 120.0))
        except Exception as e:
            print(f"[relabel] skipping {pid}: {type(e).__name__}: {e}")
            todo.pop(0)
            return jsonify({"status": "generating"}), 202
        # sides re-randomised independently of the original session: a repeat
        # that always showed the same way round would measure memory, not taste
        left, right = ("a", "b") if random.random() < 0.5 else ("b", "a")
        base = f"/midi/{src.id}/{m['set']}"
        pair = {
            "pair_id": pid, "idx": m["idx"], "source": src.id,
            "prompt_source": "repeat check",
            "prompt_name": Path(meta.get("prompt_file", "")).name,
            "prompt_url": f"{base}/pairs/{pid}_prompt.mid",
            "left_url": f"{base}/{urls[left]}",
            "right_url": f"{base}/{urls[right]}",
            "left_timeline_url": f"{base}/{urls[left]}",
            "right_timeline_url": f"{base}/{urls[right]}",
            "left_roll": rolls[left], "right_roll": rolls[right],
            "mode": mode, "prompt_roll": prompt_roll,
            "left_is": left, "right_is": right,
            "left_model": "", "right_model": "",
        }
        return jsonify({"status": "ok", "pair": pair, "queued": len(todo) - 1})

    @app.route("/api/vote", methods=["POST"])
    def vote():
        data = request.get_json(force=True)
        src = sources.get(data.get("source")) or ordered()[0]
        choice = data.get("choice")
        # the shared template can also post "drums_as_piano": a defect report
        # about the pair, not a preference, so it is recorded as a non-vote
        if choice not in ("left", "right", "tie", "bad", "skip", "bad_prompt",
                          "drums_as_piano"):
            return jsonify({"error": f"bad choice {choice!r}"}), 400
        pid = data.get("pair_id", "")
        rec = {
            "ts": utcnow(), "session_id": data.get("session_id", ""),
            "pair_id": pid, "idx": data.get("idx"), "choice": choice,
            "left_is": data.get("left_is", ""), "right_is": data.get("right_is", ""),
            "preferred": (data.get(f"{choice}_is", "")
                          if choice in ("left", "right") else choice),
            "replay": True,
        }
        with src.labels_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if src.todo and src.todo[0]["pair_id"] == pid:
            src.todo.pop(0)
        return jsonify({"ok": True})

    @app.route("/api/undo", methods=["POST"])
    def undo():
        data = request.get_json(force=True)
        src = sources.get(data.get("source")) or ordered()[0]
        pid = data.get("pair_id", "")
        # with no pair named, the newest vote in this set — that covers a
        # mistake made before the page was reloaded (its history is gone)
        removed = retract_vote(src.labels_path, pid, data.get("session_id", "") if pid else "",
                               data.get("idx") if pid else None)
        if removed is None:
            return jsonify({"error": "no vote of yours on that pair"}), 404
        pid = removed.get("pair_id", pid)
        # back to the front of the line so it is served again next
        row = next((m for m in src.manifest if m["pair_id"] == pid and m["idx"] == removed.get("idx", m["idx"])), None)
        if row is not None and row not in src.todo:
            src.todo.insert(0, row)
        return jsonify({"ok": True, "removed": removed})

    @app.route("/api/stats")
    def stats():
        return jsonify(pick().counts())

    @app.route("/midi/<src_id>/<set_name>/<kind>/<path:name>")
    def serve_midi(src_id, set_name, kind, name):
        src = sources.get(src_id)
        if src is None or set_name not in src.set_dirs or kind not in ("pairs", "cache"):
            return "forbidden", 403
        base = src.cache_dir.resolve() if kind == "cache" else src.set_dirs[set_name]
        full = (base / name).resolve()
        if base not in full.parents or full.suffix.lower() not in (".mid", ".midi"):
            return "forbidden", 403
        return send_from_directory(base, name)

    return app


# ---------------------------------------------------------------- scoring

def score(args) -> None:
    out_dir = Path(args.out)
    manifest = {m["pair_id"]: m for m in
                (json.loads(l) for l in (out_dir / "manifest.jsonl").read_text().splitlines() if l.strip())}
    labels_path = out_dir / "labels.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"no votes yet at {labels_path}")

    repeats = {}
    for line in labels_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            repeats[r["pair_id"]] = r  # last vote wins

    both, agree = [], 0
    by_choice: dict[str, int] = {}
    unusable = []  # both sides bad: the pair itself is not rankable
    for pid, r in repeats.items():
        m = manifest.get(pid)
        if not m:
            continue
        if r.get("preferred") not in ("a", "b"):
            choice = r.get("choice", "skip")
            by_choice[choice] = by_choice.get(choice, 0) + 1
            if choice == "bad":
                unusable.append(pid)
            continue
        both.append(pid)
        agree += r["preferred"] == m["original"]
    # "both bad" is a verdict on the pair, not an abstention by the labeler:
    # neither rater should be scored on a pair with no good answer, so those
    # leave the denominator entirely rather than counting against anyone.
    abstained = sum(n for c, n in by_choice.items() if c != "bad")

    n = len(both)
    if not n:
        raise SystemExit("no pairs decided in both passes yet")
    p = agree / n
    se = (p * (1 - p) / n) ** 0.5
    n_seen = n + abstained
    skip_rate = abstained / n_seen if n_seen else 0.0

    print(f"[ceiling] {n} pairs decided in both passes, {abstained} skipped on "
          f"the repeat ({skip_rate:.0%}), {len(unusable)} both-bad (excluded)")
    if by_choice:
        print(f"[ceiling] repeat non-votes by kind: {by_choice}")
    print(f"[ceiling] self-consistency: {p:.3f}  \u00b1{1.96 * se:.3f} (95% CI)")
    print(f"[ceiling] this is the number to read against the judge: both are "
          f"scored on what they were willing to decide")
    # A skip on the repeat is a pair that was called once and could not be
    # called again — evidence it is a coin flip, not evidence of a wrong
    # answer. Counting them as disagreements is a floor, not the ceiling: it
    # holds the labeler to a stricter rule than the judge, whose ties are
    # likewise dropped rather than scored wrong.
    print(f"[ceiling] pessimistic floor (skips counted as disagreement): "
          f"{agree / n_seen:.3f} on {n_seen}")
    print(f"[ceiling] abstention: labeler {skip_rate:.0%} vs judge 15% ties — "
          f"if these track, the two raters find the same pairs too close")
    result = {"n": n, "self_consistency": p, "se": se, "n_abstained": abstained,
              "skip_rate": skip_rate, "floor_skips_as_disagreement": agree / n_seen,
              "non_votes_by_kind": by_choice, "unusable_pairs": unusable,
              "pairs": both}
    if unusable:
        (out_dir / "unusable_pairs.txt").write_text("\n".join(unusable) + "\n")
        print(f"[ceiling] wrote {len(unusable)} both-bad pair ids to "
              f"{out_dir / 'unusable_pairs.txt'} — drop these from judge scoring too")
    (out_dir / "ceiling.json").write_text(json.dumps(result, indent=1))
    print(f"[ceiling] wrote {out_dir / 'ceiling.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="choose which already-voted pairs to redo")
    s.add_argument("-n", type=int, default=60)
    s.add_argument("--out", default="evals/ceiling")
    s.add_argument("--seed", type=int, default=0)

    sd = sub.add_parser("select-dir",
                        help="manifest for a directory of pre-generated pairs")
    sd.add_argument("--pairs", required=True)
    sd.add_argument("-n", type=int, default=80)
    sd.add_argument("--dup", type=int, default=10, help="blind repeats to add")
    sd.add_argument("--name", default="pairs")
    sd.add_argument("--out", default="evals/labeling_accompany")
    sd.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("serve", help="serve them blind for re-voting")
    v.add_argument("--out", action="append", default=None,
                   help="a set to serve; repeat for several (the page switches "
                        "between them; the first is the default)")
    v.add_argument("--sets-dir", default=None,
                   help="hub mode: serve every labeling_*/ with a manifest under this "
                        "directory, and pick up new ones as they appear; an optional "
                        f"<dir>/{SETS_CONFIG} fixes the order and hides sets")
    v.add_argument("--rescan", type=float, default=30.0,
                   help="hub mode: seconds between looks for new sets")
    v.add_argument("--live", action=argparse.BooleanOptionalAction, default=True,
                   help="reach running label_app (live generation) servers through "
                        "/live/<port>/ on this server, so one tunnel covers them")
    v.add_argument("--port", type=int, default=7789)
    v.add_argument("--seed", type=int, default=1)

    c = sub.add_parser("score", help="agreement between the two passes")
    c.add_argument("--out", default="evals/ceiling")

    a = p.parse_args()
    if a.cmd == "select":
        select(a)
    elif a.cmd == "select-dir":
        select_dir(a)
    elif a.cmd == "score":
        score(a)
    else:
        if not a.out and not a.sets_dir:
            a.out = ["evals/ceiling"]
        build_app(a).run(port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
