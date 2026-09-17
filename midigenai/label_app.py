"""
Local preference-labeling app: generates A/B continuation pairs *live* from the
current model and records which one you prefer, with everything needed to train
on the result (token ids, model version, sampling params).

A background worker keeps a queue of pre-generated pairs so labeling never
waits on the model. Sides are randomized per pair. Votes append to an
append-only JSONL; MIDI + per-pair metadata land next to it.

Run (model pulled from the HF hub by default):
    python -m midigenai.label_app --prompts evals/prompts

Compare two checkpoints instead of self-vs-self:
    python -m midigenai.label_app --prompts evals/prompts \\
        --hub-version v3 --hub-version-b v2-100m

Each pair plays itself: prompt + take 1, a short pause, prompt + take 2,
then the page waits for a vote and the next pair starts once it lands.
Keyboard: 1 = left, 2 = right, t = tie, x = both bad, s = skip, v = voice.
Voice (Chrome/Safari, localhost or https): say "one" / "two" / "tie" /
"both bad" / "skip" / "again" / "play one" / "play two".
Seed with your own MIDI: drop a .mid on the page (or use the file picker).
It is saved under <out>/uploads/, the next N pairs continue from it (head of
the file, as in production), and it then joins the regular prompt rotation.
Open http://localhost:7788.

Output layout (default --out evals/labeling):
    labels.jsonl          one line per vote
    uploads/user_*.mid    seeds dropped onto the page
    pairs/<id>.json       per-pair metadata (token ids, models, params)
    pairs/<id>_prompt.mid
    pairs/<id>_a.mid      prompt + continuation A
    pairs/<id>_b.mid      prompt + continuation B
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import queue
import random
import re
import shlex
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def retract_vote(labels_path: Path, pair_id: str, session_id: str = "",
                 idx=None) -> dict | None:
    """Remove the most recent record for `pair_id` (or the newest record of
    all when `pair_id` is empty; and this session's, when given) from an
    append-only labels file, and log it next door in
    corrections.jsonl so the retraction is traceable. Returns the removed
    record, or None when there was nothing to remove."""
    if not labels_path.exists():
        return None
    lines = labels_path.read_text().splitlines()
    hit = None
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        r = json.loads(lines[i])
        if pair_id and r.get("pair_id") != pair_id:   # no pair_id: the newest record
            continue
        if session_id and r.get("session_id") and r["session_id"] != session_id:
            continue
        if idx is not None and r.get("idx") is not None and r["idx"] != idx:
            continue
        hit = i
        break
    if hit is None:
        return None
    removed = json.loads(lines[hit])
    del lines[hit]
    labels_path.write_text("".join(l + "\n" for l in lines))
    with (labels_path.parent / "corrections.jsonl").open("a") as f:
        f.write(json.dumps({"ts": utcnow(), "action": "undo", "pair_id": pair_id,
                            "idx": removed.get("idx"), "removed": removed}) + "\n")
    return removed


def discover_label_servers() -> list[dict]:
    """Other label servers on this machine, read off their command lines.

    Each `label_app` / `relabel_app serve` process names its output dir(s)
    and port in argv, which is enough for the page's source selector to
    link across servers — including ones started by another session, which
    have no way to register themselves anywhere.
    """
    try:
        ps = subprocess.run(["ps", "-axo", "pid=,command="],
                            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    found = []
    for line in ps.splitlines():
        # only a python process running the module — not a shell whose
        # command string merely mentions it (the `zsh -c` that launched it)
        m = re.search(r"^\s*(\d+)\s+\S*python[\d.]* -m midigenai\.(re)?label_app\b(.*)$", line, re.I)
        if not m:
            continue
        pid, kind, rest = int(m.group(1)), ("relabel" if m.group(2) else "live"), m.group(3)
        if pid == os.getpid():
            continue
        try:
            argv = shlex.split(rest)
        except ValueError:
            argv = rest.split()
        val = lambda flag: [argv[i + 1] for i, a in enumerate(argv) if a == flag and i + 1 < len(argv)]
        outs = val("--out") or (["evals/labeling"] if kind == "live" else ["evals/ceiling"])
        try:
            port = int(val("--port")[0])
        except (IndexError, ValueError):
            port = 7788 if kind == "live" else 7789
        found.append({"pid": pid, "kind": kind, "port": port,
                      "outs": [Path(o).name for o in outs]})
    return sorted(found, key=lambda f: f["port"])


def sources_payload(local: list[dict], current: str, host: str,
                    proxy_live: bool = False) -> dict:
    """The selector's entries: this server's own sources first (`local`,
    each with id/label/mode/counts), then the other label servers found on
    the machine. A hub (`proxy_live`) reaches live-generation servers
    through its own /live/<port>/ path, so one tunnel covers them; other
    servers are plain links. Servers serving a set the hub already has are
    left out."""
    known = {src["id"] for src in local}
    entries = [{**src, "url": f"/?source={src['id']}", "here": True,
                "current": src["id"] == current} for src in local]
    for srv in discover_label_servers():
        for out in srv["outs"]:
            if out in known:
                continue
            if srv["kind"] == "live" and proxy_live:
                sid = f"live:{srv['port']}"
                entries.append({"id": sid, "here": True, "current": sid == current,
                                "mode": "continuation",
                                "label": f"{out} · live pairs from the model",
                                "url": f"/live/{srv['port']}/"})
                continue
            entries.append({
                "id": f"{srv['port']}:{out}", "here": False, "current": False,
                "label": f"{out} ({'live pairs' if srv['kind'] == 'live' else 'pre-generated'}) · :{srv['port']}",
                "url": f"http://{host}:{srv['port']}/" + (f"?source={out}" if srv["kind"] == "relabel" and len(srv["outs"]) > 1 else ""),
            })
    return {"current": current, "sources": entries}


class PairFactory:
    """Generates labeled-pair candidates in a background thread."""

    def __init__(self, args, out_dir: Path):
        self.args = args
        self.out_dir = out_dir
        self.pairs_dir = out_dir / "pairs"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = out_dir / "uploads"

        self.gen_a, self.label_a = load_generator(args, "a")
        gen_b, label_b = load_generator(args, "b")
        self.gen_b = gen_b or self.gen_a
        self.label_b = label_b or self.label_a
        self.cross_model = gen_b is not None

        self.blacklist_path = Path(args.out).resolve() / "bad_prompts.txt"
        blacklisted = set()
        if self.blacklist_path.exists():
            blacklisted = set(self.blacklist_path.read_text().split())
        self.prompt_files = [
            f for f in sorted(Path(args.prompts).glob("*.mid")) +
                       sorted(Path(args.prompts).glob("*.midi")) +
                       sorted(self.uploads_dir.glob("user_*.mid"))   # earlier seeds stay in rotation
            if f.name not in blacklisted]
        self.prompt_order = {f: i for i, f in enumerate(
            random.Random(12345).sample(self.prompt_files, len(self.prompt_files)))} \
            if self.prompt_files else {}
        self.prompt_uses = {f: 0 for f in self.prompt_files}
        self._prompt_lock = threading.Lock()
        # carry over usage from pairs already generated into this output dir
        for meta in Path(args.out).glob("pairs/*.json"):
            try:
                used = Path(json.loads(meta.read_text())["prompt_file"])
            except Exception:
                continue
            if used in self.prompt_uses:
                self.prompt_uses[used] += 1
        if not self.prompt_files:
            raise SystemExit(f"no .mid files in {args.prompts}")
        print(f"[label] {len(self.prompt_files)} prompt files; "
              f"models: {self.label_a} vs {self.label_b}")

        self.queue: queue.Queue[dict] = queue.Queue(maxsize=args.queue_size)
        # Seeds dropped onto the page jump the line: the worker generates from
        # them before anything else and /api/next serves them first, so an
        # upload is heard within one generation rather than after the whole
        # pre-generated queue has drained.
        self.priority: collections.deque[Path] = collections.deque()
        self.priority_queue: queue.Queue[dict] = queue.Queue()
        self.priority_pending = 0          # requested, not yet in priority_queue
        self.rng = random.Random()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _slice_with_program(self, ids: list[int], start: int, n: int) -> list[int]:
        """Window of `n` tokens starting at `start`, with the instrument that
        was sounding at that point restored in front of it.

        `Program_*` tokens are stateful: everything after one keeps that
        program until the next. A window cut after the file's `Program_-1`
        therefore decodes a drum kit as piano — which is exactly what
        happened to drum-only prompts before this.
        """
        if not hasattr(self, "_inv"):
            self._inv = {v: k for k, v in self.gen_a.tokenizer.vocab.items()}
        prefix: list[int] = []
        for j in range(start - 1, -1, -1):
            if self._inv.get(ids[j], "").startswith("Program_"):
                prefix = [ids[j]]
                break
        return prefix + list(ids[start:start + n])

    def _next_prompt(self) -> Path:
        """Round-robin over a shuffled prompt list, least-used first.

        Sampling with replacement made the same prompt come up five times in
        150 pairs; prompts already used in this output directory start with
        that many uses, so a restart continues rather than resets.
        """
        with self._prompt_lock:
            f = min(self.prompt_files, key=lambda p: (self.prompt_uses[p], self.prompt_order[p]))
            self.prompt_uses[f] += 1
            return f

    def _generate_one(self, prompt_file: Path | None = None) -> dict:
        args = self.args
        if prompt_file is None:
            prompt_file = self._next_prompt()
        else:   # a seed served out of turn still counts as used in the rotation
            with self._prompt_lock:
                self.prompt_uses[prompt_file] = self.prompt_uses.get(prompt_file, 0) + 1
        from symusic import Score

        from midigenai.tokenizer import normalize_drums
        prompt_score = Score(str(prompt_file))
        # fix drum tracks mislabeled as pitched (Ableton exports, v1-era files)
        normalize_drums(prompt_score, prompt_file.name)
        prompt_ids = self.gen_a.tokenizer(prompt_score).ids
        if len(prompt_ids) > args.prompt_tokens:
            # a dropped seed is continued from its head, as production does
            # with an upload; corpus prompts take a random window so one long
            # file yields many different pairs
            start = 0 if prompt_file.name.startswith("user_") else \
                self.rng.randrange(0, len(prompt_ids) - args.prompt_tokens)
            prompt_ids = self._slice_with_program(prompt_ids, start, args.prompt_tokens)
        tempo = self.gen_a.detect_tempo(prompt_file)

        pair_id = f"{datetime.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)
        # cross-model fairness: each side can run at its own best temperature
        kwargs_by_side = {
            "a": gen_kwargs,
            "b": {**gen_kwargs,
                  "temperature": args.temperature_b or args.temperature},
        }

        from symusic import Tempo

        prompt_path = self.pairs_dir / f"{pair_id}_prompt.mid"
        prompt_score = self.gen_a.tokenizer.decode(list(prompt_ids))
        prompt_score.tempos = [Tempo(time=0, qpm=tempo)]
        prompt_score.dump_midi(prompt_path)          # unscaled: models read this
        vscale = velocity_scale(prompt_score) if args.normalize_velocity else 1.0

        # Cross-vocabulary pairs (e.g. v3 MIDILike vs a v4 REMI checkpoint):
        # each side re-tokenizes the SAME prompt score with its own tokenizer.
        # A v4 side also gets the prompt's attribute header and a closed bar,
        # which is how it is prompted in production.
        side_prompt_ids = {}
        conts = {}
        rolls = {}
        candidate_log = None

        def side_ids(gen):
            if gen is self.gen_a and not getattr(gen, "v4", False):
                return list(prompt_ids)
            ids = gen.tokenizer(Score(str(prompt_path))).ids
            if getattr(gen, "v4", False):
                # `--v4-close-bar` pads the prompt to its bar line so the answer
                # lands on a downbeat. Off by default here: it moves v4's
                # starting point later than v3's, so the two rows would show
                # different slices of the prompt and v4 would open with up to a
                # bar of padding. The downbeat behaviour is measured properly in
                # eval_checkpoint --pad-to-bar; a blind A/B wants both models
                # continuing from the identical instant.
                if args.v4_close_bar:
                    ids = gen.close_bar(ids)
                ids = [*gen.make_header(prompt_path), *ids]
            return ids

        def continuation(gen, ids, kw):
            """Returns (new_ids, continuation-only Score, timeline Score,
            note list). The timeline Score is the last `--prompt-tail-beats`
            of the prompt followed by the continuation, so the two sides of a
            pair line up in time; the note list (seconds) drives the piano
            roll in the UI, with prompt notes marked."""
            decoded_prompt = gen.tokenizer.decode(list(ids))
            cut_tick = decoded_prompt.end()
            if getattr(gen, "v4", False) and args.v4_close_bar:
                # a padded prompt ends at its bar line, and the tokenizer emits
                # nothing after the last note, so the handoff is the bar line
                from midigenai.attributes import ticks_per_bar
                nbars = gen.count_bars(ids)
                if nbars > 1:
                    cut_tick = max(cut_tick, (nbars - 1) * ticks_per_bar(decoded_prompt))
            new_ids = list(gen.generate_ids(ids, **kw))
            full = gen.tokenizer.decode(list(ids) + new_ids)
            tpq = max(full.ticks_per_quarter, 1)
            tail0 = max(0, cut_tick - int(args.prompt_tail_beats * tpq))
            spt = 60.0 / (tempo * tpq)
            cap = cut_tick + int(args.max_cont_seconds * tempo / 60.0 * tpq)
            cont = full.copy()
            timeline = full.copy()
            notes = []
            for track, ct, tt in zip(full.tracks, cont.tracks, timeline.tracks):
                ct.notes = [n for n in track.notes if cut_tick <= n.start < cap]
                for n in ct.notes:
                    n.start -= cut_tick
                tt.notes = [n for n in track.notes if tail0 <= n.start < cap]
                for n in tt.notes:
                    n.start -= tail0
                for n in tt.notes:
                    # the roll carries the velocities the served file plays at
                    # (scaled below), so the page's loudness estimate is right
                    notes.append({"s": round(n.start * spt, 3), "e": round((n.start + n.duration) * spt, 3),
                                  "p": int(n.pitch), "v": max(1, min(127, int(round(n.velocity * vscale)))),
                                  "d": bool(track.is_drum), "prompt": n.start < (cut_tick - tail0)})
            cont.tempos = [Tempo(time=0, qpm=tempo)]
            timeline.tempos = [Tempo(time=0, qpm=tempo)]
            apply_velocity_scale(cont, vscale)
            apply_velocity_scale(timeline, vscale)
            return new_ids, cont, timeline, {"notes": notes, "prompt_end_s": round((cut_tick - tail0) * spt, 3)}

        if not self.cross_model and args.candidates > 2:
            # Curated same-model pairs: sample K continuations, drop the
            # degenerate ones (near-empty, stuck on one pitch, looping), and
            # show two plausible survivors. Votes between two plausible
            # answers teach the reward about musical choices; a vote against
            # a broken sample adds nothing the metrics don't already know.
            # Never done in cross-model mode (per-model cherry-picking biases).
            ids = side_ids(self.gen_a)
            side_prompt_ids = {"a": ids, "b": ids}
            cands = []
            for k in range(args.candidates):
                new_ids, sc, tl, nj = continuation(self.gen_a, ids, kwargs_by_side["a"])
                cands.append((new_ids, sc, _degeneracy(sc), tl, nj))
            ok = [c for c in cands if not c[2]["degenerate"]]
            pool = ok if len(ok) >= 2 else sorted(cands, key=lambda c: c[2]["badness"])[:2]
            picked = self.rng.sample(pool, 2)
            for name, (new_ids, sc, m, tl, nj) in zip(("a", "b"), picked):
                conts[name] = new_ids
                sc.dump_midi(self.pairs_dir / f"{pair_id}_{name}.mid")
                tl.dump_midi(self.pairs_dir / f"{pair_id}_{name}_timeline.mid")
                rolls[name] = nj
            candidate_log = [{**c[2], "chosen": c in picked, "n_ids": len(c[0])} for c in cands]
        else:
            for name, gen in (("a", self.gen_a), ("b", self.gen_b)):
                ids = side_ids(gen)
                side_prompt_ids[name] = ids
                new_ids, sc, tl, nj = continuation(gen, ids, kwargs_by_side[name])
                conts[name] = new_ids
                sc.dump_midi(self.pairs_dir / f"{pair_id}_{name}.mid")
                tl.dump_midi(self.pairs_dir / f"{pair_id}_{name}_timeline.mid")
                rolls[name] = nj

        meta = {
            "pair_id": pair_id,
            "created": utcnow(),
            "prompt_file": str(prompt_file),
            "prompt_ids": prompt_ids,
            "prompt_ids_a": side_prompt_ids["a"],
            "prompt_ids_b": side_prompt_ids["b"],
            "cont_a_ids": conts["a"],
            "cont_b_ids": conts["b"],
            "model_a": self.label_a,
            "model_b": self.label_b,
            "cross_model": self.cross_model,
            "tempo_bpm": tempo,
            "temperature_b": kwargs_by_side["b"]["temperature"],
            "candidates": candidate_log,
            **gen_kwargs,
        }
        (self.pairs_dir / f"{pair_id}.json").write_text(json.dumps(meta))

        # randomize which continuation shows on which side
        left, right = ("a", "b") if self.rng.random() < 0.5 else ("b", "a")
        name = prompt_file.name
        source = ("your upload" if name.startswith("user_")
                  else f"held-out val ({name.split('_')[1]})" if name.startswith("val_")
                  else "prompt set")
        return {
            "pair_id": pair_id,
            "prompt_source": source,
            "prompt_name": name,
            "prompt_url": f"/midi/{pair_id}_prompt.mid",
            "left_url": f"/midi/{pair_id}_{left}.mid",
            "right_url": f"/midi/{pair_id}_{right}.mid",
            "left_timeline_url": f"/midi/{pair_id}_{left}_timeline.mid",
            "right_timeline_url": f"/midi/{pair_id}_{right}_timeline.mid",
            "left_roll": rolls[left],
            "right_roll": rolls[right],
            "left_is": left,
            "right_is": right,
            "left_model": meta[f"model_{left}"],
            "right_model": meta[f"model_{right}"],
        }

    def _pop_priority(self) -> Path | None:
        with self._prompt_lock:
            return self.priority.popleft() if self.priority else None

    def _run(self):
        held = None   # a regular pair waiting for room in the queue
        while not self._stop.is_set():
            seed = self._pop_priority()
            if seed is not None:
                try:
                    self.priority_queue.put(self._generate_one(seed))
                except Exception as e:
                    print(f"[label] generation error on upload {seed.name}: {e}")
                finally:
                    with self._prompt_lock:
                        self.priority_pending -= 1
                continue
            if held is None:
                try:
                    held = self._generate_one()
                except Exception as e:  # keep the worker alive on bad prompt files
                    print(f"[label] generation error: {e}")
                    continue
            # a short wait, not a blocking put: a seed dropped while the queue
            # is full must not sit behind it
            try:
                self.queue.put(held, timeout=1.0)
                held = None
            except queue.Full:
                pass

    def add_upload(self, data: bytes, filename: str, n_pairs: int) -> Path:
        """Save a dropped MIDI file as a seed and queue `n_pairs` pairs from
        it ahead of everything else. Afterwards it stays in the rotation like
        any other prompt. Raises ValueError for a file symusic cannot read."""
        from symusic import Score
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")[:40] or "seed"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        path = self.uploads_dir / f"user_{stem}_{uuid.uuid4().hex[:6]}.mid"
        path.write_bytes(data)
        try:
            score = Score(str(path))
        except Exception as e:
            path.unlink(missing_ok=True)
            raise ValueError(f"not a readable MIDI file: {e}") from None
        if sum(len(t.notes) for t in score.tracks) == 0:
            path.unlink(missing_ok=True)
            raise ValueError("the file has no notes")
        with self._prompt_lock:
            self.prompt_files.append(path)
            self.prompt_order[path] = len(self.prompt_order)
            self.prompt_uses[path] = 0
            for _ in range(n_pairs):
                self.priority.append(path)
            self.priority_pending += n_pairs
        print(f"[label] seed uploaded: {path.name} -> next {n_pairs} pair(s)")
        return path

    def stop(self):
        self._stop.set()


def build_app(args) -> Flask:
    # absolute: Flask's send_from_directory resolves relative paths against
    # the app root (v2/), not the CWD
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    factory = PairFactory(args, out_dir)
    app = Flask(__name__,
                template_folder=str(Path(__file__).parent / "templates"))

    # Pairs already voted on this session, eligible for blind re-serving.
    # Repeats measure the labeler's self-consistency — the accuracy ceiling
    # for any reward fit on these labels. The UI is never told it's a repeat.
    voted_pairs: dict[str, dict] = {}
    served: dict[str, dict] = {}        # the exact payload each pair was shown with
    repeat_rng = random.Random()

    # every field the UI needs that comes in left/right halves; a flipped
    # repeat has to swap all of them together or the page is inconsistent
    _SIDED = ("url", "timeline_url", "roll", "is", "model")

    def make_repeat() -> dict | None:
        candidates = [p for p in voted_pairs.values() if not p.get("_repeated")]
        if len(candidates) < args.min_before_repeat:
            return None
        pair = repeat_rng.choice(candidates)
        pair["_repeated"] = True
        out = dict(pair)
        out.pop("_repeated", None)
        if repeat_rng.random() < 0.5:
            # Swap every sided field, not just the URLs. Rebuilding a partial
            # payload here is what broke repeats before: the template reads
            # left_roll and left_timeline_url, and a payload without them
            # throws before anything renders — so the repeat was served, the
            # page died, and the vote never happened.
            for f in _SIDED:
                out[f"left_{f}"], out[f"right_{f}"] = pair[f"right_{f}"], pair[f"left_{f}"]
        return out

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "label.html")

    @app.route("/api/next")
    def next_pair():
        # pairs from a dropped seed come first, and while one is still being
        # generated the client waits for it rather than taking a corpus pair
        try:
            pair = factory.priority_queue.get_nowait()
        except queue.Empty:
            pair = None
            if factory.priority_pending > 0:
                try:
                    pair = factory.priority_queue.get(timeout=args.next_timeout)
                except queue.Empty:
                    return jsonify({"status": "generating"}), 202
        if pair is not None:
            served[pair["pair_id"]] = pair
            return jsonify({"status": "ok", "pair": pair,
                            "queued": factory.queue.qsize()})
        if repeat_rng.random() < args.dup_rate:
            repeat = make_repeat()
            if repeat is not None:
                return jsonify({"status": "ok", "pair": repeat,
                                "queued": factory.queue.qsize()})
        try:
            pair = factory.queue.get(timeout=args.next_timeout)
        except queue.Empty:
            return jsonify({"status": "generating"}), 202
        served[pair["pair_id"]] = pair
        return jsonify({"status": "ok", "pair": pair,
                        "queued": factory.queue.qsize()})

    @app.route("/api/vote", methods=["POST"])
    def vote():
        data = request.get_json(force=True)
        choice = data.get("choice")  # left | right | tie | bad | skip | bad_prompt
        # "drums_as_piano" is a defect report about the pair, not a preference:
        # recorded as a non-vote so the rendering bug can be traced later
        if choice not in ("left", "right", "tie", "bad", "skip", "bad_prompt",
                          "drums_as_piano"):
            return jsonify({"error": f"bad choice {choice!r}"}), 400
        if choice == "bad_prompt" and data.get("pair_id"):
            # the source example itself is unusable: blacklist it from future
            # serving here, and downstream from training/eval prompt sets
            meta_path = factory.pairs_dir / f"{data['pair_id']}.json"
            if meta_path.exists():
                pf = Path(json.loads(meta_path.read_text())["prompt_file"])
                with factory.blacklist_path.open("a") as bf:
                    bf.write(pf.name + "\n")
                factory.prompt_files = [f for f in factory.prompt_files
                                        if f.name != pf.name]
        record = {
            "ts": utcnow(),
            "session_id": data.get("session_id", ""),
            "pair_id": data.get("pair_id", ""),
            "choice": choice,
            "left_is": data.get("left_is", ""),
            "right_is": data.get("right_is", ""),
            # resolve to canonical a/b so downstream never depends on sides
            "preferred": (data.get(f"{choice}_is", "")
                          if choice in ("left", "right") else choice),
            "left_model": data.get("left_model", ""),
            "right_model": data.get("right_model", ""),
        }
        # optional per-side quality flags ("degrades", "too_long", "too_short"),
        # resolved to canonical a/b like the preference itself
        flags = data.get("flags") or {}
        record["flags"] = {
            data.get(f"{side}_is", side): sorted(set(v))
            for side, v in flags.items()
            if side in ("left", "right") and isinstance(v, list) and v
        }
        with labels_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if choice in ("left", "right", "tie") and record["pair_id"]:
            # re-serve exactly what was shown, rolls and timelines included
            full = served.get(record["pair_id"])
            if full is not None:
                voted_pairs.setdefault(record["pair_id"], full)
        return jsonify({"ok": True})

    @app.route("/api/upload", methods=["POST"])
    def upload():
        f = request.files.get("midi")
        if f is None or not f.filename:
            return jsonify({"error": "no file"}), 400
        if Path(f.filename).suffix.lower() not in (".mid", ".midi"):
            return jsonify({"error": "not a .mid file"}), 400
        try:
            n = max(1, min(20, int(request.form.get("n", 3))))
        except ValueError:
            n = 3
        try:
            path = factory.add_upload(f.read(), f.filename, n)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "name": path.name, "n": n})

    @app.route("/api/undo", methods=["POST"])
    def undo():
        data = request.get_json(force=True)
        removed = retract_vote(labels_path, data.get("pair_id", ""), data.get("session_id", ""))
        if removed is None:
            return jsonify({"error": "no vote of yours on that pair"}), 404
        # the page needs the pair back on screen; it is re-served as it was
        # shown if this process served it, else the page just moves on
        again = served.get(removed.get("pair_id"))
        if removed.get("choice") == "bad_prompt":
            # the prompt was blacklisted by that vote; let it back in
            meta_path = factory.pairs_dir / f"{removed['pair_id']}.json"
            if meta_path.exists():
                pf = Path(json.loads(meta_path.read_text())["prompt_file"])
                if factory.blacklist_path.exists():
                    kept = [n for n in factory.blacklist_path.read_text().split() if n != pf.name]
                    factory.blacklist_path.write_text("".join(n + "\n" for n in kept))
                if pf.exists() and pf not in factory.prompt_files:
                    factory.prompt_files.append(pf)
                    factory.prompt_order.setdefault(pf, len(factory.prompt_order))
                    factory.prompt_uses.setdefault(pf, 1)
        return jsonify({"ok": True, "removed": removed, "pair": again})

    @app.route("/api/stats")
    def stats():
        n = 0
        if labels_path.exists():
            with labels_path.open() as f:
                n = sum(1 for line in f if line.strip())
        return jsonify({"total_labels": n, "queued": factory.queue.qsize(),
                        "upload_pending": factory.priority_pending})

    @app.route("/api/sources")
    def sources():
        me = {"id": out_dir.name, "mode": "continuation",
              "label": f"{out_dir.name} · live {factory.label_a} vs {factory.label_b}"}
        return jsonify(sources_payload([me], me["id"], request.host.split(":")[0]))

    @app.route("/midi/<path:name>")
    def serve_midi(name):
        full = (factory.pairs_dir / name).resolve()
        if factory.pairs_dir.resolve() not in full.parents or \
                full.suffix.lower() not in (".mid", ".midi"):
            return "forbidden", 403
        return send_from_directory(factory.pairs_dir, name)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True,
                   help="directory of seed .mid files (held-out from training)")
    p.add_argument("--out", default="evals/labeling")
    # model A: local checkpoint takes precedence over hub version
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--hub-version", default=None)
    # model B (optional; omit for same-model preference pairs)
    p.add_argument("--checkpoint-b", default=None)
    p.add_argument("--tokenizer-b", default=None)
    p.add_argument("--hub-version-b", default=None)
    # sampling
    p.add_argument("--prompt-tokens", type=int, default=256)
    # ~64 notes / ~30-45s of music: enough to judge, short enough to label fast
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--v4-close-bar", action="store_true",
                   help="pad a v4 side's prompt to its bar line (production jam "
                        "behaviour). Off by default so both sides of a pair start "
                        "from the identical instant.")
    p.add_argument("--normalize-velocity", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="scale the review clips up to a usable listening level, "
                        "with one factor per pair so the two takes stay comparable")
    p.add_argument("--max-cont-seconds", type=float, default=8.0,
                   help="hard cap on continuation length in the review clips: notes "
                        "starting after this are dropped (a fixed token budget gives "
                        "wildly different durations across tempos)")
    p.add_argument("--prompt-tail-beats", type=float, default=8,
                   help="beats of prompt kept in front of each continuation in the "
                        "timeline view / playback")
    p.add_argument("--candidates", type=int, default=1,
                   help="same-model pairs only: sample this many continuations per "
                        "prompt, drop degenerate ones, show two plausible survivors")
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--temperature-b", type=float, default=None,
                   help="model B's temperature (cross-model fairness); defaults to --temperature")
    p.add_argument("--top-k", type=int, default=50)
    # plumbing
    p.add_argument("--queue-size", type=int, default=12)
    p.add_argument("--next-timeout", type=float, default=25.0)
    p.add_argument("--dup-rate", type=float, default=0.1,
                   help="probability of blindly re-serving an already-voted "
                        "pair (sides re-randomized) to measure self-consistency")
    p.add_argument("--min-before-repeat", type=int, default=5)
    p.add_argument("--port", type=int, default=7788)
    args = p.parse_args()

    app = build_app(args)
    print(f"[label] open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
