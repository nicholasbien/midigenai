"""
Local preference-labeling app: generates A/B continuation pairs *live* from the
current model and records which one you prefer, with everything needed to train
on the result (token ids, model version, sampling params).

A background worker keeps a queue of pre-generated pairs so labeling never
waits on the model. Sides are randomized per pair. Votes append to an
append-only JSONL; MIDI + per-pair metadata land next to it.

Run (model pulled from the HF hub by default):
    python -m v2.label_app --prompts evals/prompts

Compare two checkpoints instead of self-vs-self:
    python -m v2.label_app --prompts evals/prompts \\
        --hub-version v2-100m --hub-version-b v2-prod

Keyboard: 1 = left, 2 = right, t = tie, x = both bad, s = skip.
Open http://localhost:7788.

Output layout (default --out evals/labeling):
    labels.jsonl          one line per vote
    pairs/<id>.json       per-pair metadata (token ids, models, params)
    pairs/<id>_prompt.mid
    pairs/<id>_a.mid      prompt + continuation A
    pairs/<id>_b.mid      prompt + continuation B
"""

from __future__ import annotations

import argparse
import datetime
import json
import queue
import random
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_generator(args, side: str):
    """Build the generator for side 'a' or 'b'. Returns (generator, label)."""
    ckpt = getattr(args, f"checkpoint{'_b' if side == 'b' else ''}", None)
    hubv = getattr(args, f"hub_version{'_b' if side == 'b' else ''}", None)
    if side == "b" and ckpt is None and hubv is None:
        return None, None  # no distinct B model -> self-comparison
    from v2.generate_v2 import V2Generator
    if ckpt:
        tok = getattr(args, f"tokenizer{'_b' if side == 'b' else ''}", None)
        return V2Generator(ckpt, tok), Path(ckpt).stem
    from v2.hub import DEFAULT_VERSION, load_v2_from_hub
    version = hubv or DEFAULT_VERSION
    return load_v2_from_hub(version=version), version


class PairFactory:
    """Generates labeled-pair candidates in a background thread."""

    def __init__(self, args, out_dir: Path):
        self.args = args
        self.out_dir = out_dir
        self.pairs_dir = out_dir / "pairs"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

        self.gen_a, self.label_a = load_generator(args, "a")
        gen_b, label_b = load_generator(args, "b")
        self.gen_b = gen_b or self.gen_a
        self.label_b = label_b or self.label_a
        self.cross_model = gen_b is not None

        self.prompt_files = sorted(Path(args.prompts).glob("*.mid")) + \
                            sorted(Path(args.prompts).glob("*.midi"))
        if not self.prompt_files:
            raise SystemExit(f"no .mid files in {args.prompts}")
        print(f"[label] {len(self.prompt_files)} prompt files; "
              f"models: {self.label_a} vs {self.label_b}")

        self.queue: queue.Queue[dict] = queue.Queue(maxsize=args.queue_size)
        self.rng = random.Random()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _generate_one(self) -> dict:
        args = self.args
        prompt_file = self.rng.choice(self.prompt_files)
        prompt_ids = self.gen_a.encode_midi_file(prompt_file)
        if len(prompt_ids) > args.prompt_tokens:
            start = self.rng.randrange(0, len(prompt_ids) - args.prompt_tokens)
            prompt_ids = prompt_ids[start : start + args.prompt_tokens]
        tempo = self.gen_a.detect_tempo(prompt_file)

        pair_id = f"{datetime.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k)

        prompt_path = self.pairs_dir / f"{pair_id}_prompt.mid"
        score = self.gen_a.tokenizer.decode(list(prompt_ids))
        score.dump_midi(prompt_path)

        conts = {}
        for name, gen in (("a", self.gen_a), ("b", self.gen_b)):
            out_path = self.pairs_dir / f"{pair_id}_{name}.mid"
            conts[name] = gen.generate_to_midi(prompt_ids, out_path,
                                               tempo_bpm=tempo, **gen_kwargs)

        meta = {
            "pair_id": pair_id,
            "created": utcnow(),
            "prompt_file": str(prompt_file),
            "prompt_ids": prompt_ids,
            "cont_a_ids": conts["a"],
            "cont_b_ids": conts["b"],
            "model_a": self.label_a,
            "model_b": self.label_b,
            "cross_model": self.cross_model,
            "tempo_bpm": tempo,
            **gen_kwargs,
        }
        (self.pairs_dir / f"{pair_id}.json").write_text(json.dumps(meta))

        # randomize which continuation shows on which side
        left, right = ("a", "b") if self.rng.random() < 0.5 else ("b", "a")
        return {
            "pair_id": pair_id,
            "prompt_url": f"/midi/{pair_id}_prompt.mid",
            "left_url": f"/midi/{pair_id}_{left}.mid",
            "right_url": f"/midi/{pair_id}_{right}.mid",
            "left_is": left,
            "right_is": right,
            "left_model": meta[f"model_{left}"],
            "right_model": meta[f"model_{right}"],
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                pair = self._generate_one()
            except Exception as e:  # keep the worker alive on bad prompt files
                print(f"[label] generation error: {e}")
                continue
            self.queue.put(pair)  # blocks while the queue is full

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
    repeat_rng = random.Random()

    def make_repeat() -> dict | None:
        candidates = [p for p in voted_pairs.values() if not p.get("_repeated")]
        if len(candidates) < args.min_before_repeat:
            return None
        pair = repeat_rng.choice(candidates)
        pair["_repeated"] = True
        flipped = repeat_rng.random() < 0.5
        out = dict(pair)
        out.pop("_repeated", None)
        if flipped:
            out.update({
                "left_url": pair["right_url"], "right_url": pair["left_url"],
                "left_is": pair["right_is"], "right_is": pair["left_is"],
                "left_model": pair["right_model"],
                "right_model": pair["left_model"],
            })
        return out

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "label.html")

    @app.route("/api/next")
    def next_pair():
        if repeat_rng.random() < args.dup_rate:
            repeat = make_repeat()
            if repeat is not None:
                return jsonify({"status": "ok", "pair": repeat,
                                "queued": factory.queue.qsize()})
        try:
            pair = factory.queue.get(timeout=args.next_timeout)
        except queue.Empty:
            return jsonify({"status": "generating"}), 202
        return jsonify({"status": "ok", "pair": pair,
                        "queued": factory.queue.qsize()})

    @app.route("/api/vote", methods=["POST"])
    def vote():
        data = request.get_json(force=True)
        choice = data.get("choice")  # left | right | tie | bad | skip
        if choice not in ("left", "right", "tie", "bad", "skip"):
            return jsonify({"error": f"bad choice {choice!r}"}), 400
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
        with labels_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if choice in ("left", "right", "tie") and record["pair_id"]:
            voted_pairs.setdefault(record["pair_id"], {
                "pair_id": record["pair_id"],
                "prompt_url": f"/midi/{record['pair_id']}_prompt.mid",
                "left_url": f"/midi/{record['pair_id']}_a.mid",
                "right_url": f"/midi/{record['pair_id']}_b.mid",
                "left_is": "a", "right_is": "b",
                "left_model": data.get(
                    "left_model" if data.get("left_is") == "a" else "right_model", ""),
                "right_model": data.get(
                    "left_model" if data.get("left_is") == "b" else "right_model", ""),
            })
        return jsonify({"ok": True})

    @app.route("/api/stats")
    def stats():
        n = 0
        if labels_path.exists():
            with labels_path.open() as f:
                n = sum(1 for line in f if line.strip())
        return jsonify({"total_labels": n, "queued": factory.queue.qsize()})

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
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top-k", type=int, default=50)
    # plumbing
    p.add_argument("--queue-size", type=int, default=4)
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
