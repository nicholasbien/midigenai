"""
Local A/B grading frontend.

Auto-discovers paired MIDI continuations from two directories (matched by file
stem), randomizes left/right position to avoid bias, and records preferences
to a CSV for use as RLHF training data.

Run:
    python -m midigenai.grade_app \\
        --dir-a evals/v1_samples --label-a v1 \\
        --dir-b evals/v2_samples --label-b v2 \\
        --prompt-dir preselected_midi \\
        --responses evals/grading_responses.csv

Then open http://localhost:7777 in a browser.

Browser plays MIDI directly via html-midi-player (no server-side rendering).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


def build_pairs(dir_a: Path, dir_b: Path) -> list[dict]:
    files_a = {f.stem: f for f in dir_a.glob("*.mid")}
    files_b = {f.stem: f for f in dir_b.glob("*.mid")}
    common = sorted(set(files_a) & set(files_b))
    return [{"key": k, "a": str(files_a[k]), "b": str(files_b[k])} for k in common]


def build_app(args) -> Flask:
    app = Flask(__name__,
                static_folder=str(Path(__file__).parent / "static"),
                template_folder=str(Path(__file__).parent / "templates"))

    pairs = build_pairs(args.dir_a, args.dir_b)
    if not pairs:
        raise SystemExit(f"no matching MIDI pairs between {args.dir_a} and {args.dir_b}")
    print(f"[grade] {len(pairs)} pairs loaded")

    args.responses.parent.mkdir(parents=True, exist_ok=True)
    if not args.responses.exists():
        with args.responses.open("w") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "pair_key", "left_model", "right_model",
                "preferred_side", "preferred_model", "scalar_left", "scalar_right",
            ])

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "grade.html")

    @app.route("/api/manifest")
    def manifest():
        # Per-load randomization: shuffle order + L/R sides per pair.
        rng = random.Random(uuid.uuid4().int)
        items = []
        for p in pairs:
            if rng.random() < 0.5:
                left, right = (args.label_a, p["a"]), (args.label_b, p["b"])
            else:
                left, right = (args.label_b, p["b"]), (args.label_a, p["a"])
            items.append({
                "key": p["key"],
                "prompt": str(args.prompt_dir / f"{p['key'].split('__')[0]}.mid"),
                "left_model": left[0], "left_url": f"/midi/{left[1]}",
                "right_model": right[0], "right_url": f"/midi/{right[1]}",
            })
        rng.shuffle(items)
        return jsonify({"items": items, "label_a": args.label_a, "label_b": args.label_b})

    @app.route("/midi/<path:fpath>")
    def serve_midi(fpath):
        # Allow serving any file under repo root that ends in .mid.
        full = Path(fpath).resolve()
        repo_root = Path(__file__).resolve().parent.parent
        if not str(full).startswith(str(repo_root)):
            return "forbidden", 403
        if full.suffix.lower() not in (".mid", ".midi"):
            return "forbidden", 403
        return send_from_directory(full.parent, full.name)

    @app.route("/api/preference", methods=["POST"])
    def preference():
        data = request.get_json(force=True)
        row = [
            datetime.datetime.utcnow().isoformat(),
            data.get("session_id", ""),
            data.get("pair_key", ""),
            data.get("left_model", ""),
            data.get("right_model", ""),
            data.get("preferred_side", ""),       # "left" | "right" | "tie"
            data.get("preferred_model", ""),       # resolved against left/right
            data.get("scalar_left", ""),
            data.get("scalar_right", ""),
        ]
        with args.responses.open("a") as f:
            csv.writer(f).writerow(row)
        return jsonify({"ok": True})

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir-a", type=Path, required=True)
    p.add_argument("--dir-b", type=Path, required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--prompt-dir", type=Path, default=Path("preselected_midi"))
    p.add_argument("--responses", type=Path,
                   default=Path("evals/grading_responses.csv"))
    p.add_argument("--port", type=int, default=7777)
    args = p.parse_args()

    app = build_app(args)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
