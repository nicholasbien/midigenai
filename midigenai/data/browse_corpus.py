"""
Browse the training corpus: per-dataset stats + a random-sample listener.

Loads a manifest written by `v2/data/clean.py`, groups entries by source
dataset (auto-detected from path), and serves a small browser at
http://localhost:7778:

- summary stats per source (file count, total notes, avg duration, etc.)
- "play random sample" button per source — streams the MIDI to the browser
  via html-midi-player, no server-side rendering

Usage:
    python -m midigenai.data.browse_corpus --manifest data/manifest.jsonl

If the manifest references files on a remote machine (e.g. you cleaned on
Lambda but want to browse locally), either rsync the dataset down first or
run this script on the remote machine and SSH-tunnel port 7778.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


SOURCE_PATTERNS = [
    ("lakh", ["/lakh/", "/lmd_full/", "/lmd_matched/"]),
    ("maestro", ["/maestro"]),
    ("pop909", ["/pop909/", "/POP909-"]),
    ("giantmidi", ["/giantmidi", "/GiantMIDI"]),
    ("lamd", ["/lamd/", "/Los-Angeles-MIDI"]),
]


def detect_source(path: str) -> str:
    p = path.lower()
    for name, patterns in SOURCE_PATTERNS:
        for pat in patterns:
            if pat.lower() in p:
                return name
    return "other"


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def group_by_source(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(detect_source(e["path"]), []).append(e)
    return groups


def summarize_group(entries: list[dict]) -> dict:
    if not entries:
        return {"n_files": 0}
    n_notes = [e["n_notes"] for e in entries]
    durs = [e["duration_seconds"] for e in entries]
    tracks = [e["n_tracks"] for e in entries]
    return {
        "n_files": len(entries),
        "total_notes": sum(n_notes),
        "median_notes": int(statistics.median(n_notes)),
        "median_duration_s": round(statistics.median(durs), 1),
        "median_tracks": int(statistics.median(tracks)),
        "p95_duration_s": round(sorted(durs)[int(len(durs) * 0.95)], 1) if len(durs) > 1 else durs[0],
    }


def build_app(manifest_path: Path, repo_root: Path) -> Flask:
    entries = load_manifest(manifest_path)
    groups = group_by_source(entries)
    # Set of canonical paths the manifest declares safe to serve.
    allowed_paths = {str(Path(e["path"]).resolve()) for e in entries}

    print(f"[browse] manifest: {manifest_path}  ({len(entries)} files)")
    for src, es in sorted(groups.items()):
        print(f"  {src:10s}: {len(es):>7d} files")

    static_dir = Path(__file__).parent.parent / "static"
    template_dir = Path(__file__).parent.parent / "templates"
    app = Flask(__name__,
                static_folder=str(static_dir),
                template_folder=str(template_dir))

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "browse_corpus.html")

    @app.route("/api/stats")
    def stats():
        return jsonify({
            "manifest": str(manifest_path),
            "total_files": len(entries),
            "sources": {src: summarize_group(es) for src, es in sorted(groups.items())},
        })

    @app.route("/api/sample")
    def sample():
        src = request.args.get("source")
        pool = groups.get(src, entries) if src else entries
        if not pool:
            return jsonify({"error": "no entries for source"}), 404
        e = random.choice(pool)
        return jsonify({
            "path": e["path"],
            "url": f"/midi/{e['path']}",
            "source": detect_source(e["path"]),
            "n_notes": e["n_notes"],
            "n_tracks": e["n_tracks"],
            "duration_s": round(e["duration_seconds"], 1),
        })

    @app.route("/midi/<path:fpath>")
    def serve_midi(fpath):
        # Browser hits /midi/<path>; Flask strips the leading slash, so
        # /home/ubuntu/foo.mid arrives as fpath="home/ubuntu/foo.mid".
        # Reattach the leading slash so absolute paths resolve correctly.
        candidate = "/" + fpath if not fpath.startswith("/") else fpath
        full = str(Path(candidate).resolve())
        # Only serve paths the manifest itself declared — manifest is the
        # access-control list; we don't expose anything else on disk.
        if full not in allowed_paths:
            return "forbidden", 403
        if not full.endswith((".mid", ".midi")):
            return "forbidden", 403
        return send_from_directory(Path(full).parent, Path(full).name)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    p.add_argument("--port", type=int, default=7778)
    p.add_argument("--stats-only", action="store_true",
                   help="print summary stats and exit (no server)")
    args = p.parse_args()

    if args.stats_only:
        entries = load_manifest(args.manifest)
        groups = group_by_source(entries)
        print(f"manifest: {args.manifest}  ({len(entries)} files)")
        print()
        print(f"{'source':10s} {'files':>10s} {'med notes':>10s} {'med dur':>10s} {'med tracks':>11s}")
        for src, es in sorted(groups.items()):
            s = summarize_group(es)
            print(f"{src:10s} {s['n_files']:>10d} {s['median_notes']:>10d} "
                  f"{s['median_duration_s']:>10.1f} {s['median_tracks']:>11d}")
        return

    repo_root = Path(__file__).resolve().parent.parent.parent
    app = build_app(args.manifest, repo_root)
    print(f"\nopen http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
