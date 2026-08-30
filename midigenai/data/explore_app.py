"""
Dataset explorer: sample candidate training corpora in the browser — listen to
random files, see their metadata (titles/columns where the source has them)
and computed note statistics — before committing to a multi-GB download.

Samplers pull small random samples without fetching whole archives:
- gigamidi   HF parquet, range-reads one row group (needs GigaMIDI terms
             accepted once at https://huggingface.co/datasets/Metacreation/GigaMIDI)
- aria       streams the tar.gz over HTTP, takes members after a random skip
- lakh       same, over the LMD-full tarball
- prompts    local evals/prompts dir (instant sanity check)

Run:
    python -m v2.data.explore_app
    # open http://localhost:7790

Samples are cached under evals/dataset_samples/<dataset>/ with a meta.jsonl.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

SAMPLE_LIMIT_BYTES = 80 * 1024 * 1024   # stop streaming a tarball after this


def note_stats(midi_path: Path) -> dict:
    try:
        from symusic import Score
        s = Score(str(midi_path))
        notes = sum(len(t.notes) for t in s.tracks)
        tpq = max(s.ticks_per_quarter, 1)
        bpm = s.tempos[0].qpm if len(s.tempos) else 120.0
        dur = s.end() / tpq * 60.0 / bpm
        programs = sorted({("drums" if t.is_drum else str(t.program))
                           for t in s.tracks if len(t.notes)}, key=str)
        return {"n_notes": notes, "n_tracks": len(s.tracks),
                "duration_s": round(dur, 1),
                "programs": ",".join(programs[:8])}
    except Exception as e:
        return {"error": f"unparseable: {type(e).__name__}"}


# ------------------------------ samplers ---------------------------------- #

def _stream_tar_sample(url: str, out_dir: Path, n: int, rng: random.Random,
                       skip_max: int = 500) -> list[dict]:
    """Open a remote .tar.gz as a stream and grab n MIDI members."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "midigenai-explorer"})
    resp = urllib.request.urlopen(req, timeout=60)

    class Counting(io.RawIOBase):
        def __init__(self, raw): self.raw, self.count = raw, 0
        def readable(self): return True
        def readinto(self, b):
            got = self.raw.readinto(b) if hasattr(self.raw, "readinto") else None
            if got is None:
                data = self.raw.read(len(b))
                b[:len(data)] = data
                got = len(data)
            self.count += got
            return got

    counting = Counting(resp)
    tf = tarfile.open(fileobj=io.BufferedReader(counting), mode="r|gz")
    skip = rng.randrange(skip_max)
    out, seen = [], 0
    for member in tf:
        if counting.count > SAMPLE_LIMIT_BYTES:
            break
        if not member.isfile() or not member.name.lower().endswith((".mid", ".midi")):
            continue
        seen += 1
        if seen <= skip:
            continue
        data = tf.extractfile(member).read()
        name = Path(member.name).name
        dest = out_dir / name
        dest.write_bytes(data)
        out.append({"file": name, "meta": {"archive_path": member.name,
                                           "size_kb": round(len(data) / 1024, 1)}})
        if len(out) >= n:
            break
    resp.close()
    return out


def sample_aria(out_dir: Path, n: int, rng: random.Random) -> list[dict]:
    return _stream_tar_sample(
        "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
        "aria-midi-v1-deduped-ext.tar.gz", out_dir, n, rng)


def sample_lakh(out_dir: Path, n: int, rng: random.Random) -> list[dict]:
    return _stream_tar_sample(
        "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
        out_dir, n, rng)


def _sample_gigamidi_parquet(parquet: str, out_dir: Path, n: int,
                             rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from huggingface_hub.utils import GatedRepoError

    try:
        fs = HfFileSystem()
        with fs.open(f"datasets/Metacreation/GigaMIDI/{parquet}", "rb") as f:
            pf = pq.ParquetFile(f)
            rg = rng.randrange(pf.metadata.num_row_groups)
            table = pf.read_row_group(rg)
    except GatedRepoError as e:
        raise RuntimeError(
            "GigaMIDI is gated (auto-approved): visit "
            "https://huggingface.co/datasets/Metacreation/GigaMIDI while "
            "logged in and click 'Agree and access', then retry.") from e

    cols = table.column_names
    bytes_col = next((c for c in cols if table[c].type == "binary"), None)
    idxs = rng.sample(range(table.num_rows), min(n, table.num_rows))
    out = []
    for i in idxs:
        row = {c: table[c][i].as_py() for c in cols}
        midi_bytes = row.pop(bytes_col, None) if bytes_col else None
        meta = {k: (str(v)[:200] if not isinstance(v, (bytes, bytearray)) else
                    f"<{len(v)} bytes>") for k, v in row.items()}
        name = f"{Path(parquet).parent.name or 'gigamidi'}_rg{rg}_{i}.mid"
        if isinstance(midi_bytes, (bytes, bytearray)):
            (out_dir / name).write_bytes(midi_bytes)
            out.append({"file": name, "meta": meta})
        else:
            out.append({"file": None, "meta": meta})
    return out


def sample_local(dir_path: Path):
    def sampler(out_dir: Path, n: int, rng: random.Random) -> list[dict]:
        files = sorted(dir_path.glob("*.mid"))
        picks = rng.sample(files, min(n, len(files)))
        out = []
        for f in picks:
            dest = out_dir / f.name
            dest.write_bytes(f.read_bytes())
            out.append({"file": f.name, "meta": {"source": str(f)}})
        return out
    return sampler


# ------------------------- pipeline views ---------------------------------- #

class PipelineViewer:
    """
    For a sampled MIDI, show what the training pipeline would do with it:
    clean.py's keep/drop verdict, token stats, a decoded round-trip (the exact
    representation the model trains on: time grid, velocity bins, tempo
    stripped), and one concrete augmentation draw (drum-aware pitch shift +
    velocity jitter) — both as playable MIDIs.
    """

    def __init__(self):
        import numpy as np
        from midigenai.data.augment import TokenAugmenter
        from midigenai.tokenizer import build_tokenizer
        self.np = np
        self.tokenizer = build_tokenizer()
        self.augmenter = TokenAugmenter(self.tokenizer)
        self.rng = np.random.default_rng()
        self.inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

    def views(self, midi_path: Path, out_dir: Path, ds_name: str) -> dict:
        from midigenai.data.clean import inspect
        out: dict = {}
        stats, verdict = inspect(midi_path)
        out["clean_verdict"] = "kept" if stats else f"DROPPED: {verdict}"

        try:
            from symusic import Score

            from midigenai.tokenizer import normalize_drums
            score = Score(str(midi_path))
            promoted = normalize_drums(score, midi_path.name)
            if promoted:
                out["drum_fix"] = f"{promoted} track(s) promoted to drums"
            ids = self.tokenizer(score).ids
        except Exception as e:
            out["tokenize_error"] = type(e).__name__
            return out

        n_notes = sum(1 for i in ids
                      if self.inv_vocab.get(i, "").startswith("NoteOn_"))
        out["n_tokens"] = len(ids)
        out["tokens_per_note"] = round(len(ids) / max(n_notes, 1), 2)
        out["token_preview"] = " ".join(
            self.inv_vocab.get(i, f"?{i}") for i in ids[:24])

        stem = midi_path.stem
        try:
            self.tokenizer.decode(list(ids)).dump_midi(
                out_dir / f"{stem}__proc.mid")
            out["processed_url"] = f"/midi/{ds_name}/{stem}__proc.mid"
        except Exception as e:
            out["decode_error"] = type(e).__name__

        try:
            seq = self.np.asarray(ids, dtype=self.np.int64)
            shift = int(self.rng.integers(-6, 7)) or 3
            jitter = int(self.rng.integers(-1, 2))
            aug = seq
            if shift:
                shifted = self.augmenter.pitch_tables[shift][seq]
                aug = self.np.where(
                    self.augmenter._drum_positions(seq), seq, shifted)
            aug = self.augmenter.velocity_tables[jitter][aug]
            self.tokenizer.decode([int(t) for t in aug]).dump_midi(
                out_dir / f"{stem}__aug.mid")
            out["augmented_url"] = f"/midi/{ds_name}/{stem}__aug.mid"
            out["augmentation"] = f"pitch {shift:+d} semitones, velocity {jitter:+d} bin"
        except Exception as e:
            out["augment_error"] = type(e).__name__
        return out


# ------------------------------- app -------------------------------------- #

def build_app(repo_root: Path) -> Flask:
    samples_root = (repo_root / "evals" / "dataset_samples").resolve()
    samples_root.mkdir(parents=True, exist_ok=True)

    samplers = {
        "gigamidi": lambda d, n, r: _sample_gigamidi_parquet(
            "all-instruments-with-drums/validation.parquet", d, n, r),
        "gigamidi-drums": lambda d, n, r: _sample_gigamidi_parquet(
            "drums-only/test.parquet", d, n, r),
        "aria": sample_aria,
        "lakh": sample_lakh,
    }
    prompts_dir = repo_root / "evals" / "prompts"
    if prompts_dir.exists():
        samplers["prompts (local)"] = sample_local(prompts_dir)

    app = Flask(__name__,
                template_folder=str(Path(__file__).parent.parent / "templates"))
    rng = random.Random()
    viewer = PipelineViewer()

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "explore.html")

    @app.route("/api/datasets")
    def datasets():
        return jsonify({"datasets": sorted(samplers)})

    @app.route("/api/sample", methods=["POST"])
    def sample():
        data = request.get_json(force=True)
        ds = data.get("dataset")
        n = min(int(data.get("n", 8)), 25)
        if ds not in samplers:
            return jsonify({"error": f"unknown dataset {ds!r}"}), 400
        ds_dir = samples_root / ds.replace(" ", "_").replace("(", "").replace(")", "")
        ds_dir.mkdir(parents=True, exist_ok=True)
        try:
            items = samplers[ds](ds_dir, n, rng)
        except Exception as e:
            return jsonify({"error": str(e)}), 502
        rows = []
        with (ds_dir / "meta.jsonl").open("a") as f:
            for it in items:
                row = dict(it)
                if it["file"]:
                    row["stats"] = note_stats(ds_dir / it["file"])
                    row["url"] = f"/midi/{ds_dir.name}/{it['file']}"
                    row["pipeline"] = viewer.views(
                        ds_dir / it["file"], ds_dir, ds_dir.name)
                f.write(json.dumps(row) + "\n")
                rows.append(row)
        return jsonify({"dataset": ds, "rows": rows})

    @app.route("/midi/<ds>/<path:name>")
    def serve_midi(ds, name):
        d = (samples_root / ds).resolve()
        full = (d / name).resolve()
        if samples_root not in full.parents or full.suffix.lower() not in (".mid", ".midi"):
            return "forbidden", 403
        return send_from_directory(d, name)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7790)
    args = p.parse_args()
    repo_root = Path(__file__).resolve().parent.parent.parent
    app = build_app(repo_root)
    print(f"[explore] open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
