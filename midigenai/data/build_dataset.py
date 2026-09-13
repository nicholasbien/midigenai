"""
Tokenize a manifest of cleaned MIDI files into training shards.

Output layout:
    out_dir/
        tokenizer.json           # saved tokenizer (load with load_tokenizer)
        shards/
            train_00000.npy      # uint16 1-D array, BOS-separated docs concatenated
            train_00001.npy
            ...
            val_00000.npy
        manifest.json            # shard names, token counts, split sizes

Why uint16: vocab fits well under 65k. Halves disk + memory vs int32.
Why concatenated docs with BOS separators: simplest to feed a sliding window
during training without per-document padding.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from midigenai.tokenizer import build_tokenizer, save_tokenizer


SHARD_TOKENS = 50_000_000  # 100 MB per shard at uint16
VAL_FRACTION = 0.005       # 0.5% held-out
TRACK_VIEWS = 2            # extra single-track docs per multi-track file
MIN_VIEW_NOTES = 64        # a solo view must have this many notes to count

# Per-worker tokenizer singleton. Each worker process builds one on init —
# config is deterministic so vocab IDs are identical across workers + main.
_TOKENIZER = None
_TRACK_VIEWS = TRACK_VIEWS
_V4 = None


def _worker_init(track_views: int = TRACK_VIEWS, scheme: str = "midilike",
                 v4_opts: dict | None = None):
    global _TOKENIZER, _TRACK_VIEWS, _V4
    _TOKENIZER = build_tokenizer(scheme=scheme)
    _TRACK_VIEWS = track_views
    if scheme.startswith("v4"):
        from midigenai.data.v4_docs import DocBuilder
        _V4 = DocBuilder(_TOKENIZER, track_views=track_views, **(v4_opts or {}))


def _worker_encode_v4(path: str) -> tuple[str, dict | str]:
    """v4: documents already carry BOS/header/EOS (see v4_docs.py). Returns
    a skip-reason string instead of docs when the file is unusable."""
    try:
        return path, _V4.build(path)
    except Exception:
        return path, "error"


def _worker_encode(path: str) -> tuple[str, list[list[int]] | None]:
    """
    Tokenize one file into 1+ documents: the full multi-track mix, plus up to
    _TRACK_VIEWS single-track views for multi-track files. Solo views teach
    the model to continue isolated lines (how users actually jam: one
    instrument at a time) on top of full-arrangement structure; deterministic
    per-path RNG keeps rebuilds reproducible. Mislabeled drum tracks are
    promoted before tokenizing so drum content gets DrumOn/DrumOff tokens.
    """
    try:
        from symusic import Score

        from midigenai.tokenizer import normalize_drums

        def trim_leading(sc):
            """Shift so the first onset is at tick 0. Pure translation: every
            inter-onset interval is unchanged, so beat/groove structure is
            untouched — it only removes wasted leading Rest tokens."""
            starts = [n.start for t in sc.tracks for n in t.notes]
            return sc.shift_time(-min(starts)) if starts and min(starts) > 0 else sc

        score = Score(path)
        normalize_drums(score, Path(path).name)
        docs = [_TOKENIZER(trim_leading(score)).ids]
        if _TRACK_VIEWS > 0:
            candidates = [i for i, t in enumerate(score.tracks)
                          if len(t.notes) >= MIN_VIEW_NOTES]
            if len(candidates) >= 2:
                rng = random.Random(path)
                rng.shuffle(candidates)
                for i in candidates[:_TRACK_VIEWS]:
                    solo = Score(path)  # fresh copy; cheap relative to tokenize
                    normalize_drums(solo, Path(path).name)
                    solo.tracks = [solo.tracks[i]]
                    # solo tracks often enter mid-song: trim their lead-in too
                    ids = _TOKENIZER(trim_leading(solo)).ids
                    if len(ids) >= 8:
                        docs.append(ids)
        return path, docs
    except Exception:
        return path, None


@dataclass
class ShardWriter:
    out_dir: Path
    prefix: str
    shard_tokens: int
    _buf: list[np.ndarray]
    _buf_len: int
    _shard_idx: int

    @classmethod
    def create(cls, out_dir: Path, prefix: str, shard_tokens: int) -> "ShardWriter":
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(out_dir, prefix, shard_tokens, [], 0, 0)

    def append(self, ids: np.ndarray) -> None:
        self._buf.append(ids)
        self._buf_len += len(ids)
        while self._buf_len >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def close(self) -> int:
        if self._buf_len > 0:
            self._flush(self._buf_len)
        return self._shard_idx

    def _flush(self, take: int) -> None:
        cat = np.concatenate(self._buf)
        out, rest = cat[:take], cat[take:]
        path = self.out_dir / f"{self.prefix}_{self._shard_idx:05d}.npy"
        np.save(path, out)
        self._shard_idx += 1
        self._buf = [rest] if len(rest) > 0 else []
        self._buf_len = len(rest)


def split_by_path(path: str, val_fraction: float) -> str:
    """Deterministic train/val split by hashing the file path."""
    h = int(hashlib.sha1(path.encode()).hexdigest(), 16)
    return "val" if (h % 1_000_000) / 1_000_000 < val_fraction else "train"


def build(
    manifest_path: Path,
    out_dir: Path,
    shard_tokens: int = SHARD_TOKENS,
    val_fraction: float = VAL_FRACTION,
    limit: int | None = None,
    workers: int | None = None,
    track_views: int = TRACK_VIEWS,
    tag: str = "",
    fragment_under_seconds: float | None = None,
    scheme: str = "midilike",
    v4_opts: dict | None = None,
) -> dict:
    """
    `fragment_under_seconds`: source files shorter than this get BOS but **no
    EOS** — they are treated as fragments of a longer piece rather than pieces
    that end. Motivation (2026-09-01): GigaMIDI is mostly loops/clips (median
    27 s, 51% under 30 s, ~24% of all training docs), and every one of them
    teaches "8 bars, then EOS". The ctx4096_ext model assigns P(EOS) > 0.2 to
    ~35% of random 8-bar Lakh excerpts as a result. Dropping the files instead
    would lose the corpus's main drums/multi-track source, so keep the tokens
    and drop only the ending signal. Off by default (None) so existing
    corpora rebuild byte-identically.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(scheme=scheme)
    save_tokenizer(tokenizer, out_dir / "tokenizer.json")

    bos_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer.vocab else tokenizer.vocab.get("BOS", 1)
    # EOS at every doc end: without it the model never sees a piece end and
    # generation can only stop at the token cap ("gets too long" failure mode)
    eos_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer.vocab else tokenizer.vocab.get("EOS", 2)

    shards_dir = out_dir / "shards"
    # a tag names the source in the shard files (train_<tag>_00000.npy) so
    # training can apply per-source mixture weights
    train_prefix = f"train_{tag}" if tag else "train"
    val_prefix = f"val_{tag}" if tag else "val"
    train_writer = ShardWriter.create(shards_dir, train_prefix, shard_tokens)
    val_writer = ShardWriter.create(shards_dir, val_prefix, shard_tokens)
    # quality buckets (v4, --quality): train shards are split per bucket,
    # train_<tag>_q<k>_NNNNN.npy, so --mixture "q0:0.25,q3:1.5" can weight
    # them; val stays in one shard per source
    quality = (v4_opts or {}).get("quality") or {}
    bucket_writers: dict[int, ShardWriter] = {}

    def writer_for(split: str, path: str) -> ShardWriter:
        if split == "val":
            return val_writer
        q = quality.get(path)
        if q is None:
            return train_writer
        if q not in bucket_writers:
            bucket_writers[q] = ShardWriter.create(
                shards_dir, f"{train_prefix}_q{q}", shard_tokens)
        return bucket_writers[q]

    n_files = 0
    n_failed = 0
    n_train_tokens = 0
    n_val_tokens = 0

    with manifest_path.open() as f:
        entries = [json.loads(line) for line in f]
    if limit:
        entries = entries[:limit]
    random.Random(0).shuffle(entries)
    paths = [e["path"] for e in entries]
    duration_by_path = {e["path"]: e.get("duration_seconds") for e in entries}

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[tokenize] {len(paths)} files, {n_workers} workers")

    n_view_docs = 0
    n_view_tokens = 0
    n_fragment_docs = 0
    # v4: token share per document kind, to tune the 60/25/15 target mix
    kind_tokens = {"continuation": 0, "accompaniment": 0, "infill": 0}
    kind_docs = {"continuation": 0, "accompaniment": 0, "infill": 0}
    skip_reasons: dict[str, int] = {"timesig": 0, "empty": 0, "error": 0}
    is_v4_scheme = scheme.startswith("v4")
    encode = _worker_encode_v4 if is_v4_scheme else _worker_encode
    with Pool(n_workers, initializer=_worker_init,
              initargs=(track_views, scheme, v4_opts)) as pool:
        for path, docs in tqdm(
            pool.imap_unordered(encode, paths, chunksize=16),
            total=len(paths), desc="tokenizing",
        ):
            if docs is None or isinstance(docs, str) or (not is_v4_scheme and len(docs[0]) < 8):
                n_failed += 1
                skip_reasons[docs if isinstance(docs, str) else "error"] += 1
                continue
            # solo views share the parent's path-hash split, so a song can
            # never straddle train and val through its views
            split = split_by_path(path, val_fraction)
            writer = writer_for(split, path)
            # solo views inherit the parent's fragment status: a 20 s loop's
            # bass line is no more "a piece that ends" than the loop itself
            dur = duration_by_path.get(path)
            is_fragment = (
                fragment_under_seconds is not None
                and dur is not None
                and dur < fragment_under_seconds
            )
            tail = [] if is_fragment else [eos_id]
            if is_v4_scheme:
                # v4 docs already carry BOS/header/EOS. The fragment rule
                # applies to continuation docs only: an accompaniment or
                # infill target's EOS means "segment complete", not "piece
                # ends", so it always stays.
                for kind, kdocs in docs.items():
                    for ids in kdocs:
                        if is_fragment and kind == "continuation" and ids[-1] == eos_id:
                            ids = ids[:-1]
                            n_fragment_docs += 1
                        arr = np.asarray(ids, dtype=np.uint16)
                        writer.append(arr)
                        if split == "val":
                            n_val_tokens += len(arr)
                        else:
                            n_train_tokens += len(arr)
                        kind_tokens[kind] += len(arr)
                        kind_docs[kind] += 1
                n_files += 1
                continue
            for d, ids in enumerate(docs):
                arr = np.asarray([bos_id, *ids, *tail], dtype=np.uint16)
                if is_fragment:
                    n_fragment_docs += 1
                writer.append(arr)
                if split == "val":
                    n_val_tokens += len(arr)
                else:
                    n_train_tokens += len(arr)
                if d > 0:
                    n_view_docs += 1
                    n_view_tokens += len(arr)
            n_files += 1

    n_train_shards = train_writer.close() + sum(w.close() for w in bucket_writers.values())
    n_val_shards = val_writer.close()

    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "vocab_size": len(tokenizer),
        "n_files_kept": n_files,
        "n_files_failed": n_failed,
        "skip_reasons": skip_reasons,
        "n_train_tokens": n_train_tokens,
        "n_val_tokens": n_val_tokens,
        "fragment_under_seconds": fragment_under_seconds,
        "n_fragment_docs": n_fragment_docs,
        "n_train_shards": n_train_shards,
        "n_val_shards": n_val_shards,
        "shard_tokens": shard_tokens,
        "track_views_per_file": track_views,
        "n_view_docs": n_view_docs,
        "n_view_tokens": n_view_tokens,
        "scheme": scheme,
        "v4_opts": {k: v for k, v in (v4_opts or {}).items() if k not in ("genres", "quality")},
        "n_quality_scored": len(quality),
        "kind_docs": kind_docs,
        "kind_tokens": kind_tokens,
        "kind_share": {k: round(v / max(1, sum(kind_tokens.values())), 3)
                       for k, v in kind_tokens.items()},
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSONL manifest from clean.py")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for shards + tokenizer")
    parser.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional cap on number of files (for pilot runs)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel tokenizer workers (default: cpu_count-1)")
    parser.add_argument("--track-views", type=int, default=TRACK_VIEWS,
                        help="extra single-track docs per multi-track file (0 disables)")
    parser.add_argument("--fragment-under-seconds", type=float, default=None,
                        help="files shorter than this get no EOS (treated as "
                             "fragments, not endings); see build_dataset docstring")
    parser.add_argument("--tag", default="",
                        help="source tag baked into shard names for mixture weighting")
    parser.add_argument("--scheme", choices=["midilike", "v4", "v4-24"], default="v4",
                        help="tokenizer scheme: v4 (REMI + header + accompaniment/"
                             "infill docs) or midilike (v2/v3 legacy)")
    parser.add_argument("--accomp-windows", type=int, default=6,
                        help="v4: accompaniment docs per multi-track file")
    parser.add_argument("--infill-windows", type=int, default=2,
                        help="v4: span-infill docs per file")
    parser.add_argument("--window-bars", type=int, default=16,
                        help="v4: bars per accompaniment window")
    parser.add_argument("--context-bars", type=int, default=16,
                        help="v4: bars of context per infill doc")
    parser.add_argument("--max-span-bars", type=int, default=4,
                        help="v4: longest infilled span")
    parser.add_argument("--genres", type=Path, default=None,
                        help="v4: optional JSON {path: [genre,...]} for Genre_ tokens")
    parser.add_argument("--quality", type=Path, default=None,
                        help="v4: quality_predictor score JSONL (path, q_bucket): adds "
                             "Quality_ header tokens and splits train shards per bucket")
    args = parser.parse_args()
    v4_opts = dict(accomp_windows=args.accomp_windows, infill_windows=args.infill_windows,
                   window_bars=args.window_bars, context_bars=args.context_bars,
                   max_span_bars=args.max_span_bars)
    if args.genres:
        v4_opts["genres"] = json.loads(args.genres.read_text())
    if args.quality:
        v4_opts["quality"] = {}
        with args.quality.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    v4_opts["quality"][row["path"]] = int(row["q_bucket"])
    build(
        manifest_path=args.manifest,
        out_dir=args.out,
        shard_tokens=args.shard_tokens,
        val_fraction=args.val_fraction,
        limit=args.limit,
        workers=args.workers,
        track_views=args.track_views,
        tag=args.tag,
        fragment_under_seconds=args.fragment_under_seconds,
        scheme=args.scheme,
        v4_opts=v4_opts,
    )
