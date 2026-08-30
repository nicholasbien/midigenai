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

from midigenai.tokenizer import build_tokenizer, encode_midi, save_tokenizer


SHARD_TOKENS = 50_000_000  # 100 MB per shard at uint16
VAL_FRACTION = 0.005       # 0.5% held-out
TRACK_VIEWS = 2            # extra single-track docs per multi-track file
MIN_VIEW_NOTES = 64        # a solo view must have this many notes to count

# Per-worker tokenizer singleton. Each worker process builds one on init —
# config is deterministic so vocab IDs are identical across workers + main.
_TOKENIZER = None
_TRACK_VIEWS = TRACK_VIEWS


def _worker_init(track_views: int = TRACK_VIEWS):
    global _TOKENIZER, _TRACK_VIEWS
    _TOKENIZER = build_tokenizer()
    _TRACK_VIEWS = track_views


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

        score = Score(path)
        normalize_drums(score, Path(path).name)
        docs = [_TOKENIZER(score).ids]
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
                    ids = _TOKENIZER(solo).ids
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
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    save_tokenizer(tokenizer, out_dir / "tokenizer.json")

    bos_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer.vocab else tokenizer.vocab.get("BOS", 1)
    # EOS at every doc end: without it the model never sees a piece end and
    # generation can only stop at the token cap ("gets too long" failure mode)
    eos_id = tokenizer["EOS_None"] if "EOS_None" in tokenizer.vocab else tokenizer.vocab.get("EOS", 2)

    shards_dir = out_dir / "shards"
    train_writer = ShardWriter.create(shards_dir, "train", shard_tokens)
    val_writer = ShardWriter.create(shards_dir, "val", shard_tokens)

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

    n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[tokenize] {len(paths)} files, {n_workers} workers")

    n_view_docs = 0
    n_view_tokens = 0
    with Pool(n_workers, initializer=_worker_init,
              initargs=(track_views,)) as pool:
        for path, docs in tqdm(
            pool.imap_unordered(_worker_encode, paths, chunksize=16),
            total=len(paths), desc="tokenizing",
        ):
            if docs is None or len(docs[0]) < 8:
                n_failed += 1
                continue
            # solo views share the parent's path-hash split, so a song can
            # never straddle train and val through its views
            split = split_by_path(path, val_fraction)
            writer = val_writer if split == "val" else train_writer
            for d, ids in enumerate(docs):
                arr = np.asarray([bos_id, *ids, eos_id], dtype=np.uint16)
                writer.append(arr)
                if split == "val":
                    n_val_tokens += len(arr)
                else:
                    n_train_tokens += len(arr)
                if d > 0:
                    n_view_docs += 1
                    n_view_tokens += len(arr)
            n_files += 1

    n_train_shards = train_writer.close()
    n_val_shards = val_writer.close()

    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "vocab_size": len(tokenizer),
        "n_files_kept": n_files,
        "n_files_failed": n_failed,
        "n_train_tokens": n_train_tokens,
        "n_val_tokens": n_val_tokens,
        "n_train_shards": n_train_shards,
        "n_val_shards": n_val_shards,
        "shard_tokens": shard_tokens,
        "track_views_per_file": track_views,
        "n_view_docs": n_view_docs,
        "n_view_tokens": n_view_tokens,
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
    args = parser.parse_args()
    build(
        manifest_path=args.manifest,
        out_dir=args.out,
        shard_tokens=args.shard_tokens,
        val_fraction=args.val_fraction,
        limit=args.limit,
        workers=args.workers,
        track_views=args.track_views,
    )
