"""
Near-duplicate MIDI detection: MinHash + LSH over pitch-interval n-grams.

clean.py's SHA1 catches only byte-identical note content; the corpus (Lakh vs
LAMD especially) is full of *near* dups — re-quantized, re-velocitied,
transposed, or lightly edited copies of the same song. Those waste training
compute and, worse, leak across the train/val split (which hashes file paths).

Features are n-grams of the pitch-interval sequence (deltas between
consecutive sorted note pitches), so detection is invariant to transposition
(which we now also apply as augmentation), tempo, quantization, and velocity.

Pipeline position: run on a clean.py manifest, before build_dataset.py:

    python -m midigenai.data.clean --input data/raw --manifest data/manifest.jsonl
    python -m midigenai.data.dedup --manifest data/manifest.jsonl \\
        --out data/manifest_deduped.jsonl --clusters data/dup_clusters.json
    python -m midigenai.data.build_dataset --manifest data/manifest_deduped.jsonl ...

Within a duplicate cluster the file with the most notes is kept (most complete
transcription wins).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

NGRAM = 6
NUM_HASHES = 128
LSH_BANDS = 32               # 32 bands x 4 rows: candidate recall down to J~0.4
ROWS_PER_BAND = NUM_HASHES // LSH_BANDS
# 0.5 verified-Jaccard: catches transposed/re-quantized/re-velocitied copies
# and edits up to ~5% of notes; two different songs sharing half their
# interval 6-grams are, for training purposes, the same material anyway.
JACCARD_THRESHOLD = 0.5
MERSENNE_PRIME = (1 << 61) - 1


def interval_shingles(midi_path: str) -> np.ndarray | None:
    """Hashed n-grams of the pitch-interval sequence, as uint64 set."""
    try:
        from symusic import Score
        score = Score(str(midi_path))
    except Exception:
        return None
    notes = []
    for track in score.tracks:
        if track.is_drum:
            continue  # drum "pitches" are instrument slots, not melody
        for n in track.notes:
            notes.append((n.start, n.pitch))
    if len(notes) < NGRAM + 1:
        return None
    notes.sort()
    pitches = np.array([p for _, p in notes], dtype=np.int64)
    intervals = np.diff(pitches)
    if len(intervals) < NGRAM:
        return None
    # rolling polynomial hash of each n-gram window
    windows = np.lib.stride_tricks.sliding_window_view(intervals + 128, NGRAM)
    coeffs = (131 ** np.arange(NGRAM, dtype=np.uint64)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    hashes = (windows.astype(np.uint64) * coeffs).sum(axis=1, dtype=np.uint64)
    return np.unique(hashes)


class MinHasher:
    def __init__(self, num_hashes: int = NUM_HASHES, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE_PRIME, size=num_hashes, dtype=np.uint64)

    def signature(self, shingles: np.ndarray) -> np.ndarray:
        # (num_hashes, n_shingles) universal hash, min over shingles
        x = shingles.astype(np.uint64)[None, :]
        h = (self.a[:, None] * x + self.b[:, None]) % np.uint64(MERSENNE_PRIME)
        return h.min(axis=1)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def find_duplicate_clusters(sigs: np.ndarray) -> list[list[int]]:
    """LSH-bucket candidates, verify by signature Jaccard, union-find clusters."""
    n = len(sigs)
    uf = UnionFind(n)
    candidates: set[tuple[int, int]] = set()
    for band in range(LSH_BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        rows = sigs[:, band * ROWS_PER_BAND : (band + 1) * ROWS_PER_BAND]
        for i in range(n):
            buckets[rows[i].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 200:  # ignore degenerate mega-buckets
                anchor = members[0]
                for other in members[1:]:
                    candidates.add((anchor, other))
    for i, j in candidates:
        est = (sigs[i] == sigs[j]).mean()
        if est >= JACCARD_THRESHOLD:
            uf.union(i, j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    return [c for c in clusters.values() if len(c) > 1]


def dedup(manifest_path: Path, out_path: Path, clusters_path: Path | None) -> dict:
    entries = [json.loads(line) for line in manifest_path.open() if line.strip()]
    hasher = MinHasher()

    sigs = np.zeros((len(entries), NUM_HASHES), dtype=np.uint64)
    has_sig = np.zeros(len(entries), dtype=bool)
    for i, e in enumerate(tqdm(entries, desc="minhash")):
        sh = interval_shingles(e["path"])
        if sh is not None and len(sh):
            sigs[i] = hasher.signature(sh)
            has_sig[i] = True

    idx_with = np.flatnonzero(has_sig)
    clusters = find_duplicate_clusters(sigs[idx_with])
    clusters = [[int(idx_with[i]) for i in c] for c in clusters]

    drop: set[int] = set()
    for c in clusters:
        keep = max(c, key=lambda i: entries[i].get("n_notes", 0))
        drop.update(i for i in c if i != keep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, e in enumerate(entries):
            if i not in drop:
                f.write(json.dumps(e) + "\n")

    if clusters_path:
        clusters_path.write_text(json.dumps(
            [[entries[i]["path"] for i in c] for c in clusters], indent=2))

    summary = {
        "n_in": len(entries),
        "n_unfingerprintable": int((~has_sig).sum()),
        "n_clusters": len(clusters),
        "n_dropped": len(drop),
        "n_out": len(entries) - len(drop),
        "manifest_out": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="JSONL manifest from clean.py")
    p.add_argument("--out", type=Path, required=True,
                   help="deduped JSONL manifest")
    p.add_argument("--clusters", type=Path, default=None,
                   help="optional JSON dump of duplicate clusters for inspection")
    args = p.parse_args()
    dedup(args.manifest, args.out, args.clusters)


if __name__ == "__main__":
    main()
