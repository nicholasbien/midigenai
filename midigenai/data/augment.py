"""
Token-level augmentation for MIDI training data.

Augmentations operate on already-tokenized sequences so they're cheap to apply
on the fly during training (or precomputed into shards if we have disk to spare):

- pitch_shift: shift all NoteOn/NoteOff tokens by ±n semitones
- velocity_jitter: nudge Velocity tokens by ±k bins
- (tempo stretch is handled at the symusic level before tokenization, not here)
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from miditok import MIDILike

_NOT_A_PROGRAM = np.iinfo(np.int64).min


class TokenAugmenter:
    """
    Vectorized on-the-fly augmentation over token-id arrays, for use in the
    training data path. Precomputes one vocab-sized remap table per pitch shift
    and per velocity jitter, so augmenting a sequence is fancy indexing.

    Drum handling: NoteOn/NoteOff tokens are shared across instruments, so we
    reconstruct the active program at each position (forward-fill from Program_*
    tokens) and leave positions under Program_-1 (drums) unshifted. A window
    that starts mid-drum-track before any Program token is assumed non-drum —
    same assumption the decoder makes.
    """

    def __init__(self, tokenizer: MIDILike, max_pitch_shift: int = 6,
                 max_velocity_jitter: int = 1):
        vocab = tokenizer.vocab
        vocab_size = len(vocab)
        self.max_pitch_shift = max_pitch_shift
        self.max_velocity_jitter = max_velocity_jitter

        self.pitch_tables: dict[int, np.ndarray] = {}
        # clip_tables[s]: note tokens that CANNOT shift by s (target out of
        # vocab). Shifting a window that contains one would transpose only
        # part of a chord — corruption, not augmentation — so the caller
        # falls back toward smaller shifts until none clip.
        self.clip_tables: dict[int, np.ndarray] = {}
        for s in range(-max_pitch_shift, max_pitch_shift + 1):
            table = np.arange(vocab_size, dtype=np.int64)
            clips = np.zeros(vocab_size, dtype=bool)
            for name, tid in vocab.items():
                for prefix in ("NoteOn_", "NoteOff_"):
                    if name.startswith(prefix):
                        shifted = f"{prefix}{int(name[len(prefix):]) + s}"
                        if shifted in vocab:
                            table[tid] = vocab[shifted]
                        else:
                            clips[tid] = s != 0
            self.pitch_tables[s] = table
            self.clip_tables[s] = clips

        # Velocity bins ordered by value; jitter moves to an adjacent bin,
        # clamped at the ends.
        vel_bins = sorted(
            (int(name.split("_")[1]), tid)
            for name, tid in vocab.items() if name.startswith("Velocity_")
        )
        self.velocity_tables: dict[int, np.ndarray] = {}
        for d in range(-max_velocity_jitter, max_velocity_jitter + 1):
            table = np.arange(vocab_size, dtype=np.int64)
            for i, (_, tid) in enumerate(vel_bins):
                j = min(max(i + d, 0), len(vel_bins) - 1)
                table[tid] = vel_bins[j][1]
            self.velocity_tables[d] = table

        # Program value per token id (sentinel where the token isn't Program_*).
        self.program_value = np.full(vocab_size, _NOT_A_PROGRAM, dtype=np.int64)
        for name, tid in vocab.items():
            if name.startswith("Program_"):
                try:
                    self.program_value[tid] = int(name.split("_")[1])
                except ValueError:
                    pass

    def _drum_positions(self, seq: np.ndarray) -> np.ndarray:
        prog = self.program_value[seq]
        is_prog = prog != _NOT_A_PROGRAM
        last_idx = np.where(is_prog, np.arange(len(seq)), -1)
        last_idx = np.maximum.accumulate(last_idx)
        filled = np.where(last_idx >= 0, prog[np.clip(last_idx, 0, None)], 0)
        return filled == -1

    def __call__(self, seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augment one int64 token sequence (a full training window)."""
        shift = int(rng.integers(-self.max_pitch_shift, self.max_pitch_shift + 1))             if self.max_pitch_shift > 0 else 0
        jitter = int(rng.integers(-self.max_velocity_jitter,
                                  self.max_velocity_jitter + 1))             if self.max_velocity_jitter > 0 else 0
        # shrink the shift toward 0 until no note in the window would clip,
        # so a chord is always transposed whole or not at all
        while shift != 0 and self.clip_tables[shift][seq].any():
            shift -= 1 if shift > 0 else -1
        out = seq
        if shift != 0:
            shifted = self.pitch_tables[shift][seq]
            out = np.where(self._drum_positions(seq), seq, shifted)
        if jitter != 0:
            out = self.velocity_tables[jitter][out]
        return out


def pitch_shift(
    tokens: Sequence[int], tokenizer: MIDILike, semitones: int
) -> list[int]:
    """Shift all pitched note tokens by `semitones`. Drum tokens are left alone."""
    if semitones == 0:
        return list(tokens)

    vocab = tokenizer.vocab  # str -> id
    inv = {v: k for k, v in vocab.items()}

    def shift(tok_id: int) -> int:
        name = inv.get(tok_id)
        if name is None:
            return tok_id
        for prefix in ("NoteOn_", "NoteOff_"):
            if name.startswith(prefix):
                pitch = int(name[len(prefix):])
                new_pitch = pitch + semitones
                if 0 <= new_pitch <= 127:
                    new_name = f"{prefix}{new_pitch}"
                    return vocab.get(new_name, tok_id)
                return tok_id  # out of range -> leave unchanged
        return tok_id

    return [shift(t) for t in tokens]


def velocity_jitter(
    tokens: Sequence[int], tokenizer: MIDILike, max_bins: int, rng: random.Random
) -> list[int]:
    """Nudge each Velocity token by a uniform random ±max_bins."""
    if max_bins <= 0:
        return list(tokens)

    vocab = tokenizer.vocab
    inv = {v: k for k, v in vocab.items()}

    velocity_ids: dict[int, int] = {}
    for name, tid in vocab.items():
        if name.startswith("Velocity_"):
            try:
                velocity_ids[int(name[len("Velocity_"):])] = tid
            except ValueError:
                pass
    if not velocity_ids:
        return list(tokens)
    sorted_vels = sorted(velocity_ids)

    def jitter(tok_id: int) -> int:
        name = inv.get(tok_id)
        if name is None or not name.startswith("Velocity_"):
            return tok_id
        try:
            v = int(name[len("Velocity_"):])
        except ValueError:
            return tok_id
        idx = sorted_vels.index(v) if v in sorted_vels else -1
        if idx < 0:
            return tok_id
        new_idx = max(0, min(len(sorted_vels) - 1, idx + rng.randint(-max_bins, max_bins)))
        return velocity_ids[sorted_vels[new_idx]]

    return [jitter(t) for t in tokens]


def random_augment(
    tokens: Sequence[int],
    tokenizer: MIDILike,
    pitch_range: int = 6,
    velocity_bins: int = 2,
    rng: random.Random | None = None,
) -> list[int]:
    rng = rng or random.Random()
    semitones = rng.randint(-pitch_range, pitch_range)
    out = pitch_shift(tokens, tokenizer, semitones)
    out = velocity_jitter(out, tokenizer, velocity_bins, rng)
    return out
