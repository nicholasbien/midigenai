"""
midigenai tokenizer: thin wrapper around MidiTok's MIDILike (event-based) scheme.

Choices:
- MIDILike over REMI: preserves microtiming for live jamming where user input
  isn't on a bar grid.
- pitch_range covers full MIDI (0, 128) so the same tokenizer handles piano,
  bass, drums, etc.
- use_programs=True with one_token_stream_for_programs=True interleaves
  instrument changes into a single autoregressive stream — keeps the model
  simple while supporting multi-track.
- 32 velocity bins: imperceptible vs MIDI's 128, much smaller vocab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from miditok import MIDILike, TokenizerConfig


SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "SEP"]
PathLike = Union[str, Path]


def default_config() -> TokenizerConfig:
    return TokenizerConfig(
        pitch_range=(0, 127),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        special_tokens=SPECIAL_TOKENS,
        use_chords=False,
        use_rests=True,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=True,
        one_token_stream_for_programs=True,
        program_changes=True,
    )


def build_tokenizer(config: TokenizerConfig | None = None) -> MIDILike:
    return MIDILike(config or default_config())


def load_tokenizer(path: PathLike) -> MIDILike:
    return MIDILike(params=Path(path))


def save_tokenizer(tokenizer: MIDILike, path: PathLike) -> None:
    tokenizer.save(Path(path))


def encode_midi(tokenizer: MIDILike, midi_path: PathLike) -> list[int]:
    return tokenizer(Path(midi_path)).ids


def decode_to_midi(
    tokenizer: MIDILike, ids: Sequence[int], out_path: PathLike
) -> None:
    score = tokenizer.decode(list(ids))
    score.dump_midi(Path(out_path))


def roundtrip(midi_path: PathLike, out_path: PathLike) -> tuple[int, list[int]]:
    """Encode then decode a MIDI file. Returns (n_tokens, token_ids)."""
    tok = build_tokenizer()
    ids = encode_midi(tok, midi_path)
    decode_to_midi(tok, ids, out_path)
    return len(ids), ids


DRUM_NAME_HINTS = ("drum", "drm", "perc", "kit", "808", "909", "kick", "snare",
                   "hat", "cymbal", "tom", "clap", "batter", "schlag", "beat")


def normalize_drums(score, filename_hint: str = "") -> int:
    """
    Promote mislabeled drum tracks to is_drum so they tokenize as
    DrumOn/DrumOff events instead of pitched piano notes.

    Many real files (Ableton exports, v1-era generations) carry drum content
    on a normal channel with program 0 — the model would learn drum rhythms
    as piano. Conservative heuristic: promote on a drum keyword in the track
    name, or in the filename when the file has a single melodic reading of it
    (drums split across many named tracks are matched per-track anyway).
    Returns the number of tracks promoted. Mutates `score` in place.
    """
    fname = filename_hint.lower()
    fname_says_drums = any(k in fname for k in DRUM_NAME_HINTS)
    changed = 0
    for t in score.tracks:
        if t.is_drum:
            continue
        name = (t.name or "").lower()
        if any(k in name for k in DRUM_NAME_HINTS) or \
                (fname_says_drums and len(score.tracks) == 1):
            t.is_drum = True
            changed += 1
    return changed
