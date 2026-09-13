"""
midigenai tokenizer: thin wrapper around MidiTok.

Two schemes live here:

- **v2/v3 (MIDILike, 641 tokens)**: NoteOn/NoteOff/TimeShift/Rest. Time is
  purely relative; there is no bar or beat-position token.
- **v4 (REMI + attribute header)**: Bar / TimeSig / Position tokens make the
  downbeat explicit, Duration replaces NoteOff, and every document starts with
  a header of control tokens (see `attributes.py`). `MASK` + `SEP` support
  accompaniment and span-infill documents (`sequence_format.py`). Rests are
  off so an empty bar is literally a `Bar` token, which keeps bar counting
  exact for the jam's stop condition.

v2/v3 choices (kept for the legacy scheme):
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

import json

from miditok import MIDILike, REMI, TokenizerConfig


SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "SEP"]
V4_SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "SEP", "MASK"]
# bars the v4 scheme can express; anything else is dropped at build time
# (MidiTok would silently re-bar a 5/4 file as 4/4)
V4_TIME_SIGNATURES = {4: [2, 3, 4], 8: [6]}
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


def v4_config() -> TokenizerConfig:
    from .attributes import header_vocab
    return TokenizerConfig(
        pitch_range=(0, 127),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        # header tokens ride along as specials: ignored on decode, no
        # collision with musical tokens
        special_tokens=V4_SPECIAL_TOKENS + header_vocab(),
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=True,
        time_signature_range=V4_TIME_SIGNATURES,
        use_programs=True,
        one_token_stream_for_programs=True,
        program_changes=True,
    )


def build_tokenizer(config: TokenizerConfig | None = None,
                    scheme: str = "midilike"):
    """`scheme`: "midilike" (v2/v3 checkpoints) or "v4" (REMI + header)."""
    if config is not None:
        return MIDILike(config)
    if scheme == "v4":
        return REMI(v4_config())
    if scheme == "midilike":
        return MIDILike(default_config())
    raise ValueError(f"unknown tokenizer scheme {scheme!r}")


def load_tokenizer(path: PathLike):
    """Load a saved tokenizer of either scheme (the json names its class)."""
    path = Path(path)
    kind = json.loads(path.read_text()).get("tokenization", "MIDILike")
    cls = {"MIDILike": MIDILike, "REMI": REMI}.get(kind)
    if cls is None:
        raise ValueError(f"unsupported tokenizer class {kind!r} in {path}")
    return cls(params=path)


def is_v4(tokenizer) -> bool:
    return "MASK_None" in tokenizer.vocab and "Bar_None" in tokenizer.vocab


def special_id(tokenizer, name: str) -> int | None:
    """Id of a special token by bare name ("BOS", "MASK", "Inst_Piano")."""
    for cand in (f"{name}_None", name):
        if cand in tokenizer.vocab:
            return tokenizer.vocab[cand]
    return None


def supported_time_signatures(score) -> bool:
    """True when every time signature in `score` is expressible in v4."""
    for ts in score.time_signatures:
        if ts.numerator not in V4_TIME_SIGNATURES.get(ts.denominator, ()):
            return False
    return True


def save_tokenizer(tokenizer, path: PathLike) -> None:
    tokenizer.save(Path(path))


def encode_midi(tokenizer, midi_path: PathLike) -> list[int]:
    return tokenizer(Path(midi_path)).ids


def decode_to_midi(tokenizer, ids: Sequence[int], out_path: PathLike) -> None:
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
