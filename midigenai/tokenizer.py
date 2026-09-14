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
# (MidiTok would silently re-bar an unknown meter as 4/4). The long tail
# (1/4 pickups, cut time, 12/8...) covers ~80% of the files a 4/4-3/4-6/8
# set would skip, for ~20 extra TimeSig/Position tokens.
V4_TIME_SIGNATURES = {2: [2, 3, 4], 4: [1, 2, 3, 4, 5, 6, 7],
                      8: [3, 5, 6, 7, 9, 12]}
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


def v4_config(res: int = 8) -> TokenizerConfig:
    """`res`: positions per beat. 8 = 32nd-note grid (v2/v3 parity);
    24 = 16ths + triplets + 32nds (pilot arm C; the likely v4 default)."""
    from .attributes import header_vocab
    return TokenizerConfig(
        pitch_range=(0, 127),
        beat_res={(0, 4): res, (4, 12): max(4, res // 2)},
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
    if scheme == "v4-24":
        # 16ths + triplets + 32nds; performed sources sit within ~5 ms of
        # this grid vs ~15-23 ms at 8/beat (measured 2026-09-13)
        return REMI(v4_config(res=24))
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

# General MIDI percussion key range, and the three voices that make a kit
# recognisable. Requiring one from each family is what separates a kit from
# a bass ostinato that happens to sit in the same pitch range.
GM_DRUM_RANGE = range(35, 60)
GM_KICK = (35, 36)
GM_SNARE = (38, 40)
GM_HAT = (42, 44, 46)


def looks_like_drums(track, min_notes: int = 16) -> bool:
    """Content-based drum detection for tracks with no naming hint.

    Deliberately strict — promoting a pitched track to drums destroys its
    melody — so all of: enough notes, a tiny pitch alphabet, nearly every
    note inside the GM percussion range, and a kick AND a snare AND a hat.
    A four-note bass ostinato sitting in the percussion range fails the last
    test, which two-of-four core pitches would have let through.
    """
    pitches = [n.pitch for n in track.notes]
    if len(pitches) < min_notes:
        return False
    distinct = set(pitches)
    if len(distinct) > 8:
        return False
    in_range = sum(1 for p in pitches if p in GM_DRUM_RANGE) / len(pitches)
    if in_range < 0.85:
        return False
    return all(any(p in distinct for p in fam)
               for fam in (GM_KICK, GM_SNARE, GM_HAT))


def normalize_drums(score, filename_hint: str = "") -> int:
    """
    Promote mislabeled drum tracks to is_drum so they tokenize as
    DrumOn/DrumOff events instead of pitched piano notes.

    Many real files (Ableton exports, v1-era generations) carry drum content
    on a normal channel with program 0 — the model would learn drum rhythms
    as piano. Conservative heuristic: promote on a drum keyword in the track
    name, or in the filename when the file has a single melodic reading of it
    (drums split across many named tracks are matched per-track anyway), or
    on drum-shaped content (`looks_like_drums`) for tracks whose name says
    nothing — a real pattern in Lakh/LAMD, and the main source of "this is
    obviously a kit but it plays as piano" prompts.
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
                (fname_says_drums and len(score.tracks) == 1) or \
                looks_like_drums(t):
            t.is_drum = True
            changed += 1
    return changed
