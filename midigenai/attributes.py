"""
v4 attribute header: control tokens derived from the MIDI itself.

Every v4 training document starts with a short header describing its
contents (instrument families, note density, polyphony, pitch range, source,
optional genre). No labels are needed: everything but Source/Genre is
computed from the notes. At inference the header is how a caller steers the
model ("sparse bass", "drums + piano", "curated-source style") — see
docs/proposals/v4-structure-and-control.md §1b.

Header tokens are registered as MidiTok *special tokens*, so the tokenizer
ignores them on decode and they never collide with musical tokens.

Bucket thresholds are fixed constants chosen from rough corpus quantiles;
they only need to be stable between build and inference.
"""

from __future__ import annotations

from collections import Counter

# GM program families (program // 8) plus a drums pseudo-family.
INSTRUMENT_FAMILIES = [
    "Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings", "Ensemble",
    "Brass", "Reed", "Pipe", "SynthLead", "SynthPad", "SynthFX", "Ethnic",
    "Percussive", "SoundFX",
]
DRUMS = "Drums"
MAX_INST_TOKENS = 6          # keep the header short; largest parts win

# notes per bar (all tracks): sparse .. dense
DENSITY_EDGES = (8, 24, 64)          # -> buckets 0..3
# notes per distinct onset: mono-ish .. thick chords
POLY_EDGES = (1.5, 3.0)              # -> buckets 0..2
# pitch span in semitones
RANGE_EDGES = (24, 48)               # -> buckets 0..2

SOURCES = ["lakh", "lamd", "aria", "gigamidi", "maestro", "pop909",
           "giantmidi", "user"]
GENRES = ["rock", "pop", "jazz", "classical", "electronic", "hiphop", "rnb",
          "country", "folk", "latin", "blues", "metal", "reggae", "soul",
          "world", "other"]

# predicted quality quartile from the human-rated corpus predictor
# (quality_predictor.py); emitted only when a score file is supplied
QUALITY_BUCKETS = 4

HEADER_PREFIXES = ("Inst_", "Density_", "Poly_", "Range_", "Source_", "Genre_",
                   "Quality_")


def header_vocab() -> list[str]:
    """All header token names, in a fixed order (vocab ids follow it)."""
    names = [f"Inst_{f}" for f in INSTRUMENT_FAMILIES] + [f"Inst_{DRUMS}"]
    names += [f"Density_{i}" for i in range(len(DENSITY_EDGES) + 1)]
    names += [f"Poly_{i}" for i in range(len(POLY_EDGES) + 1)]
    names += [f"Range_{i}" for i in range(len(RANGE_EDGES) + 1)]
    names += [f"Source_{s}" for s in SOURCES]
    names += [f"Genre_{g}" for g in GENRES]
    names += [f"Quality_{i}" for i in range(QUALITY_BUCKETS)]
    return names


def is_header_token(name: str) -> bool:
    return name.startswith(HEADER_PREFIXES)


def _bucket(x: float, edges) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def family_of(program: int, is_drum: bool) -> str:
    if is_drum:
        return DRUMS
    return INSTRUMENT_FAMILIES[max(0, min(127, int(program))) // 8]


def instrument_tokens(score) -> list[str]:
    counts: Counter[str] = Counter()
    for t in score.tracks:
        if len(t.notes):
            counts[family_of(t.program, t.is_drum)] += len(t.notes)
    fams = [f for f, _ in counts.most_common(MAX_INST_TOKENS)]
    order = {f: i for i, f in enumerate(INSTRUMENT_FAMILIES + [DRUMS])}
    return [f"Inst_{f}" for f in sorted(fams, key=order.__getitem__)]


def ticks_per_bar(score) -> int:
    """Ticks in the first bar (4/4 when the file carries no time signature)."""
    num, den = 4, 4
    if len(score.time_signatures):
        ts = min(score.time_signatures, key=lambda x: x.time)
        num, den = ts.numerator, ts.denominator
    return int(score.tpq * 4 * num / den)


def content_tokens(score) -> list[str]:
    """Density / Poly / Range tokens for a score (empty score -> [])."""
    notes = [n for t in score.tracks for n in t.notes]
    if not notes:
        return []
    start = min(n.time for n in notes)
    end = max(n.time + n.duration for n in notes)
    n_bars = max(1.0, (end - start) / ticks_per_bar(score))
    density = len(notes) / n_bars
    onsets = len({n.time for n in notes})
    poly = len(notes) / max(1, onsets)
    pitched = [n.pitch for t in score.tracks if not t.is_drum for n in t.notes]
    span = (max(pitched) - min(pitched)) if pitched else 0
    return [
        f"Density_{_bucket(density, DENSITY_EDGES)}",
        f"Poly_{_bucket(poly, POLY_EDGES)}",
        f"Range_{_bucket(span, RANGE_EDGES)}",
    ]


def header_for_score(score, source: str | None = None,
                     genres: list[str] | None = None,
                     quality: int | None = None) -> list[str]:
    """Full header (token names) for a document whose content is `score`."""
    names = instrument_tokens(score) + content_tokens(score)
    if source in SOURCES:
        names.append(f"Source_{source}")
    for g in genres or []:
        if g in GENRES:
            names.append(f"Genre_{g}")
    if quality is not None and 0 <= int(quality) < QUALITY_BUCKETS:
        names.append(f"Quality_{int(quality)}")
    return names


def source_from_path(path: str) -> str | None:
    """Corpus layout is <root>/raw/<source>/...; None when not recognisable."""
    for s in SOURCES:
        if f"/raw/{s}/" in path:
            return s
    return None


def family_token(names: list[str], prefix: str) -> str | None:
    for n in names:
        if n.startswith(prefix):
            return n
    return None
