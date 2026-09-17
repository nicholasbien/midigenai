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

# task tokens open accompaniment / infill documents (right after BOS) so a
# plain continuation prompt can never be mistaken for a condition: pilot E'
# (2026-09-13) emitted SEP/MASK/BOS in 40% of continuations without them
TASKS = ["accomp", "infill"]

HEADER_PREFIXES = ("Task_", "Inst_", "Density_", "Poly_", "Range_", "Source_",
                   "Genre_", "Quality_")
NEVER_DROP = ("Task_",)      # header dropout must leave the task token alone


def header_vocab() -> list[str]:
    """All header token names, in a fixed order (vocab ids follow it)."""
    names = [f"Task_{t}" for t in TASKS]
    names += [f"Inst_{f}" for f in INSTRUMENT_FAMILIES] + [f"Inst_{DRUMS}"]
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


# ---------- picking an instrument at inference ---------- #
#
# The header's Inst_ tokens name what the finished document contains, so for
# accompaniment "add a bass" is written as the condition's family plus
# Inst_Bass (see data/v4_docs.py, where accompaniment headers are built from
# condition + target). Callers speak in instrument names, not GM families, so
# map the common ones here rather than making every caller know that an
# electric piano is a Piano and a sax is a Reed.
INSTRUMENT_ALIASES = {
    "acoustic piano": "Piano",
    "electric piano": "Piano",
    "keys": "Piano",
    "keyboard": "Piano",
    "rhodes": "Piano",
    "chromatic percussion": "ChromPerc",
    "vibraphone": "ChromPerc",
    "electric guitar": "Guitar",
    "acoustic guitar": "Guitar",
    "electric bass": "Bass",
    "bassline": "Bass",
    "sub": "Bass",
    "808": "Bass",
    "string section": "Strings",
    "violin": "Strings",
    "cello": "Strings",
    "choir": "Ensemble",
    "horns": "Brass",
    "trumpet": "Brass",
    "sax": "Reed",
    "saxophone": "Reed",
    "clarinet": "Reed",
    "flute": "Pipe",
    "lead": "SynthLead",
    "synth": "SynthLead",
    "pad": "SynthPad",
    "synth bass": "Bass",
    "fx": "SynthFX",
    "sound effects": "SoundFX",
    "percussion": "Percussive",
    "drum": "Drums",
    "drum kit": "Drums",
    "beat": "Drums",
}

# Values meaning "no preference, let the model choose" rather than a family.
# The site's playback picker sends "original", so accept that too: an unknown
# name is an error, and a deliberate "don't care" should not look like one.
AUTO_INSTRUMENT = {"", "auto", "any", "none", "original", "model"}


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_FAMILY_BY_NAME = {_norm(f): f for f in INSTRUMENT_FAMILIES + [DRUMS]}
_FAMILY_BY_NAME.update({_norm(k): v for k, v in INSTRUMENT_ALIASES.items()})


def resolve_family(name: str | None) -> str | None:
    """GM family for an instrument name, or None if it isn't one we know.

    Case and separators are ignored, so "SynthLead", "synth lead" and
    "synth_lead" all land on the same family. Returns None for both unknown
    names and the AUTO_INSTRUMENT values, so callers that need to tell those
    apart should check `is_auto_instrument` first.
    """
    if not name:
        return None
    return _FAMILY_BY_NAME.get(_norm(name))


def is_auto_instrument(name: str | None) -> bool:
    """True when the caller asked for no particular instrument."""
    return (name or "").strip().lower() in AUTO_INSTRUMENT


def sort_header(names: list[str]) -> list[str]:
    """Canonical family order, as the dataset builder writes it. Stable, so
    the order within a family (e.g. several Inst_ tokens) is preserved."""
    rank = {p: i for i, p in enumerate(HEADER_PREFIXES)}
    return sorted(
        names,
        key=lambda n: rank[next(p for p in HEADER_PREFIXES if n.startswith(p))])


def with_instruments(names: list[str], families: list[str]) -> list[str]:
    """`names` with its Inst_ tokens replaced by exactly `families`.

    Duplicates are dropped and the result is capped at MAX_INST_TOKENS, the
    same ceiling the builder applies, so a header written here stays in the
    shape the model was trained on.
    """
    order = {f: i for i, f in enumerate(INSTRUMENT_FAMILIES + [DRUMS])}
    unknown = [f for f in families if f not in order]
    if unknown:
        raise ValueError(f"not instrument families: {unknown}")
    picked = sorted(dict.fromkeys(families), key=order.__getitem__)[:MAX_INST_TOKENS]
    return sort_header([f"Inst_{f}" for f in picked]
                       + [n for n in names if not n.startswith("Inst_")])
