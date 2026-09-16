"""
v4 attribute header: control tokens derived from the MIDI itself.

Every v4 training document starts with a short header describing its
contents (instrument families, note density, polyphony, pitch range, tempo,
source, optional genre). No labels are needed: everything but Source/Genre
is computed from the notes and tempo events. At inference the header is how a caller steers the
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
# tempo in BPM: slow .. fast. The body stream is tempo-invariant (Bar /
# Position, `use_tempos=False`), so this is the only place the model learns
# that 90 and 170 BPM drums are different animals. Coarse on purpose: half-
# and double-time notation is inconsistent across files (pop909 sits at a
# median 73), so fine bins would mostly encode convention.
TEMPO_EDGES = (70, 90, 110, 130, 150, 175)   # -> buckets 0..6
# performance transcriptions carry a placeholder tempo, not a musical one
# (sampled 2026-09-16: maestro 100% at 120, giantmidi 73%; aria is built
# the same way) -- no Tempo token for them, the header simply omits the family
TEMPO_PLACEHOLDER_SOURCES = ("aria", "maestro", "giantmidi")

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

HEADER_PREFIXES = ("Task_", "Inst_", "Density_", "Poly_", "Range_", "Tempo_",
                   "Source_", "Genre_", "Quality_")
NEVER_DROP = ("Task_",)      # header dropout must leave the task token alone


def header_vocab() -> list[str]:
    """All header token names, in a fixed order (vocab ids follow it)."""
    names = [f"Task_{t}" for t in TASKS]
    names += [f"Inst_{f}" for f in INSTRUMENT_FAMILIES] + [f"Inst_{DRUMS}"]
    names += [f"Density_{i}" for i in range(len(DENSITY_EDGES) + 1)]
    names += [f"Poly_{i}" for i in range(len(POLY_EDGES) + 1)]
    names += [f"Range_{i}" for i in range(len(RANGE_EDGES) + 1)]
    names += [f"Tempo_{i}" for i in range(len(TEMPO_EDGES) + 1)]
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


def tempo_bucket(bpm: float) -> int:
    return _bucket(bpm, TEMPO_EDGES)


def dominant_tempo(score) -> float | None:
    """The tempo in force for the most ticks of `score`; None without a
    tempo event. A setup event at tick 0 that the real tempo replaces a
    beat later, or a closing ritardando, must not decide the token."""
    tempos = sorted(score.tempos, key=lambda t: t.time)
    if not tempos:
        return None
    end = max((n.time + n.duration for t in score.tracks for n in t.notes),
              default=0)
    if len(tempos) == 1 or end <= 0:
        return float(tempos[0].qpm)
    weight: Counter[float] = Counter()
    for i, t in enumerate(tempos):
        start = max(0, t.time)                    # trimmed scores can sit < 0
        stop = tempos[i + 1].time if i + 1 < len(tempos) else end
        weight[float(t.qpm)] += max(0, min(stop, end) - start)
    bpm, ticks = max(weight.items(), key=lambda kv: kv[1])
    return bpm if ticks > 0 else float(tempos[0].qpm)


def tempo_tokens(score, source: str | None = None,
                 tempo: float | None = None) -> list[str]:
    """[Tempo_<bucket>] or [] when the file's tempo can't be trusted (a
    placeholder source, no tempo event). An explicit `tempo` -- the DAW
    clock at inference -- wins over whatever the score says."""
    if tempo is None:
        if source in TEMPO_PLACEHOLDER_SOURCES or score is None:
            return []
        tempo = dominant_tempo(score)
    if tempo is None or tempo <= 0:
        return []
    return [f"Tempo_{tempo_bucket(tempo)}"]


def header_for_score(score, source: str | None = None,
                     genres: list[str] | None = None,
                     quality: int | None = None,
                     tempo: float | None = None) -> list[str]:
    """Full header (token names) for a document whose content is `score`.
    `tempo` overrides the score's own tempo events (see tempo_tokens)."""
    names = instrument_tokens(score) + content_tokens(score)
    names += tempo_tokens(score, source, tempo)
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
