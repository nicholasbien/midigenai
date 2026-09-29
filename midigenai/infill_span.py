"""Regenerate a span of bars inside an upload: "keep this, redo bars 5-6".

Continuation writes after the upload and accompaniment writes over it; infill
writes *between* two parts of it. The model sees the bars before the span and
the bars after it (`BOS Task_infill header prefix MASK suffix SEP`) and writes
the missing bars, which are spliced back into the original file in place. Every
note outside the span is returned untouched, on its own track.

The span is limited to the shape v4/v5 were trained on (data/v4_docs.py,
"span infill windows"): 1-MAX_SPAN_BARS bars, at least one bar of context on
each side, inside a CONTEXT_BARS window. Anything else is refused with a
message saying why, the way track subsets for continuation were refused until
measured; widen the limits after measuring, not before.

Split from modal_serve so the planning and splicing run (and are tested) on
CPU without Modal; `MidiGen.infill_batch` only adds the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from symusic import Score, Track

from midigenai.data.v4_docs import _window, bar_edges
from midigenai.tokenizer import normalize_drums

# The trained infill shape: v4_docs.DocBuilder(context_bars=16, max_span_bars=4)
CONTEXT_BARS = 16
MAX_SPAN_BARS = 4


class InfillError(ValueError):
    """The requested span cannot be infilled; the message says why and what can."""


@dataclass
class InfillPlan:
    score: Score        # the upload, drums normalized, original ticks and bar numbers
    edges: list[int]    # bar lines of `score`
    start: int          # first bar regenerated (0-based, file bar numbers)
    bars: int           # bars regenerated
    ctx_start: int      # context window [ctx_start, ctx_end) the model sees
    ctx_end: int

    @property
    def span_ticks(self) -> tuple[int, int]:
        return self.edges[self.start], self.edges[self.start + self.bars]

    def window(self) -> Score:
        """The context window, re-based to tick 0 (header source)."""
        return _window(self.score, self.edges[self.ctx_start], self.edges[self.ctx_end])

    def prefix(self) -> Score:
        return _window(self.score, self.edges[self.ctx_start], self.edges[self.start])

    def suffix(self) -> Score:
        return _window(self.score, self.edges[self.start + self.bars], self.edges[self.ctx_end])

    @property
    def prefix_bars(self) -> int:
        return self.start - self.ctx_start

    @property
    def suffix_bars(self) -> int:
        return self.ctx_end - self.start - self.bars


def context_window(n_bars: int, start: int, bars: int,
                   context_bars: int = CONTEXT_BARS) -> tuple[int, int]:
    """[b0, b1) of at most `context_bars` bars around the span, the span as
    near the middle as the file allows. Assumes the span leaves >= 1 bar on
    each side inside the file; the window then does too."""
    ctx = min(context_bars, n_bars)
    b0 = start - (ctx - bars) // 2
    b0 = max(0, min(b0, n_bars - ctx))
    return b0, b0 + ctx


def plan(midi_bytes: bytes, start: int, bars: int) -> InfillPlan:
    """Validate a span request against the upload and the trained shape.

    `start` is a 0-based bar index in the file's own bar numbering (bar 1 in a
    DAW is start=0); leading silence is NOT trimmed, so the numbers match
    what the user sees."""
    score = Score.from_midi(BytesIO(midi_bytes).read())
    normalize_drums(score, "upload.mid")
    if not sum(len(t.notes) for t in score.tracks):
        raise InfillError("upload has no notes")
    edges = bar_edges(score)
    n_bars = len(edges) - 1
    if not 1 <= bars <= MAX_SPAN_BARS:
        raise InfillError(f"bars={bars}: infill can regenerate 1-{MAX_SPAN_BARS} bars "
                          f"at a time (the span lengths it was trained on)")
    if start < 1 or start + bars > n_bars - 1:
        # training always kept >=1 bar on each side; an open-ended span is
        # continuation (at the end) or untrained (at the start)
        hi = n_bars - bars - 1
        where = f"start={start}..{hi}" if hi >= 1 else "none: the upload is too short"
        raise InfillError(
            f"start={start}, bars={bars}: infill needs at least one kept bar before "
            f"and after the span; this upload has {n_bars} bars (valid {where})")
    b0, b1 = context_window(n_bars, start, bars)
    p = InfillPlan(score, edges, start, bars, b0, b1)
    if not sum(len(t.notes) for sc in (p.prefix(), p.suffix()) for t in sc.tracks):
        raise InfillError("the bars around the span are empty; nothing to infill from")
    return p


def _track_key(t) -> tuple[bool, int]:
    return (bool(t.is_drum), 0 if t.is_drum else int(t.program))


def splice(p: InfillPlan, answer: Score) -> Score:
    """The upload with bars [start, start+bars) replaced by `answer`.

    `answer` is a self-contained segment starting at bar 0 of the span, at
    any tick rate. Notes starting inside the span are removed; notes that
    start before it and ring into it are cut at the span start (the model's
    prefix saw them cut there too). Generated notes land on the upload's
    track with the same instrument, else on a new track, and are cut at the
    span end so they cannot bleed over the kept bars."""
    s, e = p.span_ticks
    out = p.score.copy()
    for tr in out.tracks:
        kept = []
        for n in tr.notes:
            if s <= n.time < e:
                continue
            n = n.copy()
            if n.time < s < n.time + n.duration:
                n.duration = s - n.time
            kept.append(n)
        tr.notes.clear()
        tr.notes.extend(kept)
    gen = answer.resample(tpq=p.score.ticks_per_quarter)
    for gt in gen.tracks:
        if not len(gt.notes):
            continue
        dst = next((t for t in out.tracks if _track_key(t) == _track_key(gt)), None)
        if dst is None:
            dst = Track(name="infill", program=gt.program, is_drum=gt.is_drum)
            out.tracks.append(dst)
        for n in gt.notes:
            n = n.copy()
            n.time += s
            if n.time >= e:
                continue
            n.duration = max(1, min(n.duration, e - n.time))
            dst.notes.append(n)
        dst.notes.sort()
    return out


def prompt_segments(gen, p: InfillPlan) -> tuple[list[int], list[int]]:
    """(prefix ids, suffix ids), each padded to its exact bar count the way
    v4_docs._segment pads training segments."""
    prefix = gen.pad_to_bars(gen.tokenizer(p.prefix()).ids, p.prefix_bars)
    suffix = gen.pad_to_bars(gen.tokenizer(p.suffix()).ids, p.suffix_bars)
    return prefix, suffix


def run(gen, midi_bytes: bytes, start: int, bars: int, *, n_samples: int = 1,
        tempo_bpm: float = 120.0, **gen_kwargs) -> tuple[InfillPlan, list[Score], list[int]]:
    """Plan, prompt and splice `n_samples` takes. Returns (plan, spliced
    scores, generated note counts). `gen` is a v4 Generator."""
    from midigenai.attributes import header_for_score

    p = plan(midi_bytes, start, bars)
    prefix, suffix = prompt_segments(gen, p)
    header = gen.sp.header_ids_for(gen.tokenizer, header_for_score(p.window(), tempo=tempo_bpm))
    outs, counts = [], []
    for _ in range(n_samples):
        ids = list(gen.infill(prefix, suffix, bars, header=header, **gen_kwargs))
        answer = gen.tokenizer.decode(ids)
        counts.append(sum(len(t.notes) for t in answer.tracks))
        outs.append(splice(p, answer))
    return p, outs, counts
