"""
Turn one cleaned MIDI file into v4 training documents (token-id lists).

Per file:
  - 1 continuation doc of the full mix (+ up to `track_views` solo views),
  - `accomp_windows` accompaniment docs: a bar-aligned window of
    `window_bars` bars, 1-2 tracks as the condition, the rest as the target,
  - `infill_windows` span-infill docs: a `context_bars` window with a span of
    1-`max_span_bars` bars cut out and moved after SEP.

Segments are clipped at bar lines and re-based to tick 0, so each segment's
Bar/Position tokens start at bar 0 of that segment (see sequence_format.py).
REMI emits no Bar tokens after a segment's last note, so every segment is
padded with empty `Bar TimeSig` pairs to its exact bar length: both sides of
SEP always carry the same number of Bar tokens, which is what makes
"stop after N bars" an exact contract at inference.
Files with a time signature v4 can't express are skipped (MidiTok would
silently re-bar them as 4/4, which is exactly the downbeat error this
tokenizer exists to remove).
"""

from __future__ import annotations

import random

from symusic import Score, TimeSignature, Track

from midigenai.attributes import header_for_score, source_from_path
from midigenai.sequence_format import (
    Specials, accompaniment_doc, continuation_doc, infill_doc,
)
from midigenai.tokenizer import normalize_drums, supported_time_signatures

MIN_VIEW_NOTES = 64       # a solo view / condition track needs this many notes
MIN_SEGMENT_NOTES = 4     # each side of SEP needs some content
MIN_DOC_TOKENS = 8

SKIP_TIMESIG = "timesig"
SKIP_EMPTY = "empty"
SKIP_ERROR = "error"


def trim_leading(sc: Score) -> Score:
    """Shift so the first onset is at tick 0 (removes empty leading bars)."""
    starts = [n.start for t in sc.tracks for n in t.notes]
    return sc.shift_time(-min(starts)) if starts and min(starts) > 0 else sc


def _subscore(sc: Score, track_idxs) -> Score:
    out = Score(sc.tpq)
    for ts in sc.time_signatures:
        out.time_signatures.append(ts)
    for i in track_idxs:
        out.tracks.append(sc.tracks[i])
    return out


def _window(sc: Score, start_tick: int, end_tick: int) -> Score:
    """Bar-aligned clip, re-based to tick 0. Notes crossing the end are cut.

    MidiTok quantizes onsets to a 1/8-beat grid, so an onset inside the last
    half-position of the window would round up onto the next bar line and
    give the segment one Bar token too many; the clip end is pulled in by
    half a position to keep the bar count exact."""
    guard = max(1, sc.tpq // 16)
    w = sc.clip(start_tick, end_tick - guard, clip_end=True)
    if start_tick:
        w = w.shift_time(-start_tick)
    # the meter in force at the window start must be restated at tick 0:
    # clip keeps only events inside the window, and MidiTok would bar the
    # opening under 4/4 until the next change
    active = None
    for ts in sc.time_signatures:
        if ts.time <= start_tick and (active is None or ts.time >= active.time):
            active = ts
    inside = [ts for ts in w.time_signatures if ts.time > 0]
    w.time_signatures.clear()
    if active is not None:
        w.time_signatures.append(TimeSignature(0, active.numerator, active.denominator))
    for ts in inside:
        w.time_signatures.append(ts)
    return w


def _n_notes(sc: Score) -> int:
    return sum(len(t.notes) for t in sc.tracks)


def bar_edges(sc: Score) -> list[int]:
    """Tick of every bar line from 0 through the end of the last bar, from the
    file's own time-signature map (bars change length mid-song in ~5% of
    multi-track files, so a fixed ticks-per-bar would drift)."""
    db = [int(x) for x in sc.get_downbeats()]
    if not db:
        db = [0]
    last = db[-1] - db[-2] if len(db) > 1 else sc.tpq * 4
    end = sc.end()
    while db[-1] <= end:
        db.append(db[-1] + last)
    return db


HAND_SPLIT_MIN, HAND_SPLIT_MAX = 50, 67      # keep the split near middle C
HAND_SPLIT_MIN_HELD = 0.35                   # see split_hands


def _held_under(low, high, margin: int) -> float:
    """Fraction of the high part's onsets that arrive over a low note which
    began at least `margin` ticks earlier and is still sounding.

    Plain overlap is useless here: legato and pedal make consecutive notes of
    a single line overlap, so 97% of Aria windows passed that test. Requiring
    the low note to have *started a beat earlier* separates them cleanly - a
    synthetic two-hand window scores 0.75, a synthetic melody 0.00, and real
    Aria windows sit at 0.59, much nearer the two-hand end.
    """
    lo = sorted((n.start, n.start + max(n.duration, 1))
                for t in low.tracks for n in t.notes)
    onsets = sorted(n.start for t in high.tracks for n in t.notes)
    if not lo or not onsets:
        return 0.0
    import bisect
    starts = [s for s, _ in lo]
    held = 0
    for o in onsets:
        i = bisect.bisect_right(starts, o - margin)
        if any(e > o for _, e in lo[max(0, i - 24):i]):
            held += 1
    return held / len(onsets)


def split_hands(win: Score) -> tuple[Score, Score] | None:
    """Split a solo keyboard window into (low hand, high hand).

    A third of the corpus is Aria: single-track piano transcriptions, which
    produce no accompaniment documents at all and left the task at 13.6% of
    the corpus instead of the intended 25%. Two hands of one piano are a
    genuine accompaniment pair, and they are the most abundant one we have.
    The split point is the window's median pitch, held near middle C so a
    bass-register passage does not get cut in an implausible place.

    Rejects windows that are really one melodic line: slicing a melody at a
    pitch produces two half-melodies, not an accompaniment pair, and a single
    line crossing the split still puts notes on both sides of almost every
    bar, so coverage cannot tell them apart. `_held_under` can.

    Note that notes-per-onset is the wrong test here (Aria's median is 1.12,
    which looks monophonic): two hands routinely strike at different moments,
    and what makes them two hands is that one sustains under the other.
    """
    notes = [n for t in win.tracks if not t.is_drum for n in t.notes]
    if len(notes) < 2 * MIN_SEGMENT_NOTES:
        return None
    pitches = sorted(n.pitch for n in notes)
    split = min(max(pitches[len(pitches) // 2], HAND_SPLIT_MIN), HAND_SPLIT_MAX)
    low, high = Score(win.tpq), Score(win.tpq)
    for dst in (low, high):
        for ts in win.time_signatures:
            dst.time_signatures.append(ts)
    for t in win.tracks:
        if t.is_drum:
            continue
        lt, ht = Track(program=t.program), Track(program=t.program)
        for n in t.notes:
            (lt if n.pitch < split else ht).notes.append(n)
        low.tracks.append(lt)
        high.tracks.append(ht)
    if _n_notes(low) < MIN_SEGMENT_NOTES or _n_notes(high) < MIN_SEGMENT_NOTES:
        return None
    beat = max(win.ticks_per_quarter, 1)
    if max(_held_under(low, high, beat), _held_under(high, low, beat)) < HAND_SPLIT_MIN_HELD:
        return None
    return low, high


def _tracks_in_window(sc: Score, start: int, end: int, min_notes: int) -> list[int]:
    out = []
    for i, t in enumerate(sc.tracks):
        n = sum(1 for x in t.notes if start <= x.time < end)
        if n >= min_notes:
            out.append(i)
    return out


class DocBuilder:
    # defaults: the first pilot corpus (6 windows) landed at 49/38/13 on
    # Lakh; 4 windows brings multi-track sources near 55/32/13 and the
    # corpus-wide share (single-track sources add no accompaniment) near
    # the 60/25/15 target
    def __init__(self, tokenizer, track_views: int = 2, accomp_windows: int = 4,
                 infill_windows: int = 2, window_bars: int = 16,
                 context_bars: int = 16, max_span_bars: int = 4,
                 single_target_frac: float = 0.6,
                 genres: dict[str, list[str]] | None = None,
                 quality: dict[str, int] | None = None,
                 segment_eos: bool = False,
                 hand_split_windows: int = 2):
        self.tok = tokenizer
        self.sp = Specials.from_tokenizer(tokenizer)
        self.bar_id = tokenizer.vocab["Bar_None"]
        self.timesig_ids = {v for k, v in tokenizer.vocab.items() if k.startswith("TimeSig_")}
        self.track_views = track_views
        self.accomp_windows = accomp_windows
        self.infill_windows = infill_windows
        self.window_bars = window_bars
        self.context_bars = context_bars
        self.max_span_bars = max_span_bars
        # Share of accompaniment documents whose target is ONE track rather
        # than every remaining track. "Add a bass" is the shape the jam asks
        # for; "write the other five parts at once" is a harder, rarer task.
        # Keeping both teaches the single-part skill without losing whole-
        # arrangement coherence.
        self.single_target_frac = single_target_frac
        self.genres = genres or {}
        self.quality = quality or {}      # path -> q_bucket (quality_predictor)
        # EOS after an accompaniment / infill target? Off by default: pilot
        # arms D vs E (2026-09-13) showed those fixed-window EOS tokens teach
        # continuation to end early (47-57% self-termination vs 17%).
        # Inference stops targets by bar count, and the next document's BOS
        # is the terminator the model sees instead.
        self.segment_eos = segment_eos
        # solo-keyboard files: accompaniment windows from a left/right hand
        # split, generated in BOTH directions (see split_hands)
        self.hand_split_windows = hand_split_windows

    # -- helpers -- #
    def _ids(self, sc: Score) -> list[int]:
        return self.tok(sc).ids

    def _segment(self, sc: Score, n_bars: int) -> list[int] | None:
        """Tokenize a re-based segment and pad to exactly `n_bars` Bar tokens.
        None when the tokenizer barred it differently than the file's
        downbeats say (meter change on an off-bar tick): the window is
        dropped rather than teach a wrong bar count."""
        ids = self._ids(sc)
        have = sum(1 for t in ids if t == self.bar_id)
        if have > n_bars:
            return None
        if have < n_bars:
            ts = next((t for t in ids if t in self.timesig_ids), None)
            if ts is None:            # empty segment: derive from the score
                tsig = min(sc.time_signatures, key=lambda x: x.time) if len(sc.time_signatures) else None
                name = f"TimeSig_{tsig.numerator}/{tsig.denominator}" if tsig else "TimeSig_4/4"
                ts = self.tok.vocab[name]
            ids = ids + [self.bar_id, ts] * (n_bars - have)
        return ids

    def _header(self, sc: Score, source: str | None, path: str) -> list[int]:
        names = header_for_score(sc, source=source, genres=self.genres.get(path),
                                 quality=self.quality.get(path))
        return self.sp.header_ids_for(self.tok, names)

    # -- entry point -- #
    def build(self, path: str) -> dict[str, list[list[int]]] | str:
        """Returns {"continuation": [...], "accompaniment": [...],
        "infill": [...]} or a SKIP_* reason string when the file is unusable."""
        try:
            score = Score(path)
        except Exception:
            return SKIP_ERROR
        if not supported_time_signatures(score):
            return SKIP_TIMESIG
        normalize_drums(score, path.rsplit("/", 1)[-1])
        score = trim_leading(score)
        if _n_notes(score) == 0:
            return SKIP_EMPTY
        source = source_from_path(path)
        rng = random.Random(path)           # reproducible per file
        sp = self.sp
        out = {"continuation": [], "accompaniment": [], "infill": []}

        # 1. full mix + solo views
        full_ids = self._ids(score)
        if len(full_ids) < MIN_DOC_TOKENS:
            return SKIP_EMPTY
        out["continuation"].append(
            continuation_doc(sp, self._header(score, source, path), full_ids))
        candidates = [i for i, t in enumerate(score.tracks) if len(t.notes) >= MIN_VIEW_NOTES]
        if self.track_views and len(candidates) >= 2:
            picks = candidates[:]
            rng.shuffle(picks)
            for i in picks[:self.track_views]:
                solo = trim_leading(_subscore(score, [i]))
                ids = self._ids(solo)
                if len(ids) >= MIN_DOC_TOKENS:
                    out["continuation"].append(
                        continuation_doc(sp, self._header(solo, source, path), ids))

        edges = bar_edges(score)
        n_bars = len(edges) - 1

        # 2. accompaniment windows (multi-track files only)
        if len(candidates) >= 2 and n_bars >= self.window_bars:
            for _ in range(self.accomp_windows):
                b0 = rng.randrange(0, n_bars - self.window_bars + 1)
                s, e = edges[b0], edges[b0 + self.window_bars]
                live = _tracks_in_window(score, s, e, MIN_SEGMENT_NOTES)
                if len(live) < 2:
                    continue
                rng.shuffle(live)
                n_cond = 1 if len(live) == 2 or rng.random() < 0.7 else 2
                cond_idx, rest = live[:n_cond], live[n_cond:]
                tgt_idx = ([rng.choice(rest)]
                           if rng.random() < self.single_target_frac else rest)
                win = _window(score, s, e)
                cond = _subscore(win, cond_idx)
                tgt = _subscore(win, tgt_idx)
                if _n_notes(cond) < MIN_SEGMENT_NOTES or _n_notes(tgt) < MIN_SEGMENT_NOTES:
                    continue
                segs = (self._segment(cond, self.window_bars),
                        self._segment(tgt, self.window_bars))
                if None in segs:
                    continue
                # the header names exactly what is in this document, condition
                # plus target, so asking for `Inst_Bass` at inference means
                # "add a bass" rather than "add everything this file had"
                header = self._header(_subscore(win, cond_idx + tgt_idx), source, path)
                doc = accompaniment_doc(sp, header, *segs)
                out["accompaniment"].append(doc if self.segment_eos else doc[:-1])

        # 2b. solo keyboard: left hand <-> right hand, both directions
        solo = [i for i, t in enumerate(score.tracks) if not t.is_drum and len(t.notes)]
        if (self.hand_split_windows and len(solo) == 1
                and not any(t.is_drum and len(t.notes) for t in score.tracks)
                and n_bars >= self.window_bars):
            for _ in range(self.hand_split_windows):
                b0 = rng.randrange(0, n_bars - self.window_bars + 1)
                win = _window(score, edges[b0], edges[b0 + self.window_bars])
                hands = split_hands(win)
                if hands is None:
                    continue
                low, high = hands
                header = self._header(win, source, path)
                for cond, tgt in ((low, high), (high, low)):
                    segs = (self._segment(cond, self.window_bars),
                            self._segment(tgt, self.window_bars))
                    if None in segs:
                        continue
                    doc = accompaniment_doc(sp, header, *segs)
                    out["accompaniment"].append(doc if self.segment_eos else doc[:-1])

        # 3. span infill windows
        if n_bars >= 2 * self.max_span_bars + 2:
            ctx = min(self.context_bars, n_bars)
            for _ in range(self.infill_windows):
                b0 = rng.randrange(0, n_bars - ctx + 1)
                span = rng.randint(1, self.max_span_bars)
                i = rng.randint(1, ctx - span - 1)          # keep >=1 bar each side
                s, e = edges[b0], edges[b0 + ctx]
                win = _window(score, s, e)
                # bar lines inside the window, re-based like the window
                we = [x - s for x in edges[b0:b0 + ctx + 1]]
                prefix = _window(win, 0, we[i])
                middle = _window(win, we[i], we[i + span])
                suffix = _window(win, we[i + span], we[ctx])
                if _n_notes(middle) < MIN_SEGMENT_NOTES or _n_notes(prefix) + _n_notes(suffix) < MIN_SEGMENT_NOTES:
                    continue
                segs = (self._segment(prefix, i),
                        self._segment(suffix, ctx - i - span),
                        self._segment(middle, span))
                if None in segs:
                    continue
                doc = infill_doc(sp, self._header(win, source, path), *segs)
                out["infill"].append(doc if self.segment_eos else doc[:-1])
        return out
