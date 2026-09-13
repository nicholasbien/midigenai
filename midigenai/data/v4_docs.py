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

from symusic import Score, TimeSignature

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
                 genres: dict[str, list[str]] | None = None,
                 quality: dict[str, int] | None = None,
                 segment_eos: bool = False):
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
        self.genres = genres or {}
        self.quality = quality or {}      # path -> q_bucket (quality_predictor)
        # EOS after an accompaniment / infill target? Off by default: pilot
        # arms D vs E (2026-09-13) showed those fixed-window EOS tokens teach
        # continuation to end early (47-57% self-termination vs 17%).
        # Inference stops targets by bar count, and the next document's BOS
        # is the terminator the model sees instead.
        self.segment_eos = segment_eos

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
                cond_idx, tgt_idx = live[:n_cond], live[n_cond:]
                win = _window(score, s, e)
                cond = _subscore(win, cond_idx)
                tgt = _subscore(win, tgt_idx)
                if _n_notes(cond) < MIN_SEGMENT_NOTES or _n_notes(tgt) < MIN_SEGMENT_NOTES:
                    continue
                # header describes the whole window: the instruments the
                # caller wants in the result, condition included
                segs = (self._segment(cond, self.window_bars),
                        self._segment(tgt, self.window_bars))
                if None in segs:
                    continue
                doc = accompaniment_doc(sp, self._header(win, source, path), *segs)
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
