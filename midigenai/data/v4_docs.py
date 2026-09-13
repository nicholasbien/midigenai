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

from symusic import Score

from midigenai.attributes import header_for_score, source_from_path
from midigenai.sequence_format import (
    Specials, accompaniment_doc, continuation_doc, infill_doc,
)
from midigenai.tokenizer import normalize_drums, supported_time_signatures
from midigenai.attributes import ticks_per_bar

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
    """Bar-aligned clip, re-based to tick 0. Notes crossing the end are cut."""
    w = sc.clip(start_tick, end_tick, clip_end=True)
    if start_tick:
        w = w.shift_time(-start_tick)
    if not len(w.time_signatures):
        for ts in sc.time_signatures:
            w.time_signatures.append(ts)
    return w


def _n_notes(sc: Score) -> int:
    return sum(len(t.notes) for t in sc.tracks)


def _tracks_in_window(sc: Score, start: int, end: int, min_notes: int) -> list[int]:
    out = []
    for i, t in enumerate(sc.tracks):
        n = sum(1 for x in t.notes if start <= x.time < end)
        if n >= min_notes:
            out.append(i)
    return out


class DocBuilder:
    # defaults measured on lakh multi-track files: ~62/27/11 token share
    # (continuation / accompaniment / infill); single-track sources have no
    # accompaniment docs, so the corpus-wide share lands near 60/25/15
    def __init__(self, tokenizer, track_views: int = 2, accomp_windows: int = 6,
                 infill_windows: int = 2, window_bars: int = 16,
                 context_bars: int = 16, max_span_bars: int = 4,
                 genres: dict[str, list[str]] | None = None):
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

    # -- helpers -- #
    def _ids(self, sc: Score) -> list[int]:
        return self.tok(sc).ids

    def _segment(self, sc: Score, n_bars: int) -> list[int]:
        """Tokenize a re-based segment and pad to exactly `n_bars` Bar tokens."""
        ids = self._ids(sc)
        have = sum(1 for t in ids if t == self.bar_id)
        if have < n_bars:
            ts = next((t for t in ids if t in self.timesig_ids), None)
            if ts is None:            # empty segment: derive from the score
                tsig = min(sc.time_signatures, key=lambda x: x.time) if len(sc.time_signatures) else None
                name = f"TimeSig_{tsig.numerator}/{tsig.denominator}" if tsig else "TimeSig_4/4"
                ts = self.tok.vocab[name]
            ids = ids + [self.bar_id, ts] * (n_bars - have)
        return ids

    def _header(self, sc: Score, source: str | None, path: str) -> list[int]:
        names = header_for_score(sc, source=source, genres=self.genres.get(path))
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

        tpb = ticks_per_bar(score)
        n_bars = int(score.end() // tpb) + 1

        # 2. accompaniment windows (multi-track files only)
        if len(candidates) >= 2 and n_bars >= self.window_bars:
            for _ in range(self.accomp_windows):
                b0 = rng.randrange(0, n_bars - self.window_bars + 1)
                s, e = b0 * tpb, (b0 + self.window_bars) * tpb
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
                out["accompaniment"].append(accompaniment_doc(
                    sp, self._header(win, source, path),
                    self._segment(cond, self.window_bars),
                    self._segment(tgt, self.window_bars)))

        # 3. span infill windows
        if n_bars >= 2 * self.max_span_bars + 2:
            ctx = min(self.context_bars, n_bars)
            for _ in range(self.infill_windows):
                b0 = rng.randrange(0, n_bars - ctx + 1)
                span = rng.randint(1, self.max_span_bars)
                i = rng.randint(1, ctx - span - 1)          # keep >=1 bar each side
                s, e = b0 * tpb, (b0 + ctx) * tpb
                win = _window(score, s, e)
                prefix = _window(win, 0, i * tpb)
                middle = _window(win, i * tpb, (i + span) * tpb)
                suffix = _window(win, (i + span) * tpb, ctx * tpb)
                if _n_notes(middle) < MIN_SEGMENT_NOTES or _n_notes(prefix) + _n_notes(suffix) < MIN_SEGMENT_NOTES:
                    continue
                out["infill"].append(infill_doc(
                    sp, self._header(win, source, path),
                    self._segment(prefix, i),
                    self._segment(suffix, ctx - i - span),
                    self._segment(middle, span)))
        return out
