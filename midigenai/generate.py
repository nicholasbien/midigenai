"""
midigenai inference: load a trained checkpoint, encode MIDI input, stream a continuation.

Two modes:
- generate_ids(): yields token IDs as they're produced (for low-level use)
- stream_notes(): yields note dicts as the model emits complete events
                  (for piping into server2.py / Ableton)
- generate_to_midi(): one-shot, returns a complete MIDI file
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .model import ModelConfig, MusicTransformer
from .tokenizer import build_tokenizer, is_v4, load_tokenizer


def _auto_device() -> torch.device:
    """
    Pick an inference device.

    Even at 100M params, batch-1 decode does too little work per step for MPS:
    kernel dispatch overhead dominates and CPU is ~3x faster on M-series Macs
    (v2-100m fp32: CPU 146 t/s vs MPS 47 t/s, measured 2026-08-30 on M3 Max).
    So skip MPS by default.
    Override with OMN_USE_MPS=1 if you have a larger model where MPS pays off.
    """
    import os
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("OMN_USE_MPS") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int
    program: int = 0


MIN_PROMPT_TOKENS = 256  # never shrink a long prompt below this to make room for generation


def fit_to_context(
    prompt_ids: list[int],
    max_new_tokens: int,
    max_seq_len: int,
    min_prompt_tokens: int = MIN_PROMPT_TOKENS,
) -> tuple[list[int], int]:
    """Make `len(prompt) + max_new_tokens <= max_seq_len`.

    A too-long prompt is cut at the *end*: the model continues from wherever
    the cut lands, and callers join `fitted_prompt + continuation` so the
    result is one coherent piece (the dropped tail is simply gone). The head
    is kept rather than the tail so the cut point is easy to show against the
    original file. `max_new_tokens` is only reduced when the prompt could not
    otherwise keep `min_prompt_tokens` of context.

    Without this, a dense upload (~7.7k+ tokens at the 8192 serving context)
    ran the KV cache past the RoPE table mid-decode and crashed attention with
    a shape mismatch — every request for that file 500'd.
    """
    prompt_ids = list(prompt_ids)
    if max_seq_len <= 0:
        return prompt_ids, max_new_tokens
    # (a tiny context, e.g. in tests, still leaves at least half for generation)
    min_prompt = min(min_prompt_tokens, len(prompt_ids), max_seq_len // 2)
    max_new_tokens = max(0, min(max_new_tokens, max_seq_len - min_prompt))
    keep = max_seq_len - max_new_tokens
    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[:keep]
    return prompt_ids, max_new_tokens


OVERLAY_TPQ = 480          # a common grid both sides resample onto


def overlay(condition, generated, tpq: int = OVERLAY_TPQ):
    """Stack a generated accompaniment on top of the part it answers.

    Both scores start at bar 0 of the same window, so the merge is a plain
    union of tracks -- but only after they agree on a tick rate. They do not
    by default: a score decoded from tokens carries the tokenizer's rate
    (16/quarter) while one parsed from an uploaded file carries the file's
    (commonly 220 or 480). Appending the tracks without resampling leaves
    every generated tick reinterpreted against the other grid, which in
    practice collapsed a whole accompaniment onto the first instant of the
    window. Resample both, then merge.

    Returns a new Score; neither input is modified.
    """
    from symusic import Score

    out = Score(tpq)
    cond = condition.resample(tpq=tpq)
    gen = generated.resample(tpq=tpq)
    for ts in (cond.time_signatures or gen.time_signatures):
        out.time_signatures.append(ts)
    for tempo in (cond.tempos or gen.tempos):
        out.tempos.append(tempo)
    for src in (cond, gen):
        for track in src.tracks:
            if len(track.notes):
                out.tracks.append(track)
    return out


def densest_track(score) -> int:
    """Index of the track carrying the most notes.

    Accompaniment is trained to answer ONE part, so a multitrack upload has
    to be narrowed to a single condition. The busiest track is the lead line
    on essentially anything real, and this is a no-op for the single-track
    files the site's presets all are.
    """
    if not score.tracks:
        raise ValueError("score has no tracks")
    return max(range(len(score.tracks)), key=lambda i: len(score.tracks[i].notes))


class Generator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: torch.device | None = None,
        inference_seq_len: int = 8192,
        dtype: torch.dtype | str = torch.float16,
        backend: str = "auto",
    ):
        """
        `backend`: "mlx", "torch", or "auto" (default). On Apple silicon with
        mlx installed, auto picks MLX — it runs the GPU without PyTorch MPS's
        per-op dispatch overhead: 819 t/s decode vs 267 on CPU fp16, and 50 ms
        TTFT vs ~2 s at 2048-token prompts (v2-100m, M3 Max). Outputs are
        token-identical to the torch backend under greedy decoding. Everywhere
        else auto falls back to torch.

        `dtype`: inference precision. fp16 decodes ~1.65x faster than fp32 on
        CPU (243 vs 148 t/s for v2-100m on M3 Max) and nearly eliminates the
        long-context slowdown (141 vs 78 t/s at ~1500-token context), since the
        per-step KV-cache concat moves half the bytes. RMSNorm and sampling
        logits are computed in fp32 internally regardless. Pass torch.float32
        to exactly reproduce pre-fp16 outputs.
        """
        if backend == "auto":
            try:
                import mlx.core  # noqa: F401
                backend = "mlx"
            except ImportError:
                backend = "torch"
        if backend not in ("mlx", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

        if backend == "mlx":
            import mlx.core as mx
            from .model_mlx import load_model
            mlx_dtype = {
                torch.float16: mx.float16,
                torch.bfloat16: mx.bfloat16,
                torch.float32: mx.float32,
            }[self.dtype]
            self.device = None  # MLX manages placement (unified memory)
            # load_model keeps a converted .safetensors sidecar next to the
            # checkpoint, so warm loads skip torch.load (~1.3 s -> ~15 ms).
            self.model, cfg = load_model(checkpoint_path, dtype=mlx_dtype)
            cfg.max_seq_len = max(cfg.max_seq_len, inference_seq_len)
        else:
            self.device = device or _auto_device()
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cfg = ModelConfig(**ckpt["model_config"])
            # RoPE extrapolates, so we can serve at longer context than training
            # without retraining. (MLX computes RoPE per-position, no table needed.)
            cfg.max_seq_len = max(cfg.max_seq_len, inference_seq_len)
            self.model = MusicTransformer(cfg).to(self.device).eval()
            self.model.load_state_dict(ckpt["model"])
            self.model = self.model.to(self.dtype)
        # Hard ceiling on prompt + generated tokens (the torch RoPE table has
        # exactly this many rows; decoding past it crashes attention).
        self.max_seq_len = cfg.max_seq_len

        self.tokenizer = (
            load_tokenizer(tokenizer_path) if tokenizer_path else build_tokenizer()
        )
        # A tokenizer from a different run decodes every id to the wrong
        # token rather than failing, so the output is plausible-looking
        # nonsense. Checking the one number that must agree turns that into
        # an error naming both sides.
        if len(self.tokenizer.vocab) != cfg.vocab_size:
            raise ValueError(
                f"tokenizer/checkpoint mismatch: tokenizer has "
                f"{len(self.tokenizer.vocab)} tokens, checkpoint was trained "
                f"with {cfg.vocab_size}. They are from different runs "
                f"(checkpoint={checkpoint_path}, tokenizer={tokenizer_path})."
            )
        self.bos_id = self._special_id("BOS_None", "BOS")
        self.eos_id = self._special_id("EOS_None", "EOS")

        # v4 (REMI + header + SEP/MASK): bar counting and the accompaniment /
        # infill prompt layouts live in sequence_format.py
        self.v4 = is_v4(self.tokenizer)
        self.sp = None
        self.bar_id = None
        self.timesig_ids: set[int] = set()
        self.stop_ids: set[int] = {self.eos_id} if self.eos_id is not None else set()
        if self.v4:
            from .sequence_format import Specials
            self.sp = Specials.from_tokenizer(self.tokenizer)
            self.bar_id = self.sp.bar
            self.timesig_ids = {v for k, v in self.tokenizer.vocab.items()
                                if k.startswith("TimeSig_")}
            # SEP / MASK mid-generation are never valid output, and BOS is
            # what follows an accompaniment / infill target in training
            # (segments carry no EOS): all three end generation
            self.stop_ids |= {self.sp.sep, self.sp.mask, self.sp.bos}

    def _special_id(self, *candidates: str) -> int | None:
        for c in candidates:
            if c in self.tokenizer.vocab:
                return self.tokenizer.vocab[c]
        return None

    def _needs_bos(self, prompt_ids: list[int]) -> bool:
        return self.bos_id is not None and (not prompt_ids or prompt_ids[0] != self.bos_id)

    def fit_to_context(
        self, prompt_ids: list[int], max_new_tokens: int
    ) -> tuple[list[int], int]:
        """Trim `prompt_ids` / `max_new_tokens` so prompt (+ the BOS that
        generation prepends) + continuation fits `self.max_seq_len`.
        Idempotent, so callers may fit first and join `fitted + new_ids`;
        `generate_ids` fits again internally as a safety net."""
        added = self._needs_bos(prompt_ids)
        ids = [self.bos_id, *prompt_ids] if added else list(prompt_ids)
        ids, max_new_tokens = fit_to_context(ids, max_new_tokens, self.max_seq_len)
        return (ids[1:] if added else ids), max_new_tokens

    # ---------- encode user input ---------- #

    def encode_midi_file(self, midi_path: str | Path) -> list[int]:
        return self.tokenizer(Path(midi_path)).ids

    def encode_midi_bytes(self, data: bytes) -> list[int]:
        # symusic accepts a path; route through a tempfile to avoid format guessing
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            return self.encode_midi_file(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

    @staticmethod
    def detect_tempo(midi_path: str | Path) -> float:
        """Read the first tempo from a MIDI file. Returns 120.0 if none."""
        from symusic import Score
        score = Score(str(midi_path))
        if len(score.tempos) > 0:
            return float(score.tempos[0].qpm)
        return 120.0

    @staticmethod
    def detect_tempo_bytes(data: bytes) -> float:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            return Generator.detect_tempo(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # ---------- v4 header / bar helpers ---------- #

    def _require_v4(self, what: str) -> None:
        if not self.v4:
            raise RuntimeError(f"{what} needs a v4 (REMI + header) checkpoint")

    def make_header(self, midi_path: str | Path | None = None, *,
                    instruments: list[str] | None = None,
                    density: int | None = None, poly: int | None = None,
                    pitch_range: int | None = None,
                    source: str | None = None,
                    genres: list[str] | None = None,
                    tempo: float | None = None) -> list[int]:
        """Attribute-header token ids. With `midi_path` the header describes
        that file (instruments, density, ...) and the keyword arguments
        override individual families; without it only the given families are
        set. `tempo` (BPM, e.g. the DAW clock) sets the Tempo_ family; with a
        file and no `tempo` the file's own tempo events decide it. `instruments` are family names from attributes.INSTRUMENT_FAMILIES
        (+ "Drums"): list the instruments you want IN THE RESULT — for
        accompaniment that means the condition's instrument plus the ones to
        add. Returns [] on a non-v4 checkpoint so callers can always prepend it."""
        if not self.v4:
            return []
        from .attributes import header_for_score, tempo_tokens
        names: list[str] = []
        if midi_path is not None:
            from symusic import Score
            names = header_for_score(Score(str(midi_path)), source=source,
                                     genres=genres, tempo=tempo)
        else:
            if source:
                names.append(f"Source_{source}")
            names += [f"Genre_{g}" for g in (genres or [])]
            names += tempo_tokens(None, tempo=tempo)

        def override(prefix: str, new: list[str]):
            nonlocal names
            names = [n for n in names if not n.startswith(prefix)] + new
        if instruments is not None:
            override("Inst_", [f"Inst_{i}" for i in instruments])
        if density is not None:
            override("Density_", [f"Density_{density}"])
        if poly is not None:
            override("Poly_", [f"Poly_{poly}"])
        if pitch_range is not None:
            override("Range_", [f"Range_{pitch_range}"])
        if source is not None and midi_path is not None:
            override("Source_", [f"Source_{source}"])
        # canonical family order, as the builder writes it
        from .attributes import sort_header
        return self.sp.header_ids_for(self.tokenizer, sort_header(names))

    def count_bars(self, ids) -> int:
        return sum(1 for t in ids if t == self.bar_id) if self.v4 else 0

    def pad_to_bars(self, ids: list[int], n_bars: int) -> list[int]:
        """Append empty `Bar TimeSig` pairs so `ids` spans exactly `n_bars`
        (a phrase that ends mid-bar is padded to its bar line, so a
        continuation starts on the downbeat; an accompaniment condition
        gets the full window). Raises if `ids` already has more bars."""
        self._require_v4("pad_to_bars")
        have = self.count_bars(ids)
        if have > n_bars:
            raise ValueError(f"prompt spans {have} bars > {n_bars}")
        ts = next((t for t in ids if t in self.timesig_ids),
                  self.tokenizer.vocab["TimeSig_4/4"])
        return list(ids) + [self.bar_id, ts] * (n_bars - have)

    def close_bar(self, ids: list[int]) -> list[int]:
        """Pad an open bar to its bar line (no-op if already on one), so a
        continuation starts on the downbeat."""
        self._require_v4("close_bar")
        if self.ends_on_bar_line(ids):
            return list(ids)
        return self.pad_to_bars(ids, self.count_bars(ids) + 1)

    def ends_on_bar_line(self, ids) -> bool:
        """True when the last musical token is a Bar (or Bar TimeSig) pair."""
        if not self.v4 or not ids:
            return False
        tail = [t for t in ids if t not in self.sp.header_ids and t != self.bos_id]
        if tail and tail[-1] in self.timesig_ids:
            tail = tail[:-1]
        return bool(tail) and tail[-1] == self.bar_id

    # ---------- generation ---------- #

    def generate_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        min_new_tokens: int = 0,
        seed: int | None = None,
        stop_after_bars: int | None = None,
        ban_ids: list[int] | None = None,
        trim_leading_bars: bool = True,
    ) -> Iterator[int]:
        """
        `min_new_tokens`: hold EOS back for the first N tokens. Any prompt whose
        voices all stop at the same instant (an excerpt cut on a bar line, a
        jam partner who stops playing) reads as a piece ending — measured
        P(EOS) as the first token was 0.5–0.93 on such prompts vs ~0.02 with
        natural note-offs — so set this to roughly the length you want
        guaranteed before the model may choose to end.

        `seed`: seeds the backend RNG (mlx / torch) for reproducible takes.

        `stop_after_bars` (v4): stop once the model has produced N bars of
        material. If the prompt ends on a bar line that bar line opens bar 1
        of the answer; otherwise the first generated Bar token does. The Bar
        token that would open bar N+1 is consumed, not yielded. SEP/MASK
        tokens end generation like EOS (they are never valid output).

        `ban_ids` (v4): token ids masked out of sampling. Default for a v4
        checkpoint is SEP / MASK / BOS, which are never valid continuation
        output; `accompany` / `infill` pass a narrower set.

        `trim_leading_bars` (v4, default on): drop `Bar`/`TimeSig` tokens
        emitted before the first note, and do not count them toward
        `stop_after_bars`. A prompt closed to a bar line reads as the end of
        a section, and ~9% of samples answer with several empty bars
        (measured on v4_full ckpt_060000/083000, 2026-09-14); the music that
        follows is fine, so the fix is to start it at the downbeat we gave
        the model instead of several bars later.
        """
        if ban_ids is None:
            ban_ids = self.default_ban_ids()
        prompt_ids, max_new_tokens = self.fit_to_context(prompt_ids, max_new_tokens)
        if self._needs_bos(prompt_ids):
            prompt_ids = [self.bos_id, *prompt_ids]
        raw = self._generate_raw(prompt_ids, max_new_tokens, temperature, top_k,
                                 min_new_tokens, seed, ban_ids)
        yield from self.postprocess(
            prompt_ids, raw,
            stop_after_bars=stop_after_bars,
            trim_leading_bars=trim_leading_bars,
        )

    def default_ban_ids(self) -> list[int] | None:
        """Ids never valid as continuation output (v4): SEP / MASK / BOS.
        None for a pre-v4 checkpoint, which has no such tokens."""
        if not self.v4:
            return None
        return sorted(t for t in (self.sp.sep, self.sp.mask, self.bos_id) if t is not None)

    def postprocess(
        self,
        prompt_ids: list[int],
        raw: Iterable[int],
        stop_after_bars: int | None = None,
        trim_leading_bars: bool = True,
    ) -> Iterator[int]:
        """Turn raw sampled ids into the continuation we hand a caller: stop
        at EOS / SEP / MASK / BOS, drop leading empty bars, and honor
        `stop_after_bars`. See `generate_ids` for what each one is for.

        Split out of `generate_ids` so every decoder shares it — in
        particular `modal_serve`, whose batched loop samples on its own and
        would otherwise serve untrimmed output."""
        if not self.v4:
            yield from raw
            return
        bars = 1 if (stop_after_bars and self.ends_on_bar_line(prompt_ids)) else 0
        started = not trim_leading_bars
        # A prompt that stops mid-bar needs one Bar to close it: Position is
        # bar-relative, so dropping every leading Bar would date the first
        # generated note to the bar the prompt is already part-way through,
        # i.e. before the prompt ends. Hold that Bar (and the TimeSig right
        # after it, which may be a meter change) and emit it with the note.
        held: list[int] = []
        held_open = self.v4 and not self.ends_on_bar_line(prompt_ids)
        for t in raw:
            if t == self.eos_id:
                yield t
                return
            if t in self.stop_ids:
                return
            if not started:
                # swallow empty bars (and their TimeSig) until the first note
                if t == self.bar_id:
                    if held_open and not held:
                        held = [t]
                    continue
                if t in self.timesig_ids:
                    if len(held) == 1:
                        held.append(t)
                    continue
                started = True
                for h in held:
                    if stop_after_bars and h == self.bar_id:
                        bars += 1
                    yield h
                held = []
            if stop_after_bars and t == self.bar_id:
                bars += 1
                if bars > stop_after_bars:
                    return
            yield t

    def _generate_raw(self, prompt_ids, max_new_tokens, temperature, top_k,
                      min_new_tokens, seed, ban_ids=None) -> Iterator[int]:
        if self.backend == "mlx":
            import mlx.core as mx
            if seed is not None:
                mx.random.seed(seed)
            yield from self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=self.eos_id,
                min_new_tokens=min_new_tokens,
                ban_ids=ban_ids or None,
            )
            return
        if seed is not None:
            torch.manual_seed(seed)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        yield from self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.eos_id,
            min_new_tokens=min_new_tokens,
            ban_ids=ban_ids or None,
        )

    # ---------- v4 tasks ---------- #

    def continue_ids(self, body_ids: list[int], header: list[int] = (),
                     bars: int | None = None, **gen_kwargs) -> Iterator[int]:
        """Continuation with an attribute header. `bars`: pad the prompt to
        its bar line so the answer starts on a downbeat, and stop after
        that many bars (v4 only; ignored otherwise)."""
        prompt = list(body_ids)
        if self.v4:
            from .sequence_format import continuation_prompt
            if bars:
                prompt = self.close_bar(prompt)
            prompt = continuation_prompt(self.sp, list(header), prompt)
            gen_kwargs.setdefault("stop_after_bars", bars)
        yield from self.generate_ids(prompt, **gen_kwargs)

    def accompany(self, cond_ids: list[int], bars: int, header: list[int] = (),
                  **gen_kwargs) -> Iterator[int]:
        """Write the other parts for `cond_ids` over exactly `bars` bars.
        Yields a self-contained token segment (decode it on its own; it
        starts at bar 0 of the window). Put the instruments you want added
        in `header` (see make_header)."""
        self._require_v4("accompany")
        from .sequence_format import accompaniment_prompt
        prompt = accompaniment_prompt(self.sp, list(header), self.pad_to_bars(cond_ids, bars))
        gen_kwargs.setdefault("max_new_tokens", 64 * bars + 64)
        # a target may end with EOS or the next document's BOS; SEP/MASK never
        gen_kwargs.setdefault("ban_ids", [self.sp.sep, self.sp.mask])
        yield from self.generate_ids(prompt, stop_after_bars=bars, **gen_kwargs)

    def infill(self, prefix_ids: list[int], suffix_ids: list[int], bars: int,
               header: list[int] = (), **gen_kwargs) -> Iterator[int]:
        """Write the `bars` bars that belong between `prefix_ids` and
        `suffix_ids` (each a bar-aligned segment). Yields a self-contained
        segment starting at bar 0 of the gap."""
        self._require_v4("infill")
        from .sequence_format import infill_prompt
        prompt = infill_prompt(self.sp, list(header), list(prefix_ids), list(suffix_ids))
        gen_kwargs.setdefault("max_new_tokens", 64 * bars + 64)
        gen_kwargs.setdefault("ban_ids", [self.sp.sep, self.sp.mask])
        yield from self.generate_ids(prompt, stop_after_bars=bars, **gen_kwargs)

    def split_bars(self, ids: list[int], at_bar: int, n_bars: int) -> tuple[list[int], list[int]]:
        """Cut a v4 token segment into (prefix, suffix) around bars
        [at_bar, at_bar+n_bars): the infill inputs for "redo bars i..j".
        Header/BOS tokens are dropped; the suffix is re-based so its first
        Bar token is bar 0 (Bar/Position are relative, so no retiming)."""
        self._require_v4("split_bars")
        body = [t for t in ids if t not in self.sp.header_ids and t != self.bos_id]
        edges = [i for i, t in enumerate(body) if t == self.bar_id]
        if at_bar + n_bars > len(edges):
            raise ValueError(f"segment has {len(edges)} bars")
        end = edges[at_bar + n_bars] if at_bar + n_bars < len(edges) else len(body)
        return body[:edges[at_bar]], body[end:]

    def stream_notes(
        self,
        prompt_ids: list[int],
        chunk_tokens: int = 16,
        tempo_bpm: float = 120.0,
        decode_new_only: bool = False,
        **gen_kwargs,
    ) -> Iterator[Note]:
        """
        Generate tokens; every `chunk_tokens` IDs, decode the cumulative sequence
        to a Score and emit any newly-completed notes.

        `tempo_bpm` controls how tick offsets are converted to seconds — pass the
        user's actual tempo so emitted notes sync to their performance. Since we
        train with `use_tempos=False`, the model's output ticks are tempo-agnostic
        (in beat units); we just rescale at decode time.
        """
        emitted = 0
        if "max_new_tokens" in gen_kwargs:
            prompt_ids, gen_kwargs["max_new_tokens"] = self.fit_to_context(
                prompt_ids, gen_kwargs["max_new_tokens"])
        # decode_new_only: the generated tokens are a self-contained segment
        # (v4 accompaniment / infill targets start at their own bar 0), so
        # decode them without the prompt
        buffer: list[int] = [] if decode_new_only else list(prompt_ids)
        n_in_buffer_since_decode = 0
        for tid in self.generate_ids(prompt_ids, **gen_kwargs):
            buffer.append(tid)
            n_in_buffer_since_decode += 1
            if n_in_buffer_since_decode < chunk_tokens:
                continue
            n_in_buffer_since_decode = 0
            score = self.tokenizer.decode(buffer)
            tpq = score.ticks_per_quarter
            seconds_per_tick = 60.0 / (tempo_bpm * tpq)
            notes_flat: list[Note] = []
            for track in score.tracks:
                program = int(track.program) if not track.is_drum else 128
                for n in track.notes:
                    notes_flat.append(Note(
                        pitch=int(n.pitch),
                        start=float(n.start) * seconds_per_tick,
                        end=float(n.start + n.duration) * seconds_per_tick,
                        velocity=int(n.velocity),
                        program=program,
                    ))
            notes_flat.sort(key=lambda x: (x.start, x.pitch))
            for note in notes_flat[emitted:]:
                yield note
            emitted = len(notes_flat)

    def generate_to_midi(
        self,
        prompt_ids: list[int],
        out_path: str | Path,
        tempo_bpm: float = 120.0,
        **gen_kwargs,
    ) -> list[int]:
        if "max_new_tokens" in gen_kwargs:
            prompt_ids, gen_kwargs["max_new_tokens"] = self.fit_to_context(
                prompt_ids, gen_kwargs["max_new_tokens"])
        new_ids = list(self.generate_ids(prompt_ids, **gen_kwargs))
        full_ids = list(prompt_ids) + new_ids
        score = self.tokenizer.decode(full_ids)
        if abs(tempo_bpm - 120.0) > 1e-6:
            from symusic import Tempo
            score.tempos = [Tempo(time=0, qpm=tempo_bpm)]
        score.dump_midi(Path(out_path))
        return new_ids


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--input-midi", required=True)
    parser.add_argument("--output-midi", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-new-tokens", type=int, default=0,
                        help="mask EOS for the first N generated tokens (guards "
                             "against prompts that look like a piece ending)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for a reproducible take")
    parser.add_argument("--tempo-bpm", type=float, default=None,
                        help="output tempo; defaults to input MIDI's tempo")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="inference precision (float16 is ~1.65x faster on CPU)")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "mlx", "torch"],
                        help="auto uses MLX when installed (Apple silicon), else torch")
    args = parser.parse_args()

    g = Generator(args.checkpoint, args.tokenizer, dtype=args.dtype,
                    backend=args.backend)
    prompt = g.encode_midi_file(args.input_midi)
    tempo = args.tempo_bpm if args.tempo_bpm else g.detect_tempo(args.input_midi)
    # v4: the header describes the prompt file and carries the tempo the
    # output will play at, so the model is told the clock it writes for
    prompt = [*g.make_header(args.input_midi, tempo=tempo), *prompt]
    new_ids = g.generate_to_midi(
        prompt, args.output_midi,
        tempo_bpm=tempo,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        min_new_tokens=args.min_new_tokens,
        seed=args.seed,
    )
    print(json.dumps({"prompt_tokens": len(prompt), "generated_tokens": len(new_ids),
                      "output": args.output_midi}))
