"""
v2 inference: load a trained checkpoint, encode MIDI input, stream a continuation.

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
from typing import Iterator

import torch

from .model_v2 import ModelConfig, MusicTransformer
from .tokenizer_v2 import build_tokenizer, load_tokenizer


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


class V2Generator:
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
        per-op dispatch overhead: 577 t/s decode vs 267 on CPU fp16, and 50 ms
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
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["model_config"]
        cfg = ModelConfig(**cfg_dict)
        # RoPE extrapolates, so we can serve at longer context than training without retraining.
        cfg.max_seq_len = max(cfg.max_seq_len, inference_seq_len)

        if backend == "mlx":
            import mlx.core as mx
            from .model_v2_mlx import load_from_torch_state_dict
            mlx_dtype = {
                torch.float16: mx.float16,
                torch.bfloat16: mx.bfloat16,
                torch.float32: mx.float32,
            }[self.dtype]
            self.device = None  # MLX manages placement (unified memory)
            self.model = load_from_torch_state_dict(ckpt["model"], cfg, dtype=mlx_dtype)
        else:
            self.device = device or _auto_device()
            self.model = MusicTransformer(cfg).to(self.device).eval()
            self.model.load_state_dict(ckpt["model"])
            self.model = self.model.to(self.dtype)

        self.tokenizer = (
            load_tokenizer(tokenizer_path) if tokenizer_path else build_tokenizer()
        )
        self.bos_id = self._special_id("BOS_None", "BOS")
        self.eos_id = self._special_id("EOS_None", "EOS")

    def _special_id(self, *candidates: str) -> int | None:
        for c in candidates:
            if c in self.tokenizer.vocab:
                return self.tokenizer.vocab[c]
        return None

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
            return V2Generator.detect_tempo(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # ---------- generation ---------- #

    def generate_ids(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Iterator[int]:
        if self.bos_id is not None and (not prompt_ids or prompt_ids[0] != self.bos_id):
            prompt_ids = [self.bos_id, *prompt_ids]
        if self.backend == "mlx":
            yield from self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=self.eos_id,
            )
            return
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        yield from self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.eos_id,
        )

    def stream_notes(
        self,
        prompt_ids: list[int],
        chunk_tokens: int = 16,
        tempo_bpm: float = 120.0,
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
        buffer: list[int] = list(prompt_ids)
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
    parser.add_argument("--tempo-bpm", type=float, default=None,
                        help="output tempo; defaults to input MIDI's tempo")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="inference precision (float16 is ~1.65x faster on CPU)")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "mlx", "torch"],
                        help="auto uses MLX when installed (Apple silicon), else torch")
    args = parser.parse_args()

    g = V2Generator(args.checkpoint, args.tokenizer, dtype=args.dtype,
                    backend=args.backend)
    prompt = g.encode_midi_file(args.input_midi)
    tempo = args.tempo_bpm if args.tempo_bpm else g.detect_tempo(args.input_midi)
    new_ids = g.generate_to_midi(
        prompt, args.output_midi,
        tempo_bpm=tempo,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(json.dumps({"prompt_tokens": len(prompt), "generated_tokens": len(new_ids),
                      "output": args.output_midi}))
