"""
Modal serving for midigenai (successor to openmusenet2/v2/modal_generate_v2.py).

Deploys as app `midigenai-serve`, class `MidiGen`:
- generate_batch(): one-shot, returns full prompt+continuation MIDI bytes
- stream_notes():   yields JSON-line note dicts as the model emits events

The checkpoint + tokenizer live in the existing `openmusenet2-v2` Modal Volume
(kept: it's just storage, and it already holds the deployed weights). On
retrain, upload the new files:
    modal volume put openmusenet2-v2 ckpt_final.pt
    modal volume put openmusenet2-v2 tokenizer.json

Deploy:
    modal deploy midigenai/modal_serve.py
"""

from __future__ import annotations

import json

import modal
from modal import Image, Volume

VOLUME_NAME = "openmusenet2-v2"  # storage name predates the repo rename
CKPT_PATH = "/models/ckpt_final.pt"
TOKENIZER_PATH = "/models/tokenizer.json"

app = modal.App("midigenai-serve")

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "miditok",
        "symusic",
        "numpy",
    )
    .add_local_python_source("midigenai")
)


@app.cls(
    image=image,
    gpu="L4",  # pinned: modern, low per-kernel latency; "any" can hand out T4s
    scaledown_window=600,
    memory=32_768,
    cpu=4,
    timeout=180,
    volumes={"/models": volume},
    enable_memory_snapshot=True,
)
class MidiGen:
    @modal.enter(snap=True)
    def load_cpu(self):
        """Runs once, then is checkpointed into the memory snapshot: later cold
        starts restore the loaded model instead of re-importing torch and
        re-reading the checkpoint."""
        from midigenai.generate import Generator
        import torch
        self.gen = Generator(
            checkpoint_path=CKPT_PATH,
            tokenizer_path=TOKENIZER_PATH,
            device=torch.device("cpu"),  # snapshot is CPU-only; GPU attaches after restore
        )

    @modal.enter(snap=False)
    def to_gpu(self):
        import torch
        if torch.cuda.is_available():
            self.gen.device = torch.device("cuda")
            self.gen.model = self.gen.model.to(self.gen.device)

    def _batched_generate(
        self,
        prompt_ids: list[int],
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[list[int]]:
        """Decode n_samples continuations in one batch on the GPU — a second
        sample rides along nearly free vs. two sequential generations."""
        import torch
        gen = self.gen
        if gen.bos_id is not None and (not prompt_ids or prompt_ids[0] != gen.bos_id):
            prompt_ids = [gen.bos_id, *prompt_ids]
        model = gen.model
        ids = torch.tensor([prompt_ids] * n_samples, dtype=torch.long, device=gen.device)
        outs: list[list[int]] = [[] for _ in range(n_samples)]
        done = [False] * n_samples
        with torch.no_grad():
            logits, caches = model(ids)
            for _ in range(max_new_tokens):
                logits = logits[:, -1, :].float() / max(temperature, 1e-6)
                if top_k is not None and top_k < logits.size(-1):
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
                for b, tid in enumerate(next_ids[:, 0].tolist()):
                    if not done[b]:
                        if gen.eos_id is not None and tid == gen.eos_id:
                            done[b] = True
                        else:
                            outs[b].append(tid)
                if all(done):
                    break
                logits, caches = model(next_ids, kv_caches=caches)
        return outs

    def _to_midi_bytes(self, full_ids: list[int], tempo_bpm: float) -> bytes:
        score = self.gen.tokenizer.decode(full_ids)
        if abs(tempo_bpm - 120.0) > 1e-6:
            from symusic import Tempo
            score.tempos = [Tempo(time=0, qpm=tempo_bpm)]
        # Write to bytes via a tempfile (symusic Score.dump_midi needs a path)
        from tempfile import NamedTemporaryFile
        from pathlib import Path
        with NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp = f.name
        try:
            score.dump_midi(tmp)
            return Path(tmp).read_bytes()
        finally:
            Path(tmp).unlink(missing_ok=True)

    @modal.method()
    def generate_batch(
        self,
        midi_bytes: bytes,
        max_new_tokens: int = 512,
        temperature: float = 1.2,
        top_k: int = 50,
        tempo_bpm: float | None = None,
        n_samples: int = 1,
    ) -> dict:
        prompt = self.gen.encode_midi_bytes(midi_bytes)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)

        sample_ids = self._batched_generate(
            prompt, n_samples, max_new_tokens, temperature, top_k)
        midis = [self._to_midi_bytes(list(prompt) + ids, tempo_bpm)
                 for ids in sample_ids]

        return {
            "prompt_tokens": len(prompt),
            "generated_tokens": [len(ids) for ids in sample_ids],
            "tempo_bpm": tempo_bpm,
            "midi": midis[0],   # backward-compatible single-sample field
            "midis": midis,
        }

    @modal.method(is_generator=True)
    def stream_notes(
        self,
        midi_bytes: bytes,
        max_new_tokens: int = 512,
        temperature: float = 1.2,
        top_k: int = 50,
        chunk_tokens: int = 16,
        tempo_bpm: float | None = None,
    ):
        prompt = self.gen.encode_midi_bytes(midi_bytes)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)
        for note in self.gen.stream_notes(
            prompt,
            chunk_tokens=chunk_tokens,
            tempo_bpm=tempo_bpm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            yield json.dumps({
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "velocity": note.velocity,
                "program": note.program,
            }) + "\n"
