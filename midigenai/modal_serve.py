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
    gpu="any",
    scaledown_window=300,
    memory=32_768,
    cpu=4,
    timeout=180,
    volumes={"/models": volume},
)
class MidiGen:
    @modal.enter()
    def load(self):
        from midigenai.generate import Generator
        self.gen = Generator(
            checkpoint_path=CKPT_PATH,
            tokenizer_path=TOKENIZER_PATH,
        )

    @modal.method()
    def generate_batch(
        self,
        midi_bytes: bytes,
        max_new_tokens: int = 512,
        temperature: float = 1.2,
        top_k: int = 50,
        tempo_bpm: float | None = None,
    ) -> dict:
        prompt = self.gen.encode_midi_bytes(midi_bytes)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)

        new_ids = list(self.gen.generate_ids(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ))
        full_ids = list(prompt) + new_ids
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
            midi_out = Path(tmp).read_bytes()
        finally:
            Path(tmp).unlink(missing_ok=True)

        return {
            "prompt_tokens": len(prompt),
            "generated_tokens": len(new_ids),
            "tempo_bpm": tempo_bpm,
            "midi": midi_out,
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
