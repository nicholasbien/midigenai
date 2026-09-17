"""
Modal serving for midigenai (successor to openmusenet2/v2/modal_generate_v2.py).

Deploys as app `midigenai-serve`, class `MidiGen`:
- generate_batch(): one-shot, returns full prompt+continuation MIDI bytes
- stream_notes():   yields JSON-line note dicts as the model emits events

Checkpoints live in the `midigenai-models` Modal Volume, one subfolder per
version, mirroring the Hugging Face repo layout:

    /models/v4/ckpt_final.pt   /models/v4/tokenizer.json
    /models/v3/ckpt_final.pt   /models/v3/tokenizer.json

`MidiGen` takes the version as a Modal class parameter, so each version gets
its own container pool and its own memory snapshot. Callers pick one with
`Cls.from_name("midigenai-serve", "MidiGen")(version="v3")`; the default is
DEFAULT_VERSION.

Populate the volume straight from Hugging Face (server-side, no local
upload) after publishing a new version:
    modal run midigenai/modal_serve.py::sync_from_hub --version v4

Deploy:
    modal deploy midigenai/modal_serve.py
"""

# NOTE: no `from __future__ import annotations` here. Modal resolves
# `modal.parameter()` annotations at decoration time and cannot read them
# once PEP 563 turns them into strings. Union syntax below needs 3.10+,
# which both the image and every supported local interpreter satisfy.

import json

import modal
from modal import Image, Volume

VOLUME_NAME = "midigenai-models"
MODELS_ROOT = "/models"
CKPT_FILENAME = "ckpt_final.pt"
TOKENIZER_FILENAME = "tokenizer.json"

DEFAULT_VERSION = "v4"

# Versions this deployment will serve, keyed by the name the site sends as
# `model=`. The value is the subfolder in both the Hub repo and the volume.
# Anything not listed here is rejected rather than passed through, so a
# stray query param can't make the server look for an arbitrary path.
SERVED_VERSIONS = {
    "v4": "v4",
    "v3": "v3",
    "v2": "v2-100m",
}


def resolve_version(name: str | None) -> str:
    """Map a `model=` value to a volume subfolder, falling back to the default."""
    return SERVED_VERSIONS.get((name or "").strip(), SERVED_VERSIONS[DEFAULT_VERSION])

app = modal.App("midigenai-serve")

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

_base = Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "miditok",
    "symusic",
    "numpy",
)
# add_local_python_source has to come last in a chain, so the Hub variant
# branches off the shared base rather than extending the finished image.
image = _base.add_local_python_source("midigenai")
hub_image = _base.pip_install("huggingface_hub").add_local_python_source("midigenai")


@app.function(
    image=hub_image,
    volumes={MODELS_ROOT: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
)
def sync_from_hub(version: str = DEFAULT_VERSION, repo_id: str = "nicholasbien/midigenai"):
    """Copy one version's checkpoint + tokenizer from the Hub into the volume.

    Runs inside Modal, so the weights go Hub -> Modal directly and never
    travel through the local machine. Safe to re-run: it overwrites the
    version's folder in place.
    """
    import os
    import shutil
    from huggingface_hub import hf_hub_download

    subfolder = SERVED_VERSIONS.get(version, version)
    dest = os.path.join(MODELS_ROOT, subfolder)
    os.makedirs(dest, exist_ok=True)
    for filename in (CKPT_FILENAME, TOKENIZER_FILENAME):
        src = hf_hub_download(repo_id, filename, subfolder=subfolder)
        out = os.path.join(dest, filename)
        shutil.copyfile(src, out)
        print(f"{subfolder}/{filename}: {os.path.getsize(out) / 1e6:.1f} MB")
    volume.commit()
    return sorted(os.listdir(dest))


@app.function(image=image, volumes={MODELS_ROOT: volume})
def list_versions() -> dict:
    """What the volume actually holds, so a deploy can be checked without SSH."""
    import os
    out = {}
    for name in sorted(os.listdir(MODELS_ROOT)):
        folder = os.path.join(MODELS_ROOT, name)
        if os.path.isdir(folder):
            out[name] = {f: os.path.getsize(os.path.join(folder, f))
                         for f in sorted(os.listdir(folder))}
    return out


@app.cls(
    image=image,
    gpu="L4",  # pinned: modern, low per-kernel latency; "any" can hand out T4s
    # Per-version container pools (see `version` below) multiply GPU demand:
    # three versions exercised at once each claimed their own L4 and then sat
    # warm for the scaledown window. That filled the workspace GPU limit,
    # which is shared with training, and every request queued behind it until
    # Railway's ~120 s ceiling cut it off -- a site-wide 500 with no error
    # from this code. Cap the pool so a burst queues on a warm container
    # instead of claiming another GPU, and let idle versions go sooner.
    max_containers=3,
    scaledown_window=180,
    memory=32_768,
    cpu=4,
    timeout=180,
    volumes={"/models": volume},
    enable_memory_snapshot=True,
)
class MidiGen:
    # One container pool and one memory snapshot per version.
    version: str = modal.parameter(default=SERVED_VERSIONS[DEFAULT_VERSION])

    @modal.enter(snap=True)
    def load_cpu(self):
        """Runs once per version, then is checkpointed into the memory
        snapshot: later cold starts restore the loaded model instead of
        re-importing torch and re-reading the checkpoint."""
        import os
        from midigenai.generate import Generator
        import torch
        ckpt = os.path.join(MODELS_ROOT, self.version, CKPT_FILENAME)
        tokenizer = os.path.join(MODELS_ROOT, self.version, TOKENIZER_FILENAME)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"no checkpoint for version {self.version!r} at {ckpt}. "
                f"Run: modal run midigenai/modal_serve.py::sync_from_hub "
                f"--version {self.version}")
        self.gen = Generator(
            checkpoint_path=ckpt,
            tokenizer_path=tokenizer,
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
        sample rides along nearly free vs. two sequential generations.

        Sampling is hand-rolled here (Generator.generate_ids decodes one
        sequence at a time), so the v4 handling it would have applied is
        applied explicitly: `default_ban_ids` masks SEP / MASK / BOS out of
        the logits, and `postprocess` trims the leading empty bars that
        would otherwise reach the player as silence."""
        import torch
        gen = self.gen
        # Long uploads: cut the prompt so prompt + continuation fits the
        # context window (decoding past it crashes attention).
        prompt_ids, max_new_tokens = gen.fit_to_context(prompt_ids, max_new_tokens)
        if gen.bos_id is not None and (not prompt_ids or prompt_ids[0] != gen.bos_id):
            prompt_ids = [gen.bos_id, *prompt_ids]
        ban_ids = gen.default_ban_ids()
        model = gen.model
        ids = torch.tensor([prompt_ids] * n_samples, dtype=torch.long, device=gen.device)
        outs: list[list[int]] = [[] for _ in range(n_samples)]
        done = [False] * n_samples
        with torch.no_grad():
            logits, caches = model(ids)
            for _ in range(max_new_tokens):
                logits = logits[:, -1, :].float() / max(temperature, 1e-6)
                if ban_ids:
                    logits[:, ban_ids] = -float("inf")
                if top_k is not None and top_k < logits.size(-1):
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
                for b, tid in enumerate(next_ids[:, 0].tolist()):
                    if not done[b]:
                        # stop_ids (EOS, and SEP/MASK/BOS on v4) end a sample;
                        # postprocess drops them, so stopping here only saves work
                        if tid in gen.stop_ids:
                            done[b] = True
                        else:
                            outs[b].append(tid)
                if all(done):
                    break
                logits, caches = model(next_ids, kv_caches=caches)
        return [list(gen.postprocess(prompt_ids, out)) for out in outs]

    def _to_midi_bytes(self, full_ids: list[int], tempo_bpm: float) -> bytes:
        return self._score_to_bytes(self.gen.tokenizer.decode(full_ids), tempo_bpm)

    def _score_to_bytes(self, score, tempo_bpm: float) -> bytes:
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
        full_prompt = self.gen.encode_midi_bytes(midi_bytes)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)

        # A prompt too long for the context window is cut at the end; the
        # continuation follows the cut, so the returned MIDI is the kept
        # prompt + continuation. `prompt_end_seconds` tells the client where
        # the model's input ended so it can mark the cut on the original.
        prompt, max_new_tokens = self.gen.fit_to_context(full_prompt, max_new_tokens)
        truncated = len(prompt) < len(full_prompt)
        prompt_score = self.gen.tokenizer.decode(list(prompt))
        tpq = max(prompt_score.ticks_per_quarter, 1)
        prompt_end_seconds = prompt_score.end() / tpq * 60.0 / tempo_bpm

        sample_ids = self._batched_generate(
            prompt, n_samples, max_new_tokens, temperature, top_k)
        midis = [self._to_midi_bytes(list(prompt) + ids, tempo_bpm)
                 for ids in sample_ids]

        return {
            "prompt_tokens": len(prompt),
            "prompt_tokens_total": len(full_prompt),
            "prompt_truncated": truncated,
            "prompt_end_seconds": prompt_end_seconds,
            "generated_tokens": [len(ids) for ids in sample_ids],
            "tempo_bpm": tempo_bpm,
            "midi": midis[0],   # backward-compatible single-sample field
            "midis": midis,
        }

    @modal.method()
    def accompany_batch(
        self,
        midi_bytes: bytes,
        bars: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
        n_samples: int = 1,
        tempo_bpm: float | None = None,
        instruments: list[str] | None = None,
    ) -> dict:
        """Write parts to go *with* the upload rather than after it.

        Continuation extends the prompt in time; accompaniment fills the same
        `bars` bars with a different instrument and the result is the two
        stacked. The upload is narrowed to its densest track, since the model
        is trained to answer a single part, and each returned MIDI is that
        condition overlaid with one generated answer.
        """
        from io import BytesIO

        from symusic import Score

        from midigenai.attributes import header_for_score
        from midigenai.data.v4_docs import _subscore, _window, bar_edges, trim_leading
        from midigenai.generate import densest_track, overlay
        from midigenai.tokenizer import normalize_drums

        if not self.gen.v4:
            raise ValueError(
                f"accompaniment needs a v4 checkpoint; {self.version!r} is not one")

        score = Score.from_midi(BytesIO(midi_bytes).read())
        normalize_drums(score, "upload.mid")
        score = trim_leading(score)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)

        edges = bar_edges(score)
        available = max(0, len(edges) - 1)
        if available < 1:
            raise ValueError("upload has no complete bar to accompany")
        bars = max(1, min(bars, available))
        window = _window(score, edges[0], edges[bars])

        cond_index = densest_track(window)
        condition = _subscore(window, [cond_index])
        if not sum(len(tr.notes) for tr in condition.tracks):
            raise ValueError("the chosen track has no notes in the first bars")

        cond_ids = self.gen.tokenizer(condition).ids
        # `instruments` = families to ADD (INSTRUMENT_FAMILIES + "Drums").
        # Training headers list condition + target, so a header naming only
        # the condition's instrument reads as "more of the same": on v4 a
        # piano condition with no request came back 100% piano. With a
        # request the model follows it, and banning drum tokens unless drums
        # were asked for roughly doubles the requested pitched family's share
        # (bass 21% -> 36%, strings 36% -> 45%) by removing the uninvited kit.
        from midigenai.attributes import INSTRUMENT_FAMILIES, family_of
        cond_fams = sorted({family_of(tr.program, tr.is_drum)
                            for tr in condition.tracks if len(tr.notes)})
        asked = [i for i in (instruments or []) if i in INSTRUMENT_FAMILIES or i == "Drums"]
        if asked:
            header = self.gen.make_header(instruments=cond_fams + asked)
        else:   # no request: describe the whole upload, as before
            header = self.gen.sp.header_ids_for(
                self.gen.tokenizer, header_for_score(window))
        want_drums = (not asked) or "Drums" in asked or "Drums" in cond_fams
        ban = [self.gen.sp.sep, self.gen.sp.mask]
        if not want_drums:
            V = self.gen.tokenizer.vocab
            ban += [v for k, v in V.items() if k.startswith("PitchDrum_") or k == "Program_-1"]

        midis, note_counts = [], []
        for _ in range(n_samples):
            new_ids = list(self.gen.accompany(
                cond_ids, bars, header=header, ban_ids=ban,
                temperature=temperature, top_k=top_k))
            answer = self.gen.tokenizer.decode(new_ids)
            note_counts.append(sum(len(tr.notes) for tr in answer.tracks))
            midis.append(self._score_to_bytes(overlay(condition, answer), tempo_bpm))

        beats_per_bar = 4.0
        if window.time_signatures:
            ts = window.time_signatures[0]
            beats_per_bar = ts.numerator * 4.0 / ts.denominator
        return {
            "bars": bars,
            "bars_available": available,
            "condition_track": cond_index,
            "condition_track_name": window.tracks[cond_index].name or "",
            "track_names": [tr.name or "" for tr in window.tracks],
            "condition_notes": sum(len(tr.notes) for tr in condition.tracks),
            "condition_instruments": cond_fams,
            "instruments_requested": asked,
            "generated_notes": note_counts,
            "tempo_bpm": tempo_bpm,
            "window_seconds": bars * beats_per_bar * 60.0 / tempo_bpm,
            "midi": midis[0],
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
