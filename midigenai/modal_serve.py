"""
Modal serving for midigenai (successor to openmusenet2/v2/modal_generate_v2.py).

Deploys as app `midigenai-serve`, class `MidiGen`:
- generate_batch(): one-shot, returns full prompt+continuation MIDI bytes
- accompany_batch(): parts to play with the upload
- infill_batch():  regenerate a span of bars inside the upload
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

Every call is also logged to the `midigenai-generations` Volume (see
log_generation): one folder per request under a UTC date, holding the prompt
MIDI, each returned MIDI and a request.json with the parameters and result
metadata. Pull a day down with
    modal volume get midigenai-generations 2026-09-29 ./generations

Deploy:
    modal deploy midigenai/modal_serve.py
"""

# NOTE: no `from __future__ import annotations` here. Modal resolves
# `modal.parameter()` annotations at decoration time and cannot read them
# once PEP 563 turns them into strings. Union syntax below needs 3.10+,
# which both the image and every supported local interpreter satisfy.

import json
import os
import uuid
from datetime import datetime, timezone

import modal
from modal import Image, Volume

VOLUME_NAME = "midigenai-models"
MODELS_ROOT = "/models"
CKPT_FILENAME = "ckpt_final.pt"
TOKENIZER_FILENAME = "tokenizer.json"

DEFAULT_VERSION = "v5-rl"

# Every request and what it returned, one folder per call (log_generation).
GENERATIONS_VOLUME_NAME = "midigenai-generations"
GENERATIONS_ROOT = "/generations"

# Versions this deployment will serve, keyed by the name the site sends as
# `model=`. The value is the subfolder in both the Hub repo and the volume.
# Anything not listed here is rejected rather than passed through, so a
# stray query param can't make the server look for an arbitrary path.
SERVED_VERSIONS = {
    "v5-rl": "v5-rl",   # v5 after GRPO; the default
    "v5": "v5",         # v5 base
    "v4": "v4",
    "v4-large": "v4-large",
    "v3": "v3",
    "v2": "v2-100m",
}

# Volume subfolders that can answer /api/accompany: accompaniment is a v4
# document type (Task_accomp + SEP), and v2/v3 were never trained on it.
# These are SERVED_VERSIONS *values*, since that is what resolve_version
# hands back -- for the v4 line the key and the folder are the same string.
# Kept next to the allowlist so the server and the site's mode lock can't
# drift apart; the health endpoint publishes it.
ACCOMPANIMENT_VERSIONS = ("v5-rl", "v5", "v4", "v4-large")
# Span infill is the other v4 document type (Task_infill + MASK + SEP), so the
# same checkpoints answer /api/infill.
INFILL_VERSIONS = ACCOMPANIMENT_VERSIONS


# An accompaniment take with no notes at all is resampled up to this many
# times. Since #73 keeps leading empty bars in place, a take that stays
# silent for the whole window comes back empty instead of being slid
# forward (1 take in 12 on the live v5-rl right after that deploy). A
# stopgap while the cause is measured; see accompany_batch.
ACCOMPANY_EMPTY_RETRIES = 2


def first_nonempty(draw, retries: int = ACCOMPANY_EMPTY_RETRIES):
    """Call `draw()` -> (result, n_notes) until a result has notes, at most
    1 + `retries` times. Returns (result, n_notes, draws). The last draw is
    returned even if it is empty: an honest empty take beats an error."""
    for i in range(retries + 1):
        result, n = draw()
        if n:
            break
    return result, n, i + 1


def resolve_version(name: str | None) -> str:
    """Map a `model=` value to a volume subfolder, falling back to the default."""
    return SERVED_VERSIONS.get((name or "").strip(), SERVED_VERSIONS[DEFAULT_VERSION])


def accompaniment_header(window, cond_index, instrument: str | None = None):
    """(header token names, family asked for) for an accompaniment document.

    The v4 header names what the finished document contains -- condition plus
    target, the way data/v4_docs.py builds it -- so asking for a part means
    listing the condition's own family alongside the requested one. With no
    request the header keeps the families the upload carries, which is what
    the model is left to interpret.

    Module level, and taking the window rather than reading it off the class,
    so the header the model is steered with can be checked without a GPU.
    """
    from midigenai.attributes import (
        family_of, header_for_score, is_auto_instrument, resolve_family,
        with_instruments,
    )

    names = header_for_score(window)
    if is_auto_instrument(instrument):
        return names, None
    family = resolve_family(instrument)
    if family is None:
        raise ValueError(f"unknown instrument {instrument!r}")
    idxs = cond_index if isinstance(cond_index, (list, tuple)) else [cond_index]
    cond_fams = []
    for i in idxs:
        f = family_of(window.tracks[i].program, window.tracks[i].is_drum)
        if f not in cond_fams:
            cond_fams.append(f)
    return with_instruments(names, [*cond_fams, family]), family


def log_generation(root: str, method: str, version: str, params: dict,
                   prompt_midi: bytes, midis: list[bytes] = (),
                   result: dict | None = None, notes: list[dict] | None = None,
                   client: str | None = None, error: str | None = None,
                   now: datetime | None = None) -> str | None:
    """Write one request to `root`/YYYY-MM-DD/<HHMMSS>_<id>_<method>/.

    The folder holds prompt.mid, out_<i>.mid per returned sample (or
    notes.json for stream_notes) and request.json: method, version, client,
    params, the result's metadata (everything but the MIDI bytes) and the
    error if the call failed. One folder per call rather than a shared JSONL
    because containers write the Volume concurrently and a Volume keeps the
    last writer of a file, not both.

    Best effort: logging must never fail or slow the request it records, so
    any error here is printed and swallowed. Returns the folder, or None.
    Module level, taking the root, so it can be checked without Modal.
    """
    try:
        now = now or datetime.now(timezone.utc)
        rid = uuid.uuid4().hex[:12]
        folder = os.path.join(root, f"{now:%Y-%m-%d}", f"{now:%H%M%S}_{rid}_{method}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "prompt.mid"), "wb") as f:
            f.write(prompt_midi)
        for i, m in enumerate(midis):
            with open(os.path.join(folder, f"out_{i}.mid"), "wb") as f:
                f.write(m)
        if notes is not None:
            with open(os.path.join(folder, "notes.json"), "w") as f:
                json.dump(notes, f)
        meta = {k: v for k, v in (result or {}).items() if k not in ("midi", "midis")}
        record = {
            "id": rid, "time": now.isoformat(), "method": method,
            "version": version, "client": client, "params": params,
            "result": meta, "outputs": len(midis) if notes is None else len(notes),
            "error": error,
        }
        with open(os.path.join(folder, "request.json"), "w") as f:
            json.dump(record, f, indent=1, default=str)
        return folder
    except Exception as e:  # noqa: BLE001
        print(f"generation log failed: {type(e).__name__}: {e}")
        return None


def accompaniment_ban_ids(tokenizer, sp, family: str | None, cond_has_drums: bool) -> list[int]:
    """Token ids an accompaniment answer may not use, given the family asked
    for (None = auto) and whether the condition already has drums.

    Measured on v5-rl through this route (48 held-out presets, 2 takes each),
    the header alone does not hold the answer to the request:
      * instrument=drums over a condition with no drums produced drums in
        only 25 of 54 takes -- the rest were pitched parts. Asking for drums
        now bans every pitched note (Pitch_*, every Program_ but -1).
      * instrument=auto added a kit nobody asked for in 18 of 54 takes.
        Uninvited drums are banned unless the condition has drums or drums
        were asked for -- what pairgen and the accompaniment probe were built
        on (pairgen.make_accompany_pair), and what the docs said this route
        already did. A pitched request (bass, piano, ...) added a kit in only
        1 of 54, so the ban changes little there.
    """
    from midigenai.attributes import DRUMS
    vocab = tokenizer.vocab
    ban = [t for t in (sp.sep, sp.mask) if t is not None]
    if family == DRUMS:
        ban += [i for k, i in vocab.items()
                if k.startswith("Pitch_") or (k.startswith("Program_") and k != "Program_-1")]
    elif not cond_has_drums:
        ban += [i for k, i in vocab.items() if k.startswith("PitchDrum_") or k == "Program_-1"]
    return sorted(ban)


app = modal.App("midigenai-serve")

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Changes reach the Volume by Modal's background commit (every few seconds
# and at container shutdown), so a request never waits on a commit.
generations_volume = Volume.from_name(GENERATIONS_VOLUME_NAME, create_if_missing=True)

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


@app.function(
    image=image,
    volumes={MODELS_ROOT: volume,
             "/runs": modal.Volume.from_name("midigenai-runs"),
             "/corpus": modal.Volume.from_name("midigenai-corpus")},
    timeout=1800,
    memory=16_384,
)
def publish_from_run(run_name: str, version: str, corpus: str,
                     checkpoint: str = "ckpt_final.pt") -> dict:
    """Copy a training run's checkpoint into the models volume as `version`,
    with optimizer state stripped, plus the tokenizer from `corpus`.

    Runs inside Modal, so the checkpoint never travels through a laptop. A
    training checkpoint carries AdamW moments that serving never reads --
    for the 113M that is 1365 MB of which 457 MB is weights -- and pulling
    the full file over a home connection stalled at 1.3 GB for three hours.
    The resumable copy stays on the runs volume; only the slim one is served.
    """
    import os
    import shutil
    import torch

    src = os.path.join("/runs", run_name, checkpoint)
    tok_src = os.path.join("/corpus", corpus, TOKENIZER_FILENAME)
    for path in (src, tok_src):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    dest = os.path.join(MODELS_ROOT, version)
    os.makedirs(dest, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in ckpt.items() if k != "optimizer"}
    out = os.path.join(dest, CKPT_FILENAME)
    torch.save(slim, out)
    shutil.copyfile(tok_src, os.path.join(dest, TOKENIZER_FILENAME))
    volume.commit()
    return {
        "version": version,
        "full_mb": os.path.getsize(src) // 1_000_000,
        "slim_mb": os.path.getsize(out) // 1_000_000,
        "step": slim.get("step"),
        "vocab": slim.get("model_config", {}).get("vocab_size"),
    }

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
    volumes={"/models": volume, GENERATIONS_ROOT: generations_volume},
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
        min_new_tokens: int = 0,
    ) -> list[list[int]]:
        """Decode n_samples continuations in one batch on the GPU — a second
        sample rides along nearly free vs. two sequential generations.

        Sampling is hand-rolled here (Generator.generate_ids decodes one
        sequence at a time), so the v4 handling it would have applied is
        applied explicitly: `default_ban_ids` masks SEP / MASK / BOS out of
        the logits, `postprocess` trims the leading empty bars that would
        otherwise reach the player as silence, and `min_new_tokens` holds
        every stop token back for the first N steps. Without that last one a
        prompt whose voices all stop together (an excerpt cut on a bar line)
        reads as a finished piece: P(EOS) as the first token is 0.5-0.93 on
        such prompts, and the site got back its own prompt, unchanged."""
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
            stop_list = sorted(gen.stop_ids)
            for step in range(max_new_tokens):
                logits = logits[:, -1, :].float() / max(temperature, 1e-6)
                if ban_ids:
                    logits[:, ban_ids] = -float("inf")
                if step < min_new_tokens and stop_list:
                    logits[:, stop_list] = -float("inf")
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

    def _log(self, method: str, params: dict, midi_bytes: bytes, **kw) -> None:
        log_generation(GENERATIONS_ROOT, method, self.version, params, midi_bytes, **kw)

    def _header_for_bytes(self, midi_bytes: bytes, tempo_bpm: float) -> list[int]:
        """v4 attribute header for an upload: describes the prompt and
        carries the tempo the answer will play at. [] on older checkpoints."""
        if not self.gen.v4:
            return []
        from io import BytesIO

        from symusic import Score

        from midigenai.attributes import header_for_score
        score = Score.from_midi(BytesIO(midi_bytes).read())
        return self.gen.sp.header_ids_for(
            self.gen.tokenizer, header_for_score(score, tempo=tempo_bpm))

    def _to_midi_bytes(self, full_ids: list[int], tempo_bpm: float | None) -> bytes:
        return self._score_to_bytes(self.gen.tokenizer.decode(full_ids), tempo_bpm)

    @staticmethod
    def _source_tempo(midi_bytes: bytes, tempo_bpm: float | None) -> float | None:
        """The tempo to stamp on the output: the caller's, else the upload's
        own tempo event, else None (the upload had none)."""
        if tempo_bpm is not None:
            return tempo_bpm
        from symusic import Score
        tempos = Score.from_midi(midi_bytes).tempos
        return float(tempos[0].qpm) if len(tempos) else None

    def _score_to_bytes(self, score, tempo_bpm: float | None) -> bytes:
        # Tempo in the output mirrors the input: if the upload (or caller)
        # had a tempo, write exactly that; if it had none, write none, so the
        # file plays at the MIDI default like the upload did. Never leave the
        # decoder's own tempo in place -- on the serving image a 120 BPM
        # prompt came back stamped 18.62 (6.4x slow, 0:20 -> 1:48), and an
        # old "skip if 120" shortcut here let exactly that through.
        from symusic import Tempo
        score.tempos = [Tempo(time=0, qpm=tempo_bpm)] if tempo_bpm else []
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
        client: str | None = None,
    ) -> dict:
        params = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                  "top_k": top_k, "tempo_bpm": tempo_bpm, "n_samples": n_samples}
        try:
            result = self._generate_batch(midi_bytes, max_new_tokens, temperature,
                                          top_k, tempo_bpm, n_samples)
        except Exception as e:
            self._log("generate", params, midi_bytes, client=client,
                      error=f"{type(e).__name__}: {e}")
            raise
        self._log("generate", params, midi_bytes, midis=result["midis"],
                  result=result, client=client)
        return result

    def _generate_batch(self, midi_bytes, max_new_tokens, temperature, top_k,
                        tempo_bpm, n_samples) -> dict:
        full_prompt = self.gen.encode_midi_bytes(midi_bytes)
        out_tempo = self._source_tempo(midi_bytes, tempo_bpm)   # stamped on the file, or None
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)  # 120 when absent: the MIDI default
        full_prompt = [*self._header_for_bytes(midi_bytes, tempo_bpm), *full_prompt]

        # A prompt too long for the context window is cut at the end; the
        # continuation follows the cut, so the returned MIDI is the kept
        # prompt + continuation. `prompt_end_seconds` tells the client where
        # the model's input ended so it can mark the cut on the original.
        prompt, max_new_tokens = self.gen.fit_to_context(full_prompt, max_new_tokens)
        truncated = len(prompt) < len(full_prompt)
        prompt_score = self.gen.tokenizer.decode(list(prompt))
        tpq = max(prompt_score.ticks_per_quarter, 1)
        prompt_end_seconds = prompt_score.end() / tpq * 60.0 / tempo_bpm

        # Floor on how much the model must write before it may stop: a quarter
        # of the budget, capped at 128 tokens (~2 bars of a busy part).
        sample_ids = self._batched_generate(
            prompt, n_samples, max_new_tokens, temperature, top_k,
            min_new_tokens=min(128, max_new_tokens // 4))
        midis = [self._to_midi_bytes(list(prompt) + ids, out_tempo)
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
        instrument: str | None = None,
        track=None,
        client: str | None = None,
    ) -> dict:
        params = {"bars": bars, "temperature": temperature, "top_k": top_k,
                  "n_samples": n_samples, "tempo_bpm": tempo_bpm,
                  "instrument": instrument, "track": track}
        try:
            result = self._accompany_batch(midi_bytes, bars, temperature, top_k,
                                           n_samples, tempo_bpm, instrument, track)
        except Exception as e:
            self._log("accompany", params, midi_bytes, client=client,
                      error=f"{type(e).__name__}: {e}")
            raise
        self._log("accompany", params, midi_bytes, midis=result["midis"],
                  result=result, client=client)
        return result

    def _accompany_batch(self, midi_bytes, bars, temperature, top_k, n_samples,
                         tempo_bpm, instrument, track) -> dict:
        """Write parts to go *with* the upload rather than after it.

        Continuation extends the prompt in time; accompaniment fills the same
        `bars` bars with a different instrument and the result is the two
        stacked. The upload is narrowed to its densest track, since the model
        is trained to answer a single part, and each returned MIDI is that
        condition overlaid with one generated answer.

        `instrument` asks for a particular part -- "bass", "drums", "piano",
        or any name attributes.resolve_family understands. It is written into
        the attribute header, which names what the finished document holds,
        so the request goes in as the condition's own family plus the one
        asked for. Left unset, the header keeps the families the upload
        already has and the model picks the part itself.
        """
        from io import BytesIO

        from symusic import Score

        from midigenai.data.v4_docs import _subscore, _window, bar_edges, trim_leading
        from midigenai.generate import densest_track, overlay
        from midigenai.tokenizer import normalize_drums

        if not self.gen.v4:
            raise ValueError(
                f"accompaniment needs a v4 checkpoint; {self.version!r} is not one")

        # Window and condition-track choice come from accompany_tracks so the
        # indices the site's picker shows (/api/tracks) are the ones used here.
        from midigenai.accompany_tracks import choose_tracks, parse_track_spec, prepare_window
        window, bars, available = prepare_window(midi_bytes, bars)
        out_tempo = self._source_tempo(midi_bytes, tempo_bpm)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)

        # `track` may be one index, a list, "0,2" or "all": several tracks can
        # be the condition at once. Training conditions are 1 track 70% of
        # the time and 2 tracks 30%; more than 2 is outside what it saw.
        cond_idx = choose_tracks(window, parse_track_spec(track))
        cond_index = cond_idx[0] if len(cond_idx) == 1 else cond_idx
        condition = _subscore(window, cond_idx)
        if not sum(len(tr.notes) for tr in condition.tracks):
            raise ValueError("the chosen track has no notes in the first bars")

        cond_ids = self.gen.tokenizer(condition).ids

        names, family = accompaniment_header(window, cond_index, instrument)
        # Tempo_ rides along with the instrument-choice header: the clock the
        # caller passes (or the file's own) becomes the family token, same as
        # header_for_score(window, tempo=...) does for continuation.
        from midigenai.attributes import tempo_tokens
        if not any(n.startswith("Tempo_") for n in names):
            names = [*names, *tempo_tokens(window, tempo=tempo_bpm)]
        header = self.gen.sp.header_ids_for(self.gen.tokenizer, names)
        ban_ids = accompaniment_ban_ids(
            self.gen.tokenizer, self.gen.sp, family,
            cond_has_drums=any(tr.is_drum and len(tr.notes) for tr in condition.tracks))

        def draw():
            new_ids = list(self.gen.accompany(
                cond_ids, bars, header=header, ban_ids=ban_ids,
                temperature=temperature, top_k=top_k))
            answer = self.gen.tokenizer.decode(new_ids)
            return answer, sum(len(tr.notes) for tr in answer.tracks)

        midis, note_counts, draws = [], [], []
        for _ in range(n_samples):
            answer, n, k = first_nonempty(draw)
            note_counts.append(n)
            draws.append(k)
            midis.append(self._score_to_bytes(overlay(condition, answer), out_tempo))

        beats_per_bar = 4.0
        if window.time_signatures:
            ts = window.time_signatures[0]
            beats_per_bar = ts.numerator * 4.0 / ts.denominator
        return {
            "bars": bars,
            "bars_available": available,
            "condition_track": cond_idx[0],                        # first, for older clients
            "condition_track_name": window.tracks[cond_idx[0]].name or "",
            "condition_tracks": cond_idx,
            "condition_track_names": [window.tracks[i].name or "" for i in cond_idx],
            "track_names": [tr.name or "" for tr in window.tracks],
            "condition_notes": sum(len(tr.notes) for tr in condition.tracks),
            "generated_notes": note_counts,
            "draws": draws,                     # >1 = an empty take was resampled
            "instrument": family,
            "header": names,
            "tempo_bpm": tempo_bpm,
            "window_seconds": bars * beats_per_bar * 60.0 / tempo_bpm,
            "midi": midis[0],
            "midis": midis,
        }

    @modal.method()
    def infill_batch(
        self,
        midi_bytes: bytes,
        start: int,
        bars: int = 2,
        temperature: float = 1.0,
        top_k: int = 50,
        n_samples: int = 1,
        tempo_bpm: float | None = None,
        client: str | None = None,
    ) -> dict:
        params = {"start": start, "bars": bars, "temperature": temperature,
                  "top_k": top_k, "n_samples": n_samples, "tempo_bpm": tempo_bpm}
        try:
            result = self._infill_batch(midi_bytes, start, bars, temperature,
                                        top_k, n_samples, tempo_bpm)
        except Exception as e:
            self._log("infill", params, midi_bytes, client=client,
                      error=f"{type(e).__name__}: {e}")
            raise
        self._log("infill", params, midi_bytes, midis=result["midis"],
                  result=result, client=client)
        return result

    def _infill_batch(self, midi_bytes, start, bars, temperature, top_k,
                      n_samples, tempo_bpm) -> dict:
        """Regenerate bars [start, start+bars) of the upload and keep the rest.

        The model sees up to 16 bars around the span (the bars before it and
        the bars after it) and writes the missing bars; each returned MIDI is
        the whole upload with that span replaced. See infill_span.py for the
        span limits (the shape v4/v5 were trained on) and the splice."""
        from midigenai import infill_span

        if not self.gen.v4:
            raise ValueError(f"infill needs a v4 checkpoint; {self.version!r} is not one")
        out_tempo = self._source_tempo(midi_bytes, tempo_bpm)
        if tempo_bpm is None:
            tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)
        p, scores, counts = infill_span.run(
            self.gen, midi_bytes, start, bars, n_samples=n_samples,
            tempo_bpm=tempo_bpm, temperature=temperature, top_k=top_k)
        tpq = max(p.score.ticks_per_quarter, 1)
        s, e = p.span_ticks
        return {
            "start": p.start,
            "bars": p.bars,
            "bars_available": len(p.edges) - 1,
            "context": [p.ctx_start, p.ctx_end],
            "span_seconds": [s / tpq * 60.0 / tempo_bpm, e / tpq * 60.0 / tempo_bpm],
            "generated_notes": counts,
            "tempo_bpm": tempo_bpm,
            "midi": self._score_to_bytes(scores[0], out_tempo),
            "midis": [self._score_to_bytes(sc, out_tempo) for sc in scores],
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
        client: str | None = None,
    ):
        params = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                  "top_k": top_k, "chunk_tokens": chunk_tokens, "tempo_bpm": tempo_bpm}
        notes, error = [], None
        # finally: the log is written even when the caller stops reading early
        try:
            prompt = self.gen.encode_midi_bytes(midi_bytes)
            if tempo_bpm is None:
                tempo_bpm = self.gen.detect_tempo_bytes(midi_bytes)
            prompt = [*self._header_for_bytes(midi_bytes, tempo_bpm), *prompt]
            for note in self.gen.stream_notes(
                prompt,
                chunk_tokens=chunk_tokens,
                tempo_bpm=tempo_bpm,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                d = {
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity,
                    "program": note.program,
                }
                notes.append(d)
                yield json.dumps(d) + "\n"
        except GeneratorExit:
            error = "client stopped reading"
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._log("stream", params, midi_bytes, notes=notes,
                      result={"tempo_bpm": tempo_bpm}, client=client, error=error)
