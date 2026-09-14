"""
Transcribe Creative Commons audio to MIDI on Modal, to fill the genres the
corpus does not have.

The corpus is a third solo-piano transcription and most of the rest is 90s
and 2000s pop arrangements; contemporary electronic and hip hop are nearly
absent, because that music was never distributed as MIDI. This turns audio
into MIDI with MuScriptor (Kyutai/Mirelo, CC BY-NC weights, so keep the
output non-commercial) and writes the result to a Modal volume.

    modal run midigenai/data/transcribe_modal.py --limit 1000 --genres electronic,hip-hop

Free Music Archive is the source: `fma_small` is 8,000 thirty-second clips
(7.2 GB) with genre labels, which is enough to answer "does transcribed data
help at all" for a few dollars. Measured on an M3 Max the large model runs at
0.68x realtime; an L4 is roughly 3x that, so ~$0.39 per hour of audio.
"""

from __future__ import annotations

import os

import modal

VOLUME = "midigenai-transcribed"
GPU = os.environ.get("MIDIGENAI_TRANSCRIBE_GPU", "L4")

app = modal.App("midigenai-transcribe")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "unzip", "curl")
    .pip_install("torch==2.4.1", "torchaudio==2.4.1", "muscriptor", "symusic",
                 "pandas", "tqdm", "huggingface_hub>=0.30")
)

FMA_AUDIO = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"


@app.function(image=image, volumes={"/data": vol}, timeout=6 * 3600)
def fetch_fma() -> str:
    """Download and unpack FMA once onto the volume (idempotent)."""
    import subprocess
    from pathlib import Path
    root = Path("/data/fma")
    root.mkdir(parents=True, exist_ok=True)
    for url, marker in ((FMA_META, "fma_metadata"), (FMA_AUDIO, "fma_small")):
        if (root / marker).exists():
            print(f"[fma] {marker} already present")
            continue
        zp = root / f"{marker}.zip"
        if not zp.exists():
            print(f"[fma] downloading {url}")
            subprocess.run(["curl", "-L", "-o", str(zp), url], check=True)
        print(f"[fma] unzipping {marker}")
        subprocess.run(["unzip", "-q", "-o", str(zp), "-d", str(root)], check=True)
        zp.unlink()
        vol.commit()
    n = len(list((root / "fma_small").rglob("*.mp3")))
    print(f"[fma] {n} audio files ready")
    return str(root)


def _select(root, genres: list[str] | None, limit: int) -> list[str]:
    """Track paths, optionally filtered to the genres we actually lack."""
    from pathlib import Path
    import pandas as pd
    audio = sorted(Path(root, "fma_small").rglob("*.mp3"))
    meta = Path(root, "fma_metadata", "tracks.csv")
    if not (genres and meta.exists()):
        return [str(p) for p in audio[:limit]]
    df = pd.read_csv(meta, index_col=0, header=[0, 1], low_memory=False)
    top = df[("track", "genre_top")].astype(str).str.lower()
    wanted = {g.strip().lower().replace("_", "-") for g in genres}
    keep = {i for i, g in top.items() if any(w in g for w in wanted)}
    sel = [p for p in audio if int(p.stem) in keep]
    print(f"[fma] {len(sel)} of {len(audio)} files match {sorted(wanted)}")
    return [str(p) for p in sel[:limit]]


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=12 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def transcribe(paths: list[str], size: str = "large") -> dict:
    """Transcribe a shard of audio files; writes <stem>.mid onto the volume."""
    import time
    from pathlib import Path
    from muscriptor import TranscriptionModel

    out = Path("/data/midi")
    out.mkdir(parents=True, exist_ok=True)
    model = TranscriptionModel.load_model(size)
    done = failed = notes = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        dst = out / f"{Path(p).stem}.mid"
        if dst.exists():
            continue
        try:
            dst.write_bytes(model.transcribe_to_midi(p))
            done += 1
            from symusic import Score
            notes += sum(len(t.notes) for t in Score(str(dst)).tracks)
        except Exception as e:                       # a bad mp3 must not kill the shard
            print(f"[transcribe] {Path(p).name}: {type(e).__name__}: {e}"[:160])
            failed += 1
        if i % 25 == 0:
            vol.commit()
            print(f"[transcribe] {i}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    vol.commit()
    return {"done": done, "failed": failed, "notes": notes,
            "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(limit: int = 1000, genres: str = "electronic,hip-hop",
         size: str = "large", shards: int = 8):
    root = fetch_fma.remote()
    paths = fetch_and_select.remote(root, genres, limit)
    if not paths:
        print("no matching audio"); return
    chunks = [paths[i::shards] for i in range(shards)]
    print(f"[transcribe] {len(paths)} files over {shards} shards on {GPU}")
    totals = {"done": 0, "failed": 0, "notes": 0}
    for r in transcribe.starmap([(c, size) for c in chunks if c]):
        for k in totals:
            totals[k] += r[k]
        print("  shard:", r)
    print(f"[transcribe] {totals['done']} files, {totals['notes']:,} notes, "
          f"{totals['failed']} failed")
    print(f"pull with: modal volume get {VOLUME} midi/ ./transcribed/")


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch_and_select(root: str, genres: str, limit: int) -> list[str]:
    return _select(root, genres.split(",") if genres else None, limit)
