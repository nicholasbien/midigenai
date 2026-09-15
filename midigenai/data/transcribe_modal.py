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
help at all" for a few dollars. An L4 costs ~$0.39 per hour of audio.

Running it locally instead is not practical for bulk. Re-measured on an M3
Max (2026-09-15, large model, one 30 s clip after a warm-up pass): 145.8 s,
i.e. 0.21x realtime. The weights do sit on MPS -- 392 parameters on mps:0 --
so this is not a CPU fallback; it is the per-kernel launch overhead that
autoregressive decoding pays on Metal, the same reason midigenai has its own
MLX backend. At that rate fma_small alone is ~13 days of pinned machine.
"""

from __future__ import annotations

import os

import modal

VOLUME = "midigenai-transcribed"
GPU = os.environ.get("MIDIGENAI_TRANSCRIBE_GPU", "L4")

app = modal.App("midigenai-transcribe")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (
    # bookworm: the bullseye slim image's security pool has moved on and
    # apt-get install 404s on ffmpeg
    modal.Image.from_registry("python:3.11-slim-bookworm", add_python=None)
    .run_commands("apt-get update && apt-get install -y --no-install-recommends "
                  "ffmpeg unzip curl ca-certificates")
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
    top = df[("track", "genre_top")]
    if hasattr(top, "columns"):          # duplicate column labels give a frame
        top = top.iloc[:, 0]
    wanted = {g.strip().lower().replace("_", "-") for g in genres}
    # untagged tracks come through as NaN floats, not strings
    keep = {i for i, g in top.items()
            if isinstance(g, str) and any(w in g.lower() for w in wanted)}
    sel = [p for p in audio if int(p.stem) in keep]
    print(f"[fma] {len(sel)} of {len(audio)} files match {sorted(wanted)}")
    return [str(p) for p in sel[:limit]]


# The workspace GPU cap is shared with training AND serving. Transcription
# is the only one of the three that nobody is waiting on, so it gets what is
# left over, not what is available. Budget at a limit of 10: 1 training
# (H100), up to 3 serving (midigenai-serve, one pool per version), 4 here,
# 2 spare. Running this uncapped took every free slot and put the public
# site into a site-wide 500 -- every request queued for a GPU past Railway's
# ~120 s ceiling.
MAX_CONTAINERS = int(os.environ.get("MIDIGENAI_TRANSCRIBE_CONTAINERS", "4"))


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=12 * 3600,
              max_containers=MAX_CONTAINERS,
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


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def genre_breakdown(top_n: int = 20) -> dict:
    """What is actually in FMA, so the transcription budget can be aimed."""
    import pandas as pd
    from pathlib import Path
    out = {}
    for subset in ("fma_small",):
        meta = Path("/data/fma/fma_metadata/tracks.csv")
        df = pd.read_csv(meta, index_col=0, header=[0, 1], low_memory=False)
        have = {int(p.stem) for p in Path("/data/fma", subset).rglob("*.mp3")}
        col = df[("track", "genre_top")]
        sub = col[col.index.isin(have)]
        out[subset] = sub.value_counts().head(top_n).to_dict()
        out[f"{subset}_total"] = len(have)
    # the full catalogue, not just what we downloaded
    col = df[("track", "genre_top")]
    out["fma_all_tracks"] = col.value_counts().head(top_n).to_dict()
    out["fma_all_total"] = int(col.notna().sum())
    return out


@app.function(image=image, gpu=os.environ.get("MIDIGENAI_BENCH_GPU", "L4"),
              volumes={"/data": vol}, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface")])
def bench(size: str = "large", batch_size: int = 6) -> dict:
    """Transcription throughput on whatever GPU this function is pinned to."""
    import time
    from pathlib import Path
    import torch
    from muscriptor import TranscriptionModel
    clip = sorted(Path("/data/fma/fma_small").rglob("*.mp3"))[0]
    m = TranscriptionModel.load_model(size)
    m.transcribe_to_midi(str(clip), batch_size=1)                 # warm
    out = {}
    for bs in (1, batch_size):
        t0 = time.perf_counter()
        for _ in range(3):
            m.transcribe_to_midi(str(clip), batch_size=bs,
                                 prelude_forcing=(bs == 1))
        dt = (time.perf_counter() - t0) / 3
        out[f"batch{bs}"] = {"seconds_per_30s_clip": round(dt, 1),
                             "realtime_factor": round(30 / dt, 2)}
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return out


@app.local_entrypoint()
def genres(top_n: int = 20):
    """Print what is in FMA so the transcription budget can be aimed."""
    d = genre_breakdown.remote(top_n)
    print(f"\ndownloaded subset (fma_small): {d['fma_small_total']} clips")
    for g, n in d["fma_small"].items():
        print(f"  {g:22s} {n:6d}")
    print(f"\nwhole FMA catalogue: {d['fma_all_total']:,} tracks with a top genre")
    for g, n in d["fma_all_tracks"].items():
        print(f"  {g:22s} {n:6d}")


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=3 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def sample_genres(per_genre: int = 2, size: str = "large") -> list[dict]:
    """One or two transcriptions per genre, so a human can pick what to fund."""
    from pathlib import Path
    import pandas as pd
    from muscriptor import TranscriptionModel

    meta = Path("/data/fma/fma_metadata/tracks.csv")
    df = pd.read_csv(meta, index_col=0, header=[0, 1], low_memory=False)
    col = df[("track", "genre_top")]
    audio = {int(p.stem): p for p in Path("/data/fma/fma_small").rglob("*.mp3")}
    by_genre: dict[str, list] = {}
    for tid, g in col.items():
        if isinstance(g, str) and tid in audio:
            by_genre.setdefault(g, []).append(audio[tid])

    out_dir = Path("/data/genre_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TranscriptionModel.load_model(size)
    made = []
    for g, files in sorted(by_genre.items()):
        for p in files[:per_genre]:
            stem = f"{g.replace('/', '-').replace(' ', '')}_{p.stem}"
            mid = out_dir / f"{stem}.mid"
            if not mid.exists():
                try:
                    mid.write_bytes(model.transcribe_to_midi(str(p)))
                except Exception as e:
                    print(f"{stem}: {type(e).__name__}"); continue
            # keep the source audio next to it so the two can be compared
            wav = out_dir / f"{stem}.mp3"
            if not wav.exists():
                wav.write_bytes(Path(p).read_bytes())
            made.append({"genre": g, "stem": stem})
        vol.commit()
    vol.commit()
    print(f"[samples] {len(made)} clips across {len(by_genre)} genres")
    return made


@app.local_entrypoint()
def samples(per_genre: int = 2):
    for m in sample_genres.remote(per_genre):
        print(f"  {m['genre']:20s} {m['stem']}")


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def untagged() -> dict:
    """How much of FMA carries no top-level genre."""
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path("/data/fma/fma_metadata/tracks.csv"),
                     index_col=0, header=[0, 1], low_memory=False)
    top = df[("track", "genre_top")]
    allg = df[("track", "genres_all")]
    has_any = allg.astype(str).str.len() > 2        # "[]" when there are none
    return {"tracks_total": int(len(df)),
            "with_top_genre": int(top.notna().sum()),
            "no_top_genre": int(top.isna().sum()),
            "no_top_but_has_subgenres": int((top.isna() & has_any).sum()),
            "no_genre_at_all": int((top.isna() & ~has_any).sum())}


@app.local_entrypoint()
def untagged_report():
    for k, v in untagged.remote().items():
        print(f"  {k:28s} {v:7,d}")
