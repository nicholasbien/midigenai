"""
Modal training: train a midigenai model on Modal-hosted H100 / A100.

Cost reference (Modal pricing as of 2026-05):
    H100  ~$3.99/hr  — fastest, best for any model >50M
    A100-80GB  ~$2.50/hr  — cheaper, ~3-4x slower than H100 on bf16
    A10G  ~$1.10/hr  — only worth it for tiny models

Volumes:
    openmusenet2-v2-corpus  — read-only training data (shards + tokenizer + manifest)
    openmusenet2-v2-runs    — checkpoints + train logs

Upload corpus once (from wherever the shards live, e.g. Lambda):
    modal volume put openmusenet2-v2-corpus /home/ubuntu/data/v2_corpus_full /

Launch training:
    modal run midigenai/modal_train.py --size medium --max-steps 15000 --gpu H100

Pull a checkpoint back:
    modal volume get openmusenet2-v2-runs <run-name>/ckpt_final.pt ./
"""

from __future__ import annotations

import os as _os

import modal
from modal import Image, Volume


CORPUS_VOLUME_NAME = "openmusenet2-v2-corpus"
RUNS_VOLUME_NAME = "openmusenet2-v2-runs"

# GPU is baked into the function decorator at import time; override per launch:
#   MIDIGENAI_TRAIN_GPU=A10G modal run midigenai/modal_train.py --size pilot ...
TRAIN_GPU = _os.environ.get("MIDIGENAI_TRAIN_GPU", "H100")

app = modal.App("midigenai-train")

corpus_volume = Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=True)
runs_volume = Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.11")
    # torch pinned: the unpinned May-2026 build ships an inductor bug that
    # breaks torch.compile ("Too few arguments for CSE")
    .pip_install("torch==2.8.0", "miditok", "symusic", "numpy", "tqdm")
    .add_local_python_source("midigenai")
)


@app.function(
    image=image,
    gpu=TRAIN_GPU,
    timeout=24 * 3600,
    volumes={
        "/corpus": corpus_volume,
        "/runs": runs_volume,
    },
)
def train(
    size: str = "medium",
    max_steps: int = 15000,
    batch_size: int = 16,
    grad_accum: int = 4,
    block_size: int = 2048,
    warmup_steps: int = 200,
    eval_interval: int = 500,
    save_interval: int = 1000,
    lr: float = 3e-4,
    schedule: str = "wsd",
    decay_steps: int = 0,
    augment: bool = True,
    aug_pitch: int = 6,
    aug_velocity: int = 1,
    doc_start_frac: float = 0.2,
    mixture: str = "",
    corpus: str = "corpus_pilot",
    compile: bool = False,
    stage_local: bool = False,
    run_name: str | None = None,
    resume: bool = False,
    resume_from: str = "",
) -> dict:
    import os
    from datetime import datetime
    from pathlib import Path

    from midigenai.train import train as train_fn, TrainConfig

    if not run_name:
        run_name = f"{size}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(f"/runs/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume from the newest checkpoint in this run's directory, or an
    # explicit checkpoint (e.g. another run's pre-decay ckpt for a WSD
    # extension / context-extension phase)
    resume_ckpt = None
    if resume_from:
        resume_ckpt = Path(f"/runs/{resume_from}")
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"resume_from not found: {resume_ckpt}")
        print(f"[modal-train] resuming from explicit ckpt {resume_ckpt}")
    elif resume:
        ckpts = sorted(out_dir.glob("ckpt_*.pt"))
        if ckpts:
            resume_ckpt = ckpts[-1]
            print(f"[modal-train] resuming from {resume_ckpt}")
        else:
            print(f"[modal-train] --resume set but no checkpoints in {out_dir}; "
                  "starting fresh")

    print(f"[modal-train] run_name={run_name}")
    print(f"[modal-train] corpus contents:")
    os.system("ls -la /corpus/ /corpus/shards/ 2>/dev/null | head -20")

    # corpus subdir on the volume (upload with:
    #   modal volume put openmusenet2-v2-corpus <local> /<corpus-name>)
    data_dir = Path(f"/corpus/{corpus}")
    if not (data_dir / "shards").exists():
        for fallback in ("/corpus/v2_corpus_full", "/corpus"):
            if (Path(fallback) / "shards").exists():
                data_dir = Path(fallback)
                break

    if stage_local:
        # Random mmap reads over the volume FUSE mount thrash once the corpus
        # outgrows the page cache; a sequential copy to container-local disk up
        # front makes every training read local.
        import shutil
        import time as _time
        staged = Path("/tmp/corpus")
        t0 = _time.time()
        shutil.copytree(data_dir, staged)
        print(f"[modal-train] staged corpus to {staged} in {_time.time()-t0:.0f}s")
        data_dir = staged

    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        size=size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        block_size=block_size,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
        save_interval=save_interval,
        lr=lr,
        dtype="bfloat16",
        compile=compile,
        schedule=schedule,
        decay_steps=decay_steps,
        augment=augment,
        aug_pitch=aug_pitch,
        aug_velocity=aug_velocity,
        doc_start_frac=doc_start_frac,
        mixture=mixture,
        resume=resume_ckpt,
    )

    # Flush checkpoints to the volume every 10 min so a preempted/crashed run
    # loses at most that much progress (final commit still happens below).
    import threading
    stop_flush = threading.Event()

    def flush_loop():
        while not stop_flush.wait(600):
            runs_volume.commit()

    threading.Thread(target=flush_loop, daemon=True).start()
    try:
        train_fn(cfg)
    finally:
        stop_flush.set()

    runs_volume.commit()  # flush checkpoints to volume
    return {"run_name": run_name, "out_dir": str(out_dir)}


@app.local_entrypoint()
def main(size: str = "medium", max_steps: int = 15000, batch_size: int = 16,
         grad_accum: int = 4, block_size: int = 2048, lr: float = 3e-4,
         eval_interval: int = 500, save_interval: int = 1000,
         schedule: str = "wsd", augment: bool = True,
         aug_pitch: int = 6, aug_velocity: int = 1,
         doc_start_frac: float = 0.2, mixture: str = "",
         corpus: str = "corpus_pilot", compile: bool = False,
         stage_local: bool = False, run_name: str = "", resume: bool = False):
    """Local entrypoint — invoke training and print result."""
    result = train.remote(size=size, max_steps=max_steps, batch_size=batch_size,
                          grad_accum=grad_accum, block_size=block_size, lr=lr,
                          eval_interval=eval_interval, save_interval=save_interval,
                          schedule=schedule, augment=augment, aug_pitch=aug_pitch,
                          aug_velocity=aug_velocity,
                          doc_start_frac=doc_start_frac, mixture=mixture,
                          corpus=corpus, compile=compile,
                          stage_local=stage_local, run_name=run_name or None,
                          resume=resume)
    print(f"\n[done] {result}")
    print(f"\nRetrieve checkpoint with:")
    print(f"  modal volume get {RUNS_VOLUME_NAME} {result['run_name']}/ckpt_final.pt ./runs/")
