"""GRPO on a Modal GPU.

Rollout dominates a GRPO step — measured at 82% — and rollout is exactly what
a laptop is worst at: a 113M policy decoding batch-8 is latency-bound, so the
GPU that sits idle at batch 1 is the whole saving. Measured 5.6x from batching
the group on GPU vs 1.5x on CPU.

    modal run midigenai/modal_grpo.py --steps 1000 --prompts-per-step 8 --lr 5e-6

Reads the policy from the `midigenai-models` volume that serving already uses
(subfolder per version), and writes checkpoints and metrics.csv to
`midigenai-runs` under the run name, so a run can be inspected mid-flight:

    modal volume ls  midigenai-runs grpo_v4_001
    modal volume get midigenai-runs grpo_v4_001/metrics.csv ./

The reward spec and prompt MIDIs are small, so they ride along in the image
rather than needing a volume upload of their own.
"""

from __future__ import annotations

import os as _os

import modal
from modal import Image, Volume

MODELS_VOLUME = "midigenai-models"   # same volume modal_serve reads
RUNS_VOLUME = "midigenai-runs"
MODELS_ROOT = "/models"
RUNS_ROOT = "/runs"

GRPO_GPU = _os.environ.get("MIDIGENAI_GRPO_GPU", "H100")

app = modal.App("midigenai-grpo")

models_volume = Volume.from_name(MODELS_VOLUME, create_if_missing=True)
runs_volume = Volume.from_name(RUNS_VOLUME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.11")
    # same pin as modal_train: the unpinned build ships an inductor bug
    .pip_install("torch==2.8.0", "miditok", "symusic", "numpy")
    .add_local_dir("evals/reward", remote_path="/work/reward")
    .add_local_python_source("midigenai")
)


@app.function(
    image=image,
    gpu=GRPO_GPU,
    timeout=12 * 3600,
    volumes={MODELS_ROOT: models_volume, RUNS_ROOT: runs_volume},
)
def grpo(
    run_name: str = "grpo_v4_001",
    version: str = "v4",
    reward: str = "reward_v4_autolabel.json",
    prompts_tar: bytes | None = None,
    accompany_tar: bytes | None = None,      # multi-track seeds; enables the second task
    accompany_frac: float = 0.5,
    reward_accompany: str | None = None,     # probe fitted on accompaniment pairs
    bars: int = 16,
    steps: int = 1000,
    prompts_per_step: int = 8,
    group_size: int = 8,
    prompt_tokens: int = 256,
    max_new_tokens: int = 256,
    temperature: float = 1.1,
    top_k: int = 50,
    lr: float = 5e-6,
    beta: float = 0.04,
    save_every: int = 50,
    eval_every: int = 25,
    eval_prompts: int = 8,
    eval_samples: int = 4,
    seed: int = 0,
) -> dict:
    import io
    import tarfile
    from pathlib import Path

    from midigenai.grpo import GRPOConfig, train

    ckpt = Path(MODELS_ROOT) / version / "ckpt_final.pt"
    tokenizer = Path(MODELS_ROOT) / version / "tokenizer.json"
    for f in (ckpt, tokenizer):
        if not f.exists():
            raise SystemExit(f"{f} missing — `modal volume put {MODELS_VOLUME} "
                             f"<local>/{version} /{version}` first")

    reward_path = Path("/work/reward") / reward
    if not reward_path.exists():
        have = sorted(p.name for p in Path("/work/reward").glob("*.json"))
        raise SystemExit(f"reward spec {reward!r} not in the image; have: {have}")

    # prompts travel as a tar in the call rather than as a volume: a few
    # hundred MIDI files is under a megabyte and this keeps the run
    # self-describing — the prompts that produced a checkpoint ship with it
    prompts_dir = Path("/work/prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    if prompts_tar:
        with tarfile.open(fileobj=io.BytesIO(prompts_tar), mode="r:gz") as tf:
            tf.extractall(prompts_dir)
    n_prompts = len(list(prompts_dir.rglob("*.mid")))
    if not n_prompts:
        raise SystemExit("no prompt MIDIs were shipped with the call")
    # extractall may have made a nested directory
    root = prompts_dir
    if not list(root.glob("*.mid")):
        root = next(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.mid")))

    acc_root = None
    if accompany_tar:
        acc_dir = Path("/work/accompany_prompts"); acc_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(accompany_tar), mode="r:gz") as tf:
            tf.extractall(acc_dir)
        acc_root = acc_dir
        if not list(acc_root.glob("*.mid")):
            acc_root = next(d for d in acc_root.iterdir() if d.is_dir() and any(d.glob("*.mid")))
        if reward_accompany is None:
            raise SystemExit("accompaniment seeds shipped but no --reward-accompany")
    reward_acc_path = Path("/work/reward") / reward_accompany if reward_accompany else None
    if reward_acc_path is not None and not reward_acc_path.exists():
        raise SystemExit(f"accompaniment reward spec {reward_accompany!r} not in the image")

    out_dir = Path(RUNS_ROOT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[modal-grpo] {run_name}: {version} policy, {n_prompts} prompts, "
          f"reward {reward}, {steps} steps x {prompts_per_step} prompts x "
          f"{group_size} samples on {GRPO_GPU}", flush=True)

    train(GRPOConfig(
        checkpoint=ckpt, tokenizer=tokenizer, reward=reward_path, prompts=root,
        out_dir=out_dir, steps=steps, prompts_per_step=prompts_per_step,
        group_size=group_size, prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        lr=lr, beta=beta, save_every=save_every, eval_every=eval_every,
        eval_prompts=eval_prompts, eval_samples=eval_samples, seed=seed,
        device="cuda",
        accompany_prompts=acc_root, accompany_frac=accompany_frac,
        reward_accompany=reward_acc_path, bars=bars,
    ))
    runs_volume.commit()          # make the checkpoints visible before exit

    metrics = (out_dir / "metrics.csv")
    tail = metrics.read_text().splitlines()[-1] if metrics.exists() else ""
    return {"run": run_name, "out": str(out_dir), "last_metrics": tail,
            "checkpoints": sorted(p.name for p in out_dir.glob("ckpt_*.pt"))}


@app.function(image=image, volumes={MODELS_ROOT: models_volume}, timeout=600)
def checkpoint_identity(version: str = "v4") -> dict:
    """sha256 of the first 8 MB + byte size of the checkpoint on the volume.

    A probe reward is only valid for the checkpoint whose activations it was
    fitted on, and grpo refuses to run if they differ. The volume copy and the
    local Hub copy are supposed to be the same file; this checks that before a
    two-hour run finds out the hard way.
    """
    import hashlib
    from pathlib import Path as P
    f = P(MODELS_ROOT) / version / "ckpt_final.pt"
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        h.update(fh.read(8 << 20))
    out = {"path": str(f), "sha256_8mb": h.hexdigest(), "bytes": f.stat().st_size}
    print(f"[identity] {out['path']}\n[identity] sha256(first 8MB) "
          f"{out['sha256_8mb']}\n[identity] {out['bytes']} bytes", flush=True)
    return out


@app.local_entrypoint()
def main(
    run_name: str = "grpo_v4_001",
    version: str = "v4",
    reward: str = "reward_v4_autolabel.json",
    prompts: str = "~/midigenai-v4/evals/prompts_heldout",
    accompany_prompts: str = "",            # directory of multi-track seeds; empty = continuation only
    accompany_frac: float = 0.5,
    reward_accompany: str = "",
    bars: int = 16,
    steps: int = 1000,
    prompts_per_step: int = 8,
    group_size: int = 8,
    lr: float = 5e-6,
    beta: float = 0.04,
    eval_every: int = 25,
    seed: int = 0,
):
    import io
    import tarfile
    from pathlib import Path

    src = Path(prompts).expanduser()
    files = sorted(src.glob("*.mid"))
    if not files:
        raise SystemExit(f"no .mid files in {src}")
    # dereference: prompt pools are directories of symlinks (prompt_pool.py),
    # and a tar of links ships no notes — every prompt is then unreadable on
    # the worker and GRPO runs 1000 steps of "no usable groups".
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", dereference=True) as tf:
        for f in files:
            tf.add(f, arcname=f.name)
    blob = buf.getvalue()
    if len(blob) < 100 * len(files):
        raise SystemExit(f"prompt tar is {len(blob)} bytes for {len(files)} files — links without content?")
    print(f"shipping {len(files)} prompts ({len(blob)/1e6:.1f} MB) with the call")

    acc_blob = None
    if accompany_prompts:
        afiles = sorted(Path(accompany_prompts).expanduser().glob("*.mid"))
        if not afiles:
            raise SystemExit(f"no .mid files in {accompany_prompts}")
        abuf = io.BytesIO()
        with tarfile.open(fileobj=abuf, mode="w:gz", dereference=True) as tf:
            for f in afiles:
                tf.add(f, arcname=f.name)
        acc_blob = abuf.getvalue()
        if len(acc_blob) < 100 * len(afiles):
            raise SystemExit(f"accompaniment tar is {len(acc_blob)} bytes for {len(afiles)} files — links without content?")
        print(f"shipping {len(afiles)} accompaniment seeds ({len(acc_blob)/1e6:.1f} MB)")
    out = grpo.remote(run_name=run_name, version=version, reward=reward,
                      prompts_tar=blob, accompany_tar=acc_blob, accompany_frac=accompany_frac,
                      reward_accompany=reward_accompany or None, bars=bars, steps=steps,
                      prompts_per_step=prompts_per_step, group_size=group_size,
                      lr=lr, beta=beta, eval_every=eval_every, seed=seed)
    print(out)
    print(f"\nmodal volume get {RUNS_VOLUME} {run_name}/metrics.csv ./")
