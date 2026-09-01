"""
Generate continuations from multiple training checkpoints, server-side —
so hearing the model learn never requires downloading multi-GB checkpoints.

Deploy once, then spawn per batch of steps:
    modal deploy midigenai/modal_sample_evolution.py
    python - <<'PY'
    import json, modal
    prompts = json.load(open("evals/dataset_samples/evolution/prompts.json"))
    f = modal.Function.from_name("midigenai-evolution", "sample_evolution")
    print(f.remote(run_name="medium_full_v1", steps=[1000, 10000], prompts=prompts))
    PY

Outputs land on the runs volume at /runs/<run>/evolution/<step>__<prompt>.mid
(continuation only, prompt context trimmed), plus a manifest JSON per step.
"""

from __future__ import annotations

import modal
from modal import Image, Volume

runs_volume = Volume.from_name("openmusenet2-v2-runs")

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "miditok", "symusic", "numpy", "tqdm")
    .add_local_python_source("midigenai")
)

app = modal.App("midigenai-evolution")


@app.function(image=image, timeout=3600, volumes={"/runs": runs_volume})
def sample_evolution(run_name: str, steps: list[int], prompts: list[dict],
                     max_new_tokens: int = 512, temperature: float = 1.0,
                     top_k: int = 50, seed: int = 0) -> dict:
    import json
    from pathlib import Path

    import torch

    from midigenai.generate import Generator
    from midigenai.tokenizer import build_tokenizer

    out_dir = Path(f"/runs/{run_name}/evolution")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    results = {}

    for step in steps:
        ckpt = Path(f"/runs/{run_name}/ckpt_{step:06d}.pt")
        if not ckpt.exists():
            results[step] = "missing checkpoint"
            continue
        torch.manual_seed(seed)
        gen = Generator(str(ckpt), None, backend="torch")
        rows = []
        for p in prompts:
            ids = list(p["ids"])
            new_ids = list(gen.generate_ids(ids, max_new_tokens=max_new_tokens,
                                            temperature=temperature, top_k=top_k))
            full = tokenizer.decode(ids + new_ids)
            cut = tokenizer.decode(ids).end()
            for t in full.tracks:
                kept = [n for n in t.notes if n.start >= cut]
                for n in kept:
                    n.start -= cut
                t.notes = kept
            name = f"{step:06d}__{p['name']}.mid"
            full.dump_midi(out_dir / name)
            rows.append({"prompt": p["name"], "file": name,
                         "n_tokens": len(new_ids),
                         "ended_via_eos": len(new_ids) < max_new_tokens})
        (out_dir / f"manifest_{step:06d}.json").write_text(json.dumps(rows))
        results[step] = f"{len(rows)} generations"
        del gen
    runs_volume.commit()
    return results
