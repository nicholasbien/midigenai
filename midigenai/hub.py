"""
HuggingFace download helper.

Loads a `Generator` straight from a HuggingFace model repo, so callers
don't have to manage checkpoint paths. Default repo:
[`nicholasbien/midigenai`](https://huggingface.co/nicholasbien/midigenai).

The repo is laid out as one subfolder per trained model:

    nicholasbien/midigenai
    ├── README.md
    ├── v2-pilot/             # current default
    │   ├── ckpt_final.pt
    │   └── tokenizer.json
    └── v2/                   # future
        ├── ckpt_final.pt
        └── tokenizer.json

Adding a new version = upload to a new subfolder; nothing in this file needs
to change. Users select via:

```python
from midigenai import load_from_hub, list_hub_versions

# default — picks up MIDIGENAI_VERSION env var if set, else DEFAULT_VERSION
gen = load_from_hub()

# explicit version
gen = load_from_hub(version="v2")

# discover what's available on the hub
print(list_hub_versions())   # ["v2-pilot", "v2", ...]
```

To change the default for a whole shell session without code edits:

    export MIDIGENAI_VERSION=v2

Files are cached at `~/.cache/huggingface/` after the first download.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_REPO = os.environ.get("MIDIGENAI_REPO_ID", "nicholasbien/midigenai")
DEFAULT_VERSION = os.environ.get("MIDIGENAI_VERSION", "v2-100m")

CKPT_FILENAME = "ckpt_final.pt"
TOKENIZER_FILENAME = "tokenizer.json"


def load_from_hub(
    version: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    **generator_kwargs,
):
    """Download checkpoint + tokenizer from HF and return a ready Generator.

    `version` defaults to `MIDIGENAI_VERSION` env var, then `DEFAULT_VERSION`.
    `repo_id` defaults to `MIDIGENAI_REPO_ID` env var, then `DEFAULT_REPO`.
    """
    from .generate import Generator
    ckpt, tok = download_files(version=version, repo_id=repo_id, revision=revision)
    return Generator(checkpoint_path=ckpt, tokenizer_path=tok, **generator_kwargs)


def download_files(
    version: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[Path, Path]:
    """Download just the files; return (ckpt_path, tokenizer_path)."""
    from huggingface_hub import hf_hub_download
    version = version or DEFAULT_VERSION
    repo_id = repo_id or DEFAULT_REPO
    ckpt = hf_hub_download(repo_id, CKPT_FILENAME, subfolder=version, revision=revision)
    tok = hf_hub_download(repo_id, TOKENIZER_FILENAME, subfolder=version, revision=revision)
    return Path(ckpt), Path(tok)


def list_hub_versions(repo_id: Optional[str] = None, revision: Optional[str] = None) -> list[str]:
    """List the available model versions (subfolders containing a checkpoint)
    on the given HF repo."""
    from huggingface_hub import HfApi
    repo_id = repo_id or DEFAULT_REPO
    api = HfApi()
    files = api.list_repo_files(repo_id, revision=revision)
    versions = sorted({
        f.split("/", 1)[0]
        for f in files
        if "/" in f and f.split("/", 1)[1] == CKPT_FILENAME
    })
    return versions
