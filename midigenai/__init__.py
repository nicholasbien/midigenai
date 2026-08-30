"""
midigenai — custom transformer for MIDI continuation.

Public surface:
- `Generator`           load a checkpoint and stream / batch generate
- `load_from_hub`      one-liner: pull a versioned model from HuggingFace
                          and return a generator
- `list_hub_versions`     list available model subfolders on the HF repo
"""

from .generate import Generator, Note
from .hub import (
    load_from_hub,
    download_files,
    list_hub_versions,
    DEFAULT_REPO,
    DEFAULT_VERSION,
)

__all__ = [
    "Generator",
    "Note",
    "load_from_hub",
    "download_files",
    "list_hub_versions",
    "DEFAULT_REPO",
    "DEFAULT_VERSION",
]

# legacy aliases (pre-rename API); prefer the names above
V2Generator = Generator
load_v2_from_hub = load_from_hub
download_v2_files = download_files
