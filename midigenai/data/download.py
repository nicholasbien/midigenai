"""
Dataset download helpers.

Each function fetches one corpus into a target directory. We don't bundle the
data — these scripts are run once on the training machine (Lambda Labs).

Sources (all freely available for research use):
- Lakh MIDI Dataset (LMD-full)         ~176k    https://colinraffel.com/projects/lmd/
- Los Angeles MIDI Dataset (LAMD)      ~400k    https://github.com/asigalov61/Los-Angeles-MIDI-Dataset
- GigaMIDI V2                          ~1.4M    https://huggingface.co/datasets/Metacreation/GigaMIDI
- Aria-MIDI (deduped)                  ~1M     https://huggingface.co/datasets/loubb/aria-midi
- MetaMIDI Dataset (MMD)               ~436k    https://zenodo.org/record/5142664  (overlaps Lakh)
- GiantMIDI-Piano                      ~10k     https://github.com/bytedance/GiantMIDI-Piano
- MAESTRO v3                           200hr    https://magenta.tensorflow.org/datasets/maestro
- POP909                               909      https://github.com/music-x-lab/POP909-Dataset
- Slakh2100                            2100     http://www.slakh.com/
- ATEPP                                classical https://github.com/anonymous-conf/ATEPP
- PDMX                                 ~250k    https://huggingface.co/datasets/openmusic/pdmx
                                                (MusicXML — needs conversion; future work)

Every new source goes through v2/data/clean.py + v2/data/dedup.py before
training — GigaMIDI and Aria overlap Lakh/LAMD and each other.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return dest
    print(f"[download] {url} -> {dest}")
    urlretrieve(url, dest)
    return dest


def _extract(archive: Path, target: Path) -> None:
    target = _ensure_dir(target)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    elif archive.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"]) or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def download_lakh(root: Path) -> Path:
    """Lakh MIDI Dataset — full version (~176k MIDI files, 1.6GB compressed)."""
    target = _ensure_dir(root / "lakh")
    archive = target / "lmd_full.tar.gz"
    _download("http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz", archive)
    if (target / "lmd_full").exists():
        print("[skip] lmd_full/ already extracted")
    else:
        _extract(archive, target)
    return target


def download_maestro(root: Path) -> Path:
    """MAESTRO v3 — expressive piano performances."""
    target = _ensure_dir(root / "maestro")
    archive = target / "maestro-v3.0.0-midi.zip"
    _download(
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        archive,
    )
    if any(p.is_dir() and p.name.startswith("maestro-v") for p in target.iterdir()):
        print("[skip] maestro already extracted")
    else:
        _extract(archive, target)
    return target


def download_pop909(root: Path) -> Path:
    """POP909 — pop melody/chord/accompaniment (~900 songs)."""
    target = _ensure_dir(root / "pop909")
    if (target / "POP909-Dataset").exists():
        print("[skip] pop909 already cloned")
        return target
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/music-x-lab/POP909-Dataset.git",
         str(target / "POP909-Dataset")]
    )
    return target


def download_giantmidi(root: Path) -> Path:
    """GiantMIDI-Piano — ByteDance's classical piano transcriptions."""
    target = _ensure_dir(root / "giantmidi")
    if (target / "GiantMIDI-Piano").exists():
        print("[skip] giantmidi already cloned")
        return target
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/bytedance/GiantMIDI-Piano.git",
         str(target / "GiantMIDI-Piano")]
    )
    return target


def download_lamd_via_hf(root: Path) -> Path:
    """
    Los Angeles MIDI Dataset — large cleaned MIDI corpus (~400k files).

    Source: HuggingFace dataset 'projectlosangeles/Los-Angeles-MIDI-Dataset'.
    We grab only the v4.0 MIDI zip (~9.2 GB), not the accompanying notebooks/code.
    Set HF_TOKEN env var if rate-limited.
    """
    target = _ensure_dir(root / "lamd")
    extracted_marker = target / "MIDIs"
    if extracted_marker.exists():
        print("[skip] lamd already extracted")
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("pip install huggingface_hub") from e

    archive_name = "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA.zip"
    archive_path = target / archive_name
    if not archive_path.exists():
        print(f"[download] {archive_name} (~9 GB) ...")
        hf_hub_download(
            repo_id="projectlosangeles/Los-Angeles-MIDI-Dataset",
            repo_type="dataset",
            filename=archive_name,
            local_dir=str(target),
        )
    _extract(archive_path, target)
    return target


def _hf_download_archive(repo_id: str, filename: str, target: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("pip install huggingface_hub") from e
    archive = target / filename
    if not archive.exists():
        print(f"[download] {repo_id}/{filename} ...")
        hf_hub_download(repo_id=repo_id, repo_type="dataset",
                        filename=filename, local_dir=str(target))
    return archive


def download_gigamidi(root: Path) -> Path:
    """
    GigaMIDI V2 — largest open MIDI corpus (~1.4M files, 5.5 GB zip), with an
    expressiveness-metadata CSV usable as a quality signal.

    Gated with auto-approval: log in with an HF token (`huggingface-cli login`
    or HF_TOKEN env var) and accept the terms once on the dataset page
    (https://huggingface.co/datasets/Metacreation/GigaMIDI); access is granted
    instantly, no human review.
    """
    target = _ensure_dir(root / "gigamidi")
    if (target / "extracted").exists():
        print("[skip] gigamidi already extracted")
        return target
    archive = _hf_download_archive(
        "Metacreation/GigaMIDI", "Final_GigaMIDI_V2.0_Final.zip", target)
    _extract(archive, target / "extracted")
    return target


def download_aria(root: Path) -> Path:
    """
    Aria-MIDI — ~1M solo-piano performance transcriptions. We take the
    *deduped* variant (2 GB tar.gz); our own near-dup pass still runs after,
    to catch overlap with the other corpora. Ungated.
    """
    target = _ensure_dir(root / "aria")
    if (target / "extracted").exists():
        print("[skip] aria already extracted")
        return target
    archive = _hf_download_archive(
        "loubb/aria-midi", "aria-midi-v1-deduped-ext.tar.gz", target)
    _extract(archive, target / "extracted")
    return target


DATASETS = {
    "lakh": download_lakh,
    "maestro": download_maestro,
    "pop909": download_pop909,
    "giantmidi": download_giantmidi,
    "lamd": download_lamd_via_hf,
    "gigamidi": download_gigamidi,
    "aria": download_aria,
}


def main(root: Path, names: list[str] | None = None) -> None:
    root = _ensure_dir(root)
    names = names or list(DATASETS)
    for name in names:
        if name not in DATASETS:
            print(f"[warn] unknown dataset: {name}")
            continue
        DATASETS[name](root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--datasets", nargs="*",
                        help="subset of datasets to download (default: all)")
    args = parser.parse_args()
    main(args.root, args.datasets)
