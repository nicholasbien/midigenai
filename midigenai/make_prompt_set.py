"""
Materialize a held-out prompt set for labeling / eval, and pin it by content
hash so two scorecards can be compared.

    build    copy held-out files out of a corpus manifest into a prompt dir
    freeze   write a manifest of content hashes for a prompt dir
    verify   check a prompt dir against a frozen manifest

Prompts must be files the model never trained on. `build_dataset` splits
train/val by hashing the file path, so the same hash identifies genuine
held-out files here — no separate bookkeeping, and the pool is as big as the
val split (~0.5% of the corpus, thousands of files).

    python -m midigenai.make_prompt_set build --out evals/prompts_heldout --n 400
    python -m midigenai.make_prompt_set freeze --prompts evals/prompts_heldout \\
        --name heldout_v1
    python -m midigenai.make_prompt_set verify --prompts evals/prompts_heldout \\
        --manifest evals/prompt_sets/heldout_v1.json

The MIDIs themselves are third-party music and stay gitignored (PR #44), so
the *set* travels with the repo instead of the audio: a manifest is a sorted
list of SHA-256s plus a `set_id` over them. It records no filenames on
purpose — the filenames alone name artists and tracks, which is why the files
left the repo in the first place. `eval_checkpoint --prompt-set` verifies a
directory against one before scoring and stamps the id into the scorecard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MANIFEST = Path.home() / "midigenai_data" / "manifest_plus_ggm_dedup.jsonl"
DEFAULT_SET_DIR = Path("evals/prompt_sets")
SOURCES = ("lakh", "lamd", "aria", "gigamidi", "maestro", "pop909", "giantmidi")


def source_of(path: str) -> str | None:
    for s in SOURCES:
        if f"/raw/{s}/" in path:
            return s
    return None


def source_of_prompt(name: str) -> str:
    """`build` names files val_<source>_<stem>.mid; the source is safe to
    record (it is a corpus name), the stem is not."""
    parts = name.split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "val" and parts[1] in SOURCES \
        else "unknown"


# ---------- hashing ---------- #

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(prompts: Path) -> dict[str, Path]:
    """{sha256: path} for the MIDI files in `prompts`. Duplicate content maps
    to one entry — a set is a set."""
    out: dict[str, Path] = {}
    for f in sorted(prompts.glob("*.mid")):
        out.setdefault(file_sha256(f), f)
    return out


def set_id_for(hashes) -> str:
    """Identity of a prompt set: one hash over its sorted member hashes."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()[:16]


def freeze_prompts(prompts: Path, name: str) -> dict:
    by_hash = hash_dir(prompts)
    if not by_hash:
        raise SystemExit(f"[prompts] no .mid files in {prompts}")
    files = [{"sha256": h, "source": source_of_prompt(by_hash[h].name),
              "bytes": by_hash[h].stat().st_size}
             for h in sorted(by_hash)]
    return {
        "name": name,
        "set_id": set_id_for(by_hash),
        "n": len(files),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "content hashes only; prompt filenames name third-party music",
        "files": files,
    }


def load_manifest(path: Path) -> dict:
    m = json.loads(Path(path).read_text())
    if "files" not in m:
        raise SystemExit(f"[prompts] {path} is not a prompt-set manifest")
    return m


def verify_prompts(prompts: Path, manifest: dict) -> dict:
    """Compare a directory's contents against a frozen manifest, by hash."""
    have = set(hash_dir(prompts))
    want = {f["sha256"] for f in manifest["files"]}
    return {
        "name": manifest.get("name", "?"),
        "set_id": manifest.get("set_id"),
        "ok": have >= want,
        "exact": have == want,
        "n_expected": len(want),
        "n_found": len(have & want),
        "missing": sorted(want - have),
        "extra": sorted(have - want),
    }


# ---------- subcommands ---------- #

def cmd_build(args) -> None:
    from midigenai.data.build_dataset import VAL_FRACTION, split_by_path

    by_source: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("n_notes", 0) < args.min_notes:
                continue
            if e.get("duration_seconds", 0) < args.min_seconds:
                continue
            if split_by_path(e["path"], VAL_FRACTION) != "val":
                continue
            src = source_of(e["path"])
            if src:
                by_source[src].append(e)
    print("[prompts] held-out pool:", {s: len(v) for s, v in sorted(by_source.items())})

    rng = random.Random(args.seed)
    live = {s: v for s, v in by_source.items() if v}
    for v in live.values():
        rng.shuffle(v)
    picked: list[tuple[str, dict]] = []
    i = 0
    while len(picked) < args.n and live:
        for s in sorted(live):
            if i < len(live[s]):
                picked.append((s, live[s][i]))
                if len(picked) >= args.n:
                    break
        i += 1
        if all(i >= len(v) for v in live.values()):
            break

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for src, e in picked:
        stem = Path(e["path"]).stem[:24]
        dst = args.out / f"val_{src}_{stem}.mid"
        if dst.exists():
            continue
        try:
            shutil.copy(e["path"], dst)
            written += 1
        except OSError:
            continue
    print(f"[prompts] wrote {written} new prompts to {args.out} "
          f"({len(list(args.out.glob('*.mid')))} total)")
    print(f"[prompts] freeze it before scoring anything against it:\n"
          f"    python -m midigenai.make_prompt_set freeze --prompts {args.out} "
          f"--name <name>")


def cmd_freeze(args) -> None:
    manifest = freeze_prompts(args.prompts, args.name)
    out = args.out or (DEFAULT_SET_DIR / f"{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        old = load_manifest(out)
        if old.get("set_id") == manifest["set_id"]:
            print(f"[prompts] {out} already pins this exact set ({old['set_id']})")
            return
        raise SystemExit(
            f"[prompts] {out} pins a different set ({old.get('set_id')} vs "
            f"{manifest['set_id']}). A frozen set is frozen: write a new name "
            f"(heldout_v2) so old scorecards keep meaning what they said, or "
            f"pass --force if this one was never used.")
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[prompts] froze {manifest['n']} prompts as {manifest['name']} "
          f"(set_id {manifest['set_id']}) -> {out}")


def cmd_verify(args) -> None:
    res = verify_prompts(args.prompts, load_manifest(args.manifest))
    print(f"[prompts] {res['name']} ({res['set_id']}): "
          f"{res['n_found']}/{res['n_expected']} present"
          + (f", {len(res['extra'])} extra file(s) in the directory"
             if res["extra"] else ""))
    if res["missing"]:
        print(f"[prompts] MISSING {len(res['missing'])}: "
              f"{', '.join(h[:12] for h in res['missing'][:5])}"
              + (" ..." if len(res["missing"]) > 5 else ""))
        print("[prompts] rebuild them with `build` (same --manifest, --n, --seed) "
              "or score against the set that is actually on disk.")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="copy held-out files into a prompt dir")
    b.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=400)
    b.add_argument("--min-notes", type=int, default=120)
    b.add_argument("--min-seconds", type=float, default=20.0)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("freeze", help="pin a prompt dir by content hash")
    f.add_argument("--prompts", type=Path, required=True)
    f.add_argument("--name", required=True, help="e.g. heldout_v1")
    f.add_argument("--out", type=Path, default=None,
                   help=f"default {DEFAULT_SET_DIR}/<name>.json")
    f.add_argument("--force", action="store_true",
                   help="overwrite a manifest that pins a different set")
    f.set_defaults(func=cmd_freeze)

    v = sub.add_parser("verify", help="check a prompt dir against a manifest")
    v.add_argument("--prompts", type=Path, required=True)
    v.add_argument("--manifest", type=Path, required=True)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
