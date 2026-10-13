"""Move the v5 models to their new names on the Hugging Face Hub (2026-10).

Before: v5/ holds the base model and v5-rl/ the GRPO-tuned one.
After:  v5/ holds the GRPO-tuned model, v5-base/ the base model, and v5-rl/
        stays as a copy of v5/, so midigenai releases from before the rename
        (whose default is v5-rl) keep loading the same model.

It is one commit, so the repo is never half moved, and it refuses to commit
if anything else was pushed after it looked. The Hub copies LFS files
(the checkpoints) itself, so nothing large is downloaded; small files such as
tokenizer.json are downloaded and added again, since the Hub copies only LFS
files.

    python scripts/rename_v5.py            # shows what it would do
    python scripts/rename_v5.py --apply    # commits it (needs a write token:
                                           # `hf auth login` or HF_TOKEN)

Then fill the serving volume's new folders and redeploy, so that no memory
snapshot keeps the old v5 weights, and only then merge the code that uses
the new names (the old code asks for v5-rl, which still holds the same model):

    modal run midigenai/modal_serve.py::sync_from_hub --version v5-base
    modal run midigenai/modal_serve.py::sync_from_hub --version v5
    modal deploy midigenai/modal_serve.py
"""
from __future__ import annotations

import argparse
import sys

REPO = "nicholasbien/midigenai"
MOVES = [("v5", "v5-base"), ("v5-rl", "v5")]    # read from the commit before this one


def snapshot(api, repo: str, revision: str) -> dict:
    """path -> (size, LFS sha256 or git blob id, is LFS) for every file at `revision`."""
    from huggingface_hub.hf_api import RepoFile
    out = {}
    for f in api.list_repo_tree(repo, recursive=True, expand=True, revision=revision):
        if isinstance(f, RepoFile):
            out[f.path] = (f.size, f.lfs.sha256 if f.lfs else f.blob_id, bool(f.lfs))
    return out


def plan(files: dict) -> list[tuple[str, str, str]]:
    """(op, path, source) for the commit: copy, add or delete."""
    folders = {p.split("/", 1)[0] for p in files if "/" in p}
    for src, _ in MOVES:
        if src not in folders:
            sys.exit(f"{src}/ is not on the Hub; nothing to move")
    if "v5-base" in folders:
        sys.exit("v5-base/ is already on the Hub: the rename looks done (run without --apply to see the files)")
    steps, kept = [], {}
    for src, dst in MOVES:
        for path in sorted(p for p in files if p.startswith(src + "/")):
            rel = path[len(src) + 1:]
            kept.setdefault(dst, set()).add(rel)
            steps.append(("copy" if files[path][2] else "add", f"{dst}/{rel}", path))
    # a file in the old v5/ that v5-rl/ doesn't have would linger in the new v5/
    for path in sorted(p for p in files if p.startswith("v5/")):
        if path[len("v5/"):] not in kept["v5"]:
            steps.append(("delete", path, ""))
    return steps


def check(before: dict, after: dict) -> list[str]:
    """What isn't where it should be after the commit (empty when it all is)."""
    want = {p: before[p] for p in before if not p.startswith("v5/")}          # v5-rl/ and the rest, as they were
    for src, dst in MOVES:
        want.update({f"{dst}/{p[len(src) + 1:]}": before[p] for p in before if p.startswith(src + "/")})
    bad = [f"{p}: missing" for p in want if p not in after]
    bad += [f"{p}: not the file it should be" for p in want if p in after and after[p][1] != want[p][1]]
    bad += [f"{p}: left over" for p in after if p.startswith(("v5/", "v5-base/")) and p not in want]
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--apply", action="store_true", help="make the commit (default: only show it)")
    args = ap.parse_args(argv)

    from huggingface_hub import (CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi,
                                 hf_hub_download)
    api = HfApi()
    head = api.repo_info(args.repo).sha
    before = snapshot(api, args.repo, head)
    steps = plan(before)
    for op, path, src in steps:
        size = before[src or path][0]
        print(f"{op:6} {path}" + (f"  <- {src}" if src else "") +
              (f"  ({size / 1e6:.0f} MB)" if size >= 1e6 else f"  ({size / 1e3:.0f} KB)"))
    if not args.apply:
        print(f"\n{len(steps)} changes on {args.repo} at {head[:8]}; run again with --apply to commit them.")
        return 0

    ops = []
    for op, path, src in steps:
        if op == "copy":
            ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=path, src_revision=head))
        elif op == "add":
            ops.append(CommitOperationAdd(path_in_repo=path,
                                          path_or_fileobj=hf_hub_download(args.repo, src, revision=head)))
        else:
            ops.append(CommitOperationDelete(path_in_repo=path))
    info = api.create_commit(args.repo, ops, parent_commit=head,
                             commit_message="v5 is the GRPO model, v5-base the one before it (v5-rl kept as a copy)")
    print(f"\ncommitted {info.oid[:8]}: {info.commit_url}")
    bad = check(before, snapshot(api, args.repo, info.oid))
    for line in bad:
        print("  " + line)
    print("checked: every file is where it should be" if not bad else "check failed: see above")
    if not bad:
        print("\nnext:\n  modal run midigenai/modal_serve.py::sync_from_hub --version v5-base\n"
              "  modal run midigenai/modal_serve.py::sync_from_hub --version v5\n"
              "  modal deploy midigenai/modal_serve.py")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
