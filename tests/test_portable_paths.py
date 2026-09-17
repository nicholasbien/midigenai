"""No machine's home directory in committed files.

These are provenance and defaults, not data: nothing loads a probe spec by
its `checkpoint` path (grpo.py verifies the checkpoint by content hash), and
the label-set roots are local working copies. So they can be written
home-relative -- which still points at the same place on the machine that
wrote them, and names nobody on any other.
"""
import importlib
import json
import os
import re
from pathlib import Path

import pytest

from midigenai.reward_probe import portable_path

REPO = Path(__file__).resolve().parent.parent
HOME_PATH = re.compile(r"/(Users|home)/[A-Za-z0-9._-]+")


def test_portable_path_rewrites_home_as_tilde(monkeypatch, tmp_path):
    # HOME, not a Path.home patch: expanduser() reads the env var, and the
    # round-trip below is the property that matters.
    monkeypatch.setenv("HOME", str(tmp_path))
    assert portable_path(tmp_path / "runs" / "ckpt.pt") == "~/runs/ckpt.pt"


def test_portable_path_round_trips(monkeypatch, tmp_path):
    """The whole point: it still resolves to the same file for its author."""
    monkeypatch.setenv("HOME", str(tmp_path))
    original = tmp_path / ".cache" / "hub" / "ckpt_final.pt"
    assert Path(portable_path(original)).expanduser() == original


def test_paths_outside_home_are_left_absolute(monkeypatch, tmp_path):
    """A shared mount or a CI checkout has no home to fold away."""
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    assert portable_path("/opt/shared/ckpt.pt") == "/opt/shared/ckpt.pt"


@pytest.mark.parametrize("name", [
    "evals/reward/probe_v4_113m_block1.json",
    "evals/reward/probe_v4_113m_block8.json",
    "evals/reward/probe_v4_autolabel.json",
    "evals/reward/probe_v4large_luna.json",
    "evals/scorecards/len512_base.json",
])
def test_committed_specs_carry_no_home_directory(name):
    path = REPO / name
    if not path.exists():
        pytest.skip(f"{name} not on this branch")
    ck = json.loads(path.read_text()).get("checkpoint", "")
    assert not HOME_PATH.match(str(ck)), f"{name} names a home directory: {ck}"


def test_the_label_set_roots_are_not_one_machines_layout(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/someone")
    monkeypatch.delenv("MIDIGENAI_SETS", raising=False)
    monkeypatch.delenv("MIDIGENAI_WORKTREE", raising=False)
    import midigenai.relabel_app as ra
    importlib.reload(ra)
    for name, root in ra.DEFAULT_SETS.items():
        assert root.startswith("/Users/someone/"), (name, root)


def test_label_set_roots_are_overridable(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/someone")
    monkeypatch.setenv("MIDIGENAI_SETS", "v4=/mnt/pairs/v4")
    import midigenai.relabel_app as ra
    importlib.reload(ra)
    assert ra.DEFAULT_SETS["v4"] == "/mnt/pairs/v4"
    assert ra.DEFAULT_SETS["v1"].startswith("/Users/someone/")   # others untouched


def test_worktree_override_moves_the_v4_era_sets_together(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/someone")
    monkeypatch.delenv("MIDIGENAI_SETS", raising=False)
    monkeypatch.setenv("MIDIGENAI_WORKTREE", "/mnt/mg-v4")
    import midigenai.relabel_app as ra
    importlib.reload(ra)
    for name in ("v3_same", "v4", "v4_final"):
        assert ra.DEFAULT_SETS[name].startswith("/mnt/mg-v4/"), name


def test_no_source_file_bakes_in_a_home_directory():
    """A home path as a VALUE -- a default, a constant, a docstring command
    someone will copy -- is the thing that breaks on another machine. A `#`
    comment using one to illustrate a rule is not, so those are allowed.
    """
    offenders = []
    for py in (REPO / "midigenai").rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if HOME_PATH.search(line):
                offenders.append(f"{py.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "home directories baked in:\n" + "\n".join(offenders)
