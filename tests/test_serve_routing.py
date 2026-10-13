"""The `model=` query param has to reach a real checkpoint folder.

Before version routing existed the site's dropdown was a label: every
selection served the same weights. These tests pin the mapping and the
fallback so that can't silently come back.
"""
import pytest

from midigenai.modal_serve import DEFAULT_VERSION, SERVED_VERSIONS, resolve_version


def test_every_dropdown_value_maps_to_a_folder():
    for name, folder in SERVED_VERSIONS.items():
        assert resolve_version(name) == folder


def test_default_is_served():
    assert DEFAULT_VERSION in SERVED_VERSIONS
    assert resolve_version(None) == SERVED_VERSIONS[DEFAULT_VERSION]


@pytest.mark.parametrize("junk", ["", "   ", "v1", "nope", "../../etc/passwd",
                                  "v4/../v3", "V4"])
def test_unknown_values_fall_back_rather_than_reaching_the_filesystem(junk):
    """An old cached frontend or a hand-edited URL must not pick the path."""
    assert resolve_version(junk) == SERVED_VERSIONS[DEFAULT_VERSION]


def test_whitespace_is_tolerated():
    assert resolve_version(" v3 ") == "v3"


def test_v4_large_is_served():
    """v4-large shipped to the Hub as its own subfolder; the site's dropdown
    value has to reach it rather than falling back to v4."""
    assert resolve_version("v4-large") == "v4-large"
    assert SERVED_VERSIONS["v4-large"] == "v4-large"


def test_accompaniment_versions_are_v4_folders():
    """The accompaniment gate compares against resolve_version's output, so
    every entry has to be a served folder -- and only v4-or-later ones (v2/v3
    have no accompaniment document type)."""
    from midigenai.modal_serve import ACCOMPANIMENT_VERSIONS
    folders = set(SERVED_VERSIONS.values())
    for name in ACCOMPANIMENT_VERSIONS:
        assert name in folders
        assert name.startswith(("v4", "v5"))
    assert resolve_version(None) in ACCOMPANIMENT_VERSIONS
    for older in ("v2", "v3"):
        assert resolve_version(older) not in ACCOMPANIMENT_VERSIONS


def test_empty_accompaniment_takes_are_resampled():
    from midigenai.modal_serve import first_nonempty
    takes = iter([("silent", 0), ("bass", 12), ("never", 9)])
    assert first_nonempty(lambda: next(takes)) == ("bass", 12, 2)
    # gives up after 1 + retries draws and returns the last (empty) one
    empties = iter([("a", 0), ("b", 0), ("c", 0), ("d", 5)])
    assert first_nonempty(lambda: next(empties), retries=2) == ("c", 0, 3)
    first = iter([("ok", 3)])
    assert first_nonempty(lambda: next(first)) == ("ok", 3, 1)


def test_v5_names():
    """Since 2026-10 v5 is the GRPO model and v5-base the model before it, each
    in the folder of its name; the old name v5-rl still reaches the GRPO model
    but isn't listed."""
    from midigenai.hub import ALIASES, folder
    from midigenai.modal_serve import VERSION_ALIASES
    assert DEFAULT_VERSION == "v5"
    assert resolve_version("v5") == resolve_version("v5-rl") == resolve_version(None) == "v5"
    assert resolve_version("v5-base") == "v5-base"
    assert ALIASES == VERSION_ALIASES and not set(VERSION_ALIASES) & set(SERVED_VERSIONS)
    for name in ("v5", "v5-base", "v5-rl", "v4", "v4-large", "v3"):
        assert folder(name) == resolve_version(name)          # the Hub and the server agree


def test_hub_versions_leave_out_old_names(monkeypatch):
    import huggingface_hub
    from midigenai.hub import list_hub_versions
    files = ["README.md", "v5/ckpt_final.pt", "v5-base/ckpt_final.pt", "v5-rl/ckpt_final.pt",
             "v5-rl/tokenizer.json", "v4/ckpt_final.pt", "v5/onnx/manifest.json"]
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, repo, revision=None: files)
    assert list_hub_versions() == ["v4", "v5", "v5-base"]
