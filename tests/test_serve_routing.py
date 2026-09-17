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
    every entry has to be a served folder -- and only v4-line ones."""
    from midigenai.modal_serve import ACCOMPANIMENT_VERSIONS
    folders = set(SERVED_VERSIONS.values())
    for name in ACCOMPANIMENT_VERSIONS:
        assert name in folders
        assert name.startswith("v4")
    assert resolve_version(None) in ACCOMPANIMENT_VERSIONS
    for older in ("v2", "v3"):
        assert resolve_version(older) not in ACCOMPANIMENT_VERSIONS
