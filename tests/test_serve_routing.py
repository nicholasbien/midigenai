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
