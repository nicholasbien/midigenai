"""The JS port in midigenai/web against golden fixtures from this Python code (tests/web/make_fixtures.py makes
them; tests/web/port.test.mjs checks them). Run here too, so a Python-only test run still notices a drift."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_js_port_matches_the_python_fixtures():
    r = subprocess.run(["node", "--test", str(ROOT / "tests" / "web" / "port.test.mjs")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
