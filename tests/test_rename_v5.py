"""scripts/rename_v5.py's plan and its check, on a made-up Hub listing."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rename_v5", Path(__file__).parents[1] / "scripts" / "rename_v5.py")
rename_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rename_v5)

BEFORE = {
    "README.md": (3054, "readme", False),
    "v4/ckpt_final.pt": (9, "v4ckpt", True),
    "v5/ckpt_final.pt": (9, "base", True),
    "v5/tokenizer.json": (5, "tok", False),
    "v5/notes.txt": (1, "old", False),
    "v5-rl/ckpt_final.pt": (9, "grpo", True),
    "v5-rl/tokenizer.json": (5, "tok", False),
    "v5-rl/onnx/model_q4.onnx": (7, "q4", True),
}


def after(steps, before=BEFORE):
    """What the Hub would hold after the commit."""
    out = dict(before)
    for op, path, src in steps:
        if op == "delete":
            del out[path]
        else:
            out[path] = before[src]
    return out


def test_the_moves():
    steps = rename_v5.plan(BEFORE)
    assert ("copy", "v5-base/ckpt_final.pt", "v5/ckpt_final.pt") in steps
    assert ("add", "v5-base/tokenizer.json", "v5/tokenizer.json") in steps        # small files aren't on LFS
    assert ("copy", "v5/onnx/model_q4.onnx", "v5-rl/onnx/model_q4.onnx") in steps
    assert ("delete", "v5/notes.txt", "") in steps                              # the new v5/ is v5-rl/ exactly
    new = after(steps)
    assert new["v5/ckpt_final.pt"][1] == new["v5-rl/ckpt_final.pt"][1] == "grpo"
    assert new["v5-base/ckpt_final.pt"][1] == "base" and new["v5-base/notes.txt"][1] == "old"
    assert rename_v5.check(BEFORE, new) == []


def test_the_check_catches_a_bad_commit():
    new = after(rename_v5.plan(BEFORE))
    new["v5/ckpt_final.pt"] = BEFORE["v5/ckpt_final.pt"]                        # the base model left in v5/
    del new["v5-rl/tokenizer.json"]
    assert sorted(rename_v5.check(BEFORE, new)) == ["v5-rl/tokenizer.json: missing",
                                                     "v5/ckpt_final.pt: not the file it should be"]


def test_it_runs_once():
    with pytest.raises(SystemExit, match="already"):
        rename_v5.plan(after(rename_v5.plan(BEFORE)))
    with pytest.raises(SystemExit, match="v5-rl/ is not on the Hub"):
        rename_v5.plan({"v5/ckpt_final.pt": (9, "base", True)})
