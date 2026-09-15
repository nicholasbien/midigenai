"""Freezing a prompt set: identity by content, not by filename."""
import json
from pathlib import Path

import pytest

from midigenai.eval_checkpoint import _prompt_pool, _stamp_prompt_set
from midigenai.make_prompt_set import (file_sha256, freeze_prompts, set_id_for,
                                       verify_prompts)


def _prompts(tmp_path: Path, contents: dict[str, bytes]) -> Path:
    d = tmp_path / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (d / name).write_bytes(data)
    return d


def test_set_id_survives_renaming_and_moves(tmp_path):
    a = _prompts(tmp_path / "a", {"val_lakh_one.mid": b"MThd-one",
                                  "val_aria_two.mid": b"MThd-two"})
    b = _prompts(tmp_path / "b", {"val_lakh_renamed.mid": b"MThd-one",
                                  "val_aria_other.mid": b"MThd-two"})
    assert freeze_prompts(a, "x")["set_id"] == freeze_prompts(b, "x")["set_id"]


def test_set_id_changes_when_the_music_changes(tmp_path):
    a = _prompts(tmp_path / "a", {"val_lakh_one.mid": b"MThd-one"})
    b = _prompts(tmp_path / "b", {"val_lakh_one.mid": b"MThd-edited"})
    assert freeze_prompts(a, "x")["set_id"] != freeze_prompts(b, "x")["set_id"]


def test_manifest_records_hashes_not_filenames(tmp_path):
    d = _prompts(tmp_path, {"val_lamd_Some Famous Song.mid": b"MThd-one"})
    blob = json.dumps(freeze_prompts(d, "heldout_v1"))
    assert "Famous" not in blob and "Some" not in blob
    assert file_sha256(d / "val_lamd_Some Famous Song.mid") in blob
    assert '"source": "lamd"' in json.dumps(freeze_prompts(d, "heldout_v1"), indent=1)


def test_verify_reports_missing_and_tolerates_extra(tmp_path):
    full = _prompts(tmp_path / "full", {"a.mid": b"one", "b.mid": b"two"})
    manifest = freeze_prompts(full, "heldout_v1")

    partial = _prompts(tmp_path / "partial", {"a.mid": b"one"})
    res = verify_prompts(partial, manifest)
    assert not res["ok"] and res["n_found"] == 1
    assert res["missing"] == [file_sha256(full / "b.mid")]

    plus = _prompts(tmp_path / "plus", {"a.mid": b"one", "b.mid": b"two",
                                        "c.mid": b"three"})
    res = verify_prompts(plus, manifest)
    assert res["ok"] and not res["exact"] and len(res["extra"]) == 1


def test_prompt_pool_orders_by_hash_and_ignores_extra(tmp_path):
    d = _prompts(tmp_path, {"zzz.mid": b"one", "aaa.mid": b"two"})
    manifest = freeze_prompts(d, "heldout_v1")
    (d / "later_addition.mid").write_bytes(b"three")

    pool = _prompt_pool(d, manifest)
    assert [h for h, _ in pool] == sorted(h for h, _ in pool)
    assert len(pool) == 2                      # the extra file is not in the set
    assert set_id_for(h for h, _ in pool) == manifest["set_id"]


def test_prompt_pool_refuses_a_partial_set(tmp_path):
    full = _prompts(tmp_path / "full", {"a.mid": b"one", "b.mid": b"two"})
    manifest = freeze_prompts(full, "heldout_v1")
    partial = _prompts(tmp_path / "partial", {"a.mid": b"one"})
    with pytest.raises(SystemExit, match="missing 1 of 2"):
        _prompt_pool(partial, manifest)


def test_prompt_pool_without_a_set_keeps_filename_order(tmp_path):
    d = _prompts(tmp_path, {"zzz.mid": b"one", "aaa.mid": b"two"})
    assert [f.name for _, f in _prompt_pool(d, None)] == ["aaa.mid", "zzz.mid"]


def test_stamp_records_only_the_prompts_that_produced_rows(tmp_path):
    d = _prompts(tmp_path, {"a.mid": b"one", "b.mid": b"two"})
    manifest = freeze_prompts(d, "heldout_v1")
    card = {}
    rows = [{"prompt_sha": file_sha256(d / "a.mid")[:12]}, {"prompt": "b.mid"}]
    _stamp_prompt_set(card, manifest, rows)
    assert card["prompt_set"]["n_files"] == 2
    assert card["prompt_set"]["n_used"] == 1
    assert card["prompt_set"]["set_id"] == manifest["set_id"]
