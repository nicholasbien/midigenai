"""Every Modal call is written to the generations Volume (log_generation).

The log has to hold the prompt, each output and the parameters, keep
concurrent requests in separate folders, and never raise into the request.
"""
import json
import os
from datetime import datetime, timezone

from midigenai.modal_serve import log_generation

WHEN = datetime(2026, 9, 29, 14, 3, 5, tzinfo=timezone.utc)


def test_writes_prompt_outputs_and_record(tmp_path):
    folder = log_generation(
        str(tmp_path), "generate", "v5-rl", {"temperature": 1.2, "n_samples": 2},
        b"PROMPT", midis=[b"A", b"B"],
        result={"prompt_tokens": 10, "midi": b"A", "midis": [b"A", b"B"]},
        client="web", now=WHEN)
    assert folder.startswith(os.path.join(str(tmp_path), "2026-09-29", "140305_"))
    assert folder.endswith("_generate")
    assert open(os.path.join(folder, "prompt.mid"), "rb").read() == b"PROMPT"
    assert open(os.path.join(folder, "out_0.mid"), "rb").read() == b"A"
    assert open(os.path.join(folder, "out_1.mid"), "rb").read() == b"B"
    rec = json.load(open(os.path.join(folder, "request.json")))
    assert rec["version"] == "v5-rl" and rec["client"] == "web"
    assert rec["params"] == {"temperature": 1.2, "n_samples": 2}
    assert rec["result"] == {"prompt_tokens": 10}      # MIDI bytes stay out of the JSON
    assert rec["outputs"] == 2 and rec["error"] is None


def test_same_second_requests_get_separate_folders(tmp_path):
    a = log_generation(str(tmp_path), "accompany", "v5", {}, b"x", now=WHEN)
    b = log_generation(str(tmp_path), "accompany", "v5", {}, b"x", now=WHEN)
    assert a != b and os.path.isdir(a) and os.path.isdir(b)


def test_stream_notes_and_errors(tmp_path):
    notes = [{"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 90, "program": 0}]
    folder = log_generation(str(tmp_path), "stream", "v4", {}, b"x", notes=notes,
                            error="client stopped reading", now=WHEN)
    assert json.load(open(os.path.join(folder, "notes.json"))) == notes
    rec = json.load(open(os.path.join(folder, "request.json")))
    assert rec["outputs"] == 1 and rec["error"] == "client stopped reading"


def test_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert log_generation(str(blocker), "generate", "v5", {}, b"x") is None
