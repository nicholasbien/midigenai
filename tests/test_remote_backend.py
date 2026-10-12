"""Generator(backend="remote"): the model runs elsewhere (web/midigen-web.mjs in a browser) and this side keeps the
tokenizer, the prompt layouts and postprocess, with no torch. The decoder's early stop must never cut what
postprocess keeps; that is checked against the JS runner's own stop rule when node is installed."""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from midigenai.generate import Generator
from midigenai.tokenizer import build_tokenizer, save_tokenizer

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "midigenai" / "web" / "midigen-web.mjs"


@pytest.fixture(scope="module")
def tok_path():
    d = Path(tempfile.mkdtemp())
    save_tokenizer(build_tokenizer(scheme="v4"), d / "tokenizer.json")
    return d / "tokenizer.json"


def remote(tok_path, decode):
    return Generator(checkpoint_path=None, tokenizer_path=tok_path, backend="remote", decode=decode)


def test_remote_needs_a_decoder_and_a_tokenizer(tok_path):
    with pytest.raises(ValueError):
        Generator(checkpoint_path=None, tokenizer_path=tok_path, backend="remote")
    with pytest.raises(ValueError):
        Generator(checkpoint_path=None, backend="remote", decode=lambda spec: [])


def test_spec_carries_the_sampling_contract_and_postprocess_runs_here(tok_path):
    seen = {}
    g = None

    def decode(spec):
        seen.update(spec)
        # two leading empty bars, a note, then EOS and junk after it
        return [g.bar_id, g.bar_id, note, g.eos_id, note]

    g = remote(tok_path, decode)
    note = next(v for k, v in g.tokenizer.vocab.items() if k.startswith("Pitch_"))
    out = list(g.generate_ids([g.bos_id, g.bar_id, note], max_new_tokens=40, temperature=0.9, top_k=20,
                              min_new_tokens=3, seed=11, stop_after_bars=2))
    assert seen["prompt"] == [g.bos_id, g.bar_id, note]
    assert (seen["max_new_tokens"], seen["temperature"], seen["top_k"], seen["min_new_tokens"], seen["seed"]) == (40, 0.9, 20, 3, 11)
    assert seen["eos_id"] == g.eos_id and seen["bar_id"] == g.bar_id and seen["stop_after_bars"] == 2
    assert set(seen["ban_ids"]) == set(g.default_ban_ids()) and set(seen["stop_ids"]) == set(g.stop_ids)
    assert set(seen["timesig_ids"]) == set(g.timesig_ids) and seen["trim_leading_bars"] is True
    json.dumps(seen)                                    # plain JSON, to send to a page
    # the prompt stops mid-bar, so one leading Bar is held and emitted with the note; EOS ends it
    assert out == [g.bar_id, note, g.eos_id]


def test_remote_runs_without_torch(tok_path):
    code = f"""
import sys
sys.modules["torch"] = None                     # any `import torch` now fails
from midigenai import remote_from_hub, Generator
from midigenai import infill_span, sequence_format, attributes
g = Generator(checkpoint_path=None, tokenizer_path={str(tok_path)!r}, backend="remote", decode=lambda s: [s["eos_id"]])
print(list(g.generate_ids([g.bos_id, g.bar_id], max_new_tokens=4)) == [g.eos_id])
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_runners_early_stop_never_cuts_what_postprocess_keeps(tok_path):
    g = remote(tok_path, lambda spec: [])
    v = g.tokenizer.vocab
    notes = [t for k, t in v.items() if k.startswith(("Pitch_", "Position_", "Velocity_", "Duration_"))][:12]
    ts = sorted(g.timesig_ids)[:2]
    pool = [g.bar_id] * 6 + ts + notes + [g.eos_id, g.sp.sep]
    rng = random.Random(0)
    cases = []
    for i in range(400):
        prompt = [g.bos_id, g.bar_id, ts[0], *rng.sample(notes, 3)] + ([g.bar_id] if i % 2 else [])
        limit = rng.choice([None, 1, 2, 3])
        trim = rng.random() < 0.7
        spec = g.remote_spec(prompt, 60, 1.0, 50, 0, None, g.default_ban_ids(), limit, trim)
        stream = [rng.choice(pool) for _ in range(60)]
        cases.append({"spec": spec, "stream": stream})
    script = f"""
import {{ stopper }} from {json.dumps(RUNNER.as_uri())};
import fs from "node:fs";
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map(c => {{
  const stop = stopper(c.spec);
  const i = c.stream.findIndex(t => stop(t));
  return i < 0 ? c.stream.length : i + 1;
}})));
"""
    r = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(cases),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cut_at = json.loads(r.stdout)
    stopped_early = 0
    for c, n in zip(cases, cut_at):
        s = c["spec"]
        kw = dict(stop_after_bars=s["stop_after_bars"], trim_leading_bars=s["trim_leading_bars"])
        full = list(g.postprocess(s["prompt"], c["stream"], **kw))
        cut = list(g.postprocess(s["prompt"], c["stream"][:n], **kw))
        assert cut == full, (c, n)
        stopped_early += n < len(c["stream"])
    assert stopped_early > 100                           # the rule does save tokens
