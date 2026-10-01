"""Infill pairs end to end on a tiny random model: pairgen writes them, the
judge renders them, the labeling hub shades the gap, the report reads them."""
import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai.pairgen import PairConfig, make_infill_pair, write_pair

TPQ = 480
BAR = 4 * TPQ


@pytest.fixture(scope="module")
def gen():
    from midigenai.generate import Generator
    from midigenai.model import ModelConfig, MusicTransformer
    from midigenai.tokenizer import build_tokenizer, save_tokenizer
    tok = build_tokenizer(scheme="v4")
    d = Path(tempfile.mkdtemp())
    save_tokenizer(tok, d / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=2048)
    torch.manual_seed(0)
    torch.save({"model": MusicTransformer(cfg).state_dict(), "model_config": asdict(cfg)},
               d / "ckpt.pt")
    return Generator(checkpoint_path=d / "ckpt.pt", tokenizer_path=d / "tokenizer.json",
                     device=torch.device("cpu"))


@pytest.fixture(scope="module")
def seed_file():
    s = Score(TPQ)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    for program, base in ((0, 60), (33, 36)):
        t = Track(program=program)
        for b in range(10):
            for beat in range(4):
                t.notes.append(Note(b * BAR + beat * TPQ, TPQ // 2, base + (b % 4), 80))
        s.tracks.append(t)
    f = Path(tempfile.mkdtemp()) / "ableton-test.mid"
    s.dump_midi(str(f))
    return f


@pytest.mark.parametrize("vs_original", [False, True])
def test_infill_pair_layout(gen, seed_file, tmp_path, vs_original):
    cfg = PairConfig(mode="infill", vs_original=vs_original, model_label="tiny")
    pair = None
    rng = random.Random(0)
    for _ in range(20):                        # a random model can write an empty fill
        pair = make_infill_pair(gen, seed_file, cfg, rng)
        if pair:
            break
    assert pair, "no pair made"
    m = pair["meta"]
    assert m["mode"] == "infill" and m["vs_original"] == vs_original
    assert m["model_b"] == ("original" if vs_original else "tiny")
    assert m["prompt_ids"][1] == gen.sp.task_infill and m["prompt_ids"][-1] == gen.sp.sep
    g0, g1 = m["gap_bars"]
    assert 1 <= g0 < g1 <= m["listen_bars"] - 1 and g1 - g0 == m["source_bars"]
    write_pair(pair, tmp_path / "pairs")

    # the prompt is silent in the gap; both sides keep the kept bars exactly
    from midigenai.infill_eval import _split
    fill, kept, _ = _split(tmp_path / "pairs" / f"{m['pair_id']}_prompt.mid", m["gap_beats"])
    assert not fill and kept
    for side in "ab":
        _, k, _ = _split(tmp_path / "pairs" / f"{m['pair_id']}_{side}.mid", m["gap_beats"])
        assert sorted(k) == sorted(kept)
    if vs_original:
        # the reference side is the source's own bars
        f_b, _, _ = _split(tmp_path / "pairs" / f"{m['pair_id']}_b.mid", m["gap_beats"])
        assert len(f_b) == 8 * m["source_bars"] and m["bars_ok_b"]

    # the judge sees the gap marked in the context and only gap bars per side
    from midigenai.llm_judge import build_prompt, render_context, render_side, rubric_name
    pd = tmp_path / "pairs"
    ctx = render_context(pd, m["pair_id"], "notes")
    assert ctx.count("(missing)") == g1 - g0
    side = render_side(pd, m["pair_id"], "b", "notes")
    bars_shown = {ln.split(":")[0] for ln in side.splitlines() if ln.startswith("bar ")}
    assert bars_shown <= {f"bar {b + 1}" for b in range(g0, g1)} or "silence" in side
    assert "FILL 1" in build_prompt(ctx, side, side, "infill")
    assert rubric_name("infill") == "infill" and rubric_name("accompany") == "fit_only"

    # the hub shades the gap and plays from the top
    from midigenai.relabel_app import build_roll
    roll, url = build_roll(pd, m["pair_id"], "a", tmp_path / "cache")
    assert url.startswith("pairs/") and roll["prompt_end_s"] == 0
    s0, s1 = roll["gap_s"]
    spb = 60.0 / m["tempo_bpm"]
    assert s0 == pytest.approx(m["gap_beats"][0] * spb, abs=0.01)
    assert all(n["prompt"] == (not s0 <= n["s"] < s1 - 1e-6) for n in roll["notes"])

    # the report reads the set and the labels
    from midigenai.infill_eval import report
    (tmp_path / "labels.jsonl").write_text(json.dumps({"pair_id": m["pair_id"], "preferred": "b",
                                                       "choice": "right"}) + "\n")
    r = report(tmp_path)
    assert r["pairs"] == 1 and r["vs_original"] == vs_original
    key = "model_win_rate_vs_original" if vs_original else "side_a_rate"
    assert r["preferences"]["all"][key] == 0.0


def test_infill_bans_early_stops(gen, monkeypatch):
    """The gap is exactly `bars` bars: EOS/BOS may not end it early."""
    seen = {}

    def fake(prompt_ids, max_new_tokens, temperature, top_k, min_new_tokens, seed, ban_ids=None):
        seen["ban"], seen["max"] = set(ban_ids), max_new_tokens
        return iter([])
    monkeypatch.setattr(gen, "_generate_raw", fake)
    bar, ts = gen.bar_id, gen.tokenizer.vocab["TimeSig_4/4"]
    list(gen.infill([bar, ts] * 4, [bar, ts] * 4, 2))
    assert {gen.eos_id, gen.bos_id, gen.sp.sep, gen.sp.mask} <= seen["ban"]
    assert seen["max"] >= 64 * 2 + 64


def test_select_dir_stores_repo_relative_paths(tmp_path, monkeypatch):
    import argparse
    from midigenai.relabel_app import select_dir
    pd = tmp_path / "set" / "pairs"
    pd.mkdir(parents=True)
    for s in ("prompt", "a", "b"):
        (pd / f"p1_{s}.mid").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    select_dir(argparse.Namespace(pairs=str(pd), out=str(tmp_path / "out"), n=10, dup=0,
                                  seed=0, name="s"))
    row = json.loads((tmp_path / "out" / "manifest.jsonl").read_text().splitlines()[0])
    assert row["pairs_dir"] == "set/pairs"
