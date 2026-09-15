"""Judge plumbing: ABC rendering, swap handling, and scoring — no API calls."""
import json
import tempfile
from pathlib import Path

from symusic import Note, Score, TimeSignature, Track

from midigenai.llm_judge import build_prompt, judge_pair, to_abc


def _midi(tmp: Path, name: str, pitches=(60, 62, 64, 65)) -> Path:
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=0)
    for i in range(16):
        t.notes.append(Note(i * 240, 220, pitches[i % len(pitches)], 80))
    s.tracks.append(t)
    p = tmp / name
    s.dump_midi(p)
    return p


def test_to_abc_strips_header_noise():
    tmp = Path(tempfile.mkdtemp())
    abc = to_abc(_midi(tmp, "a.mid"))
    assert abc and "T:" not in abc and "X:" not in abc
    assert "K:" in abc                      # key line kept: the judge needs it
    assert to_abc(tmp / "missing.mid") is None


def test_build_prompt_labels_both_sides():
    text = build_prompt("K:C\nCDEF", "GABc", "cBAG")
    assert "CONTINUATION 1" in text and "CONTINUATION 2" in text
    assert text.index("CONTINUATION 1") < text.index("CONTINUATION 2")


class _Stub:
    """Returns a queued verdict per call, recording the prompts it saw."""
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen = []

    class _Chat:
        def __init__(self, outer): self.completions = outer

    def create(self, model, temperature, messages):
        self.seen.append(messages[-1]["content"])
        w = self.verdicts.pop(0)
        payload = json.dumps({"winner": w, "reason": "because"})
        return type("R", (), {"choices": [type("C", (), {
            "message": type("M", (), {"content": payload})()})()]})()

    @property
    def chat(self): return _Stub._Chat(self)


def test_swap_consistency_detects_position_bias():
    # a judge that always answers "1" is inconsistent once the sides swap
    biased = _Stub(["1", "1"])
    r = judge_pair(biased, "m", "K:C", "AAA", "BBB")
    assert r["first_pass"] == "1" and r["second_pass"] == "2"
    assert not r["consistent"] and r["verdict"] == "tie"

    # a judge with a real opinion mirrors its answer and keeps it
    stable = _Stub(["2", "1"])
    r = judge_pair(stable, "m", "K:C", "AAA", "BBB")
    assert r["consistent"] and r["verdict"] == "2"
    assert len(stable.seen) == 2 and stable.seen[0] != stable.seen[1]


def test_unparseable_reply_is_a_tie():
    class Bad(_Stub):
        def create(self, model, temperature, messages):
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {"content": "I cannot decide."})()})()]})()
    r = judge_pair(Bad([]), "m", "K:C", "A", "B")
    assert r["verdict"] == "tie"


def test_to_notes_is_readable_and_names_drums():
    from symusic import Note, Score, TimeSignature, Track

    from midigenai.llm_judge import render, to_notes
    tmp = Path(tempfile.mkdtemp())
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    piano = Track(program=0)
    piano.notes.append(Note(0, 480, 60, 80))          # C4, one beat, bar 1
    piano.notes.append(Note(1920 + 240, 240, 67, 80))  # G4, beat 0.5 of bar 2
    kit = Track(program=0, is_drum=True)
    kit.notes.append(Note(0, 60, 36, 100))            # kick
    kit.notes.append(Note(960, 60, 42, 90))           # closed hat, beat 2
    s.tracks.append(piano)
    s.tracks.append(kit)
    p = tmp / "x.mid"
    s.dump_midi(p)

    txt = to_notes(p)
    assert "tempo" in txt and "meter 4/4" in txt
    assert "bar 1:" in txt and "bar 2:" in txt
    assert "0:C4:1" in txt and "0.5:G4:0.5" in txt
    assert "0:Kick" in txt and "2:HatClosed" in txt   # named, not pitch numbers
    assert render(p, "notes") == txt and render(p, "abc") != txt
    assert to_notes(tmp / "nope.mid") is None


def test_prompt_variants_and_split():
    """Variants share the JSON contract; dev/test split is disjoint and stable."""
    from midigenai.llm_judge import PROMPTS, load_cases

    assert {"base", "taste", "strict", "fit_only"} <= set(PROMPTS)
    for name, text in PROMPTS.items():
        assert '"winner"' in text and "JSON only" in text, name
    assert PROMPTS["taste"] != PROMPTS["base"]
    assert "restraint beats busyness" in PROMPTS["taste"]
    assert "abstention costs nothing" in PROMPTS["strict"]

    labels = Path("evals/labeling_v3_same/labels.jsonl")
    if not labels.exists():
        return
    dev = {c[0] for c in load_cases(labels, 0, 0, "dev")}
    test = {c[0] for c in load_cases(labels, 0, 0, "test")}
    allc = {c[0] for c in load_cases(labels, 0, 0, "all")}
    # load_cases keeps only pairs whose MIDI is on disk, and pairs/ is
    # gitignored — in a fresh clone every split is empty, which says nothing
    # about the split logic
    if not allc:
        return
    assert dev and test and not (dev & test) and dev | test == allc
    assert dev == {c[0] for c in load_cases(labels, 0, 0, "dev")}   # deterministic
