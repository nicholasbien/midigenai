"""The v4 serving surface: the attribute header that reaches the model, the
controls a caller can set, and the infill round trip — exercised against a
tiny local checkpoint, with Modal's decorators stepped around."""
from dataclasses import asdict

import pytest
import torch
from symusic import Note, Score, TimeSignature, Track

from midigenai.generate import Generator
from midigenai.modal_serve import MidiGen, local_method
from midigenai.model import ModelConfig, MusicTransformer
from midigenai.tokenizer import build_tokenizer, save_tokenizer


def _score(n_bars=6, program=0):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, 4, 4))
    t = Track(program=program)
    for b in range(n_bars):
        for beat in range(4):
            t.notes.append(Note(b * 1920 + beat * 480, 240, 60 + (b + beat) % 12, 80))
    s.tracks.append(t)
    return s


def _checkpoint(tmp_path, scheme="v4"):
    tok = build_tokenizer(scheme=scheme)
    save_tokenizer(tok, tmp_path / "tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      d_ff=64, max_seq_len=1024)
    torch.manual_seed(0)
    torch.save({"model": MusicTransformer(cfg).state_dict(),
                "model_config": asdict(cfg)}, tmp_path / "ckpt.pt")
    return Generator(tmp_path / "ckpt.pt", tmp_path / "tokenizer.json",
                     backend="torch", dtype=torch.float32, inference_seq_len=1024)


def _stub(gen, version="v4"):
    cls = MidiGen._get_user_cls()
    obj = cls.__new__(cls)
    obj.gen = gen
    obj.version = version
    return obj


@pytest.fixture(scope="module")
def v4(tmp_path_factory):
    return _stub(_checkpoint(tmp_path_factory.mktemp("v4")))


@pytest.fixture(scope="module")
def midi(tmp_path_factory):
    f = tmp_path_factory.mktemp("midi") / "riff.mid"
    _score().dump_midi(str(f))
    return f.read_bytes()


def test_v4_requests_carry_the_attribute_header(v4, midi):
    """Every training document had one and both offline eval paths build one;
    serving used to be the odd one out."""
    out = local_method("generate_batch")(v4, midi, max_new_tokens=8, n_samples=1)
    assert any(h.startswith("Density_") for h in out["header"])
    assert any(h.startswith("Inst_") for h in out["header"])


def test_controls_override_one_family_and_leave_the_rest_described(v4, midi):
    out = local_method("generate_batch")(v4, midi, max_new_tokens=8, n_samples=1,
                                         controls={"density": 3,
                                                   "instruments": ["Drums"]})
    assert "Density_3" in out["header"] and "Inst_Drums" in out["header"]
    assert "Inst_Piano" not in out["header"]
    assert any(h.startswith("Poly_") for h in out["header"])   # still described


def test_a_bad_control_is_refused_before_it_reaches_the_vocabulary(v4, midi):
    with pytest.raises(ValueError, match="unknown instrument"):
        local_method("generate_batch")(v4, midi, max_new_tokens=4,
                                       controls={"instruments": ["Kazoo"]})
    with pytest.raises(ValueError, match="out of range"):
        local_method("generate_batch")(v4, midi, max_new_tokens=4,
                                       controls={"density": 7})


def test_controls_on_a_v3_checkpoint_are_an_error_not_a_no_op(tmp_path, midi):
    v3 = _stub(_checkpoint(tmp_path, scheme="midilike"), version="v3")
    assert local_method("generate_batch")(v3, midi, max_new_tokens=4)["header"] == []
    with pytest.raises(ValueError, match="need a v4 checkpoint"):
        local_method("generate_batch")(v3, midi, max_new_tokens=4,
                                       controls={"density": 1})


def test_infill_replaces_a_span_and_keeps_the_piece_the_same_length(v4, midi):
    out = local_method("infill_batch")(v4, midi, at_bar=2, bars=2, n_samples=1)
    assert out["at_bar"] == 2 and out["bars"] == 2
    assert out["bars_available"] == 6
    assert out["span_start_seconds"] == pytest.approx(4.0, abs=0.01)

    # the two scores use different tick resolutions, so compare in beats
    before = Score.from_midi(midi)
    after = Score.from_midi(out["midi"])
    assert after.end() / after.tpq == pytest.approx(before.end() / before.tpq,
                                                    abs=0.5)
    # the bars after the span keep their place, whatever the model wrote
    tail = sorted(n.start / after.tpq for t in after.tracks for n in t.notes)[-4:]
    assert tail == pytest.approx([20.0, 21.0, 22.0, 23.0], abs=0.01)


def test_infill_never_rewrites_the_whole_upload(v4, midi):
    out = local_method("infill_batch")(v4, midi, at_bar=99, bars=99, n_samples=1)
    assert out["bars"] == out["bars_available"] - 1
    assert out["at_bar"] == out["bars_available"] - out["bars"]


def test_infill_needs_a_v4_checkpoint(tmp_path, midi):
    v3 = _stub(_checkpoint(tmp_path, scheme="midilike"), version="v3")
    with pytest.raises(ValueError, match="needs a v4 checkpoint"):
        local_method("infill_batch")(v3, midi, at_bar=0, bars=1)


def test_accompaniment_header_can_name_the_part_to_add(v4, tmp_path):
    f = tmp_path / "duo.mid"
    duo = _score()
    bass = Track(program=33)
    for b in range(6):
        bass.notes.append(Note(b * 1920, 480, 40 + b, 80))
    duo.tracks.append(bass)
    duo.dump_midi(str(f))

    out = local_method("accompany_batch")(v4, f.read_bytes(), bars=2, n_samples=1,
                                          controls={"instruments": ["Drums"]})
    assert "Inst_Drums" in out["header"]
    assert out["bars"] == 2 and out["condition_track"] in (0, 1)
