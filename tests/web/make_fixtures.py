"""Golden fixtures for the JS port in midigenai/web: what symusic, MidiTok and modal_serve make of the same inputs.

    python tests/web/make_fixtures.py         # rewrites tests/web/fixtures/; then: node --test tests/web/*.test.mjs

Inputs are the site presets (preselected_midi/), deterministic odd files (`fuzz`: running status, zero-length and
unclosed notes, meter changes, stray events) and multitrack mixes of the presets. For each one it records, as hashes:
symusic's parse written back out, a resample, a clip, the bar lines; MidiTok's ids and their decode; the header.

The tasks run the real MidiGen._generate_batch / _accompany_batch / _infill_batch with the model replaced by a canned
one (see `canned`, which tests/web/tasks.test.mjs mirrors): each draw returns the tokens of another preset, or empty
bars, so the fixture pins everything around the sampling (prompts, budgets, bans, postprocess, retries, overlay,
splice, tempo) without a checkpoint. Needs torch, miditok, symusic and modal importable; no network, no GPU.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import random
import re
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "web" / "fixtures"
PRESETS = ROOT / "preselected_midi"
TOKENIZER = OUT / "tokenizer.json"          # a copy of the hub's v5/tokenizer.json


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids) -> str:
    return sha(json.dumps([int(i) for i in ids], separators=(",", ":")).encode())


def dump(score) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        score.dump_midi(f.name)
        return Path(f.name).read_bytes()


# ---------------------------------------------------------------- inputs

def _vlq(n):
    b = [n & 0x7F]; n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80); n >>= 7
    return bytes(reversed(b))


def _trk(events, running=False):
    out = b""; last = 0; prev = None
    for t, data in sorted(events, key=lambda e: e[0]):
        out += _vlq(t - last); last = t
        if running and data[0] < 0xF0 and data[0] == prev:
            out += data[1:]
        else:
            out += data
        prev = data[0] if data[0] < 0xF0 else prev
    out += _vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(out)) + out


def fuzz(seed: int) -> bytes:
    """A small odd MIDI file: several tpqs, channels, meters, tempos, running status, CCs, bends, lyrics."""
    r = random.Random(seed)
    tpq = r.choice([96, 120, 220, 384, 480, 960, 1024])
    tracks = []
    for k in range(r.randint(1, 4)):
        ev = []
        if r.random() < .7:
            ev.append((0, b"\xff\x03" + _vlq(5) + r.choice([b"Piano", b"Drums", b"bass ", b"Kit 1", b"Lead\xff"])))
        if k == 0 or r.random() < .2:
            for _ in range(r.randint(0, 3)):
                us = r.randint(300000, 1200000)
                ev.append((r.randint(0, tpq * 16), b"\xff\x51\x03" + us.to_bytes(3, "big")))
            for _ in range(r.randint(0, 3)):
                n, d = r.choice([(4, 4), (3, 4), (6, 8), (2, 4), (7, 8), (5, 4), (12, 8), (2, 2), (5, 16), (9, 8)])
                ev.append((r.choice([0, r.randint(0, tpq * 16), tpq * 4 * r.randint(0, 4)]),
                           b"\xff\x58\x04" + bytes([n, int(math.log2(d)), 24, 8])))
            if r.random() < .3:
                ev.append((r.randint(0, tpq * 8), b"\xff\x59\x02" + bytes([r.randint(-7, 7) & 0xFF, r.randint(0, 1)])))
            if r.random() < .2:
                ev.append((r.randint(0, tpq * 8), b"\xff\x06\x02m1"))
        chans = r.sample([0, 1, 2, 9, 5, 15], r.randint(1, 3))
        t = 0
        for _ in range(r.randint(0, 60)):
            t += r.choice([0, 0, 1, tpq // 4, tpq // 2, tpq, r.randint(0, tpq * 2)])
            ch = r.choice(chans); p = r.choice([36, 38, 42, 60, 62, 64, 67, r.randint(0, 127)])
            x = r.random()
            if x < .08:
                ev.append((t, bytes([0xC0 | ch, r.randint(0, 127)])))
            elif x < .14:
                ev.append((t, bytes([0xB0 | ch, r.choice([7, 64, 10, 1]), r.choice([0, 63, 64, 127])])))
            elif x < .17:
                v = r.randint(-8192, 8191) + 8192
                ev.append((t, bytes([0xE0 | ch, v & 0x7F, v >> 7])))
            elif x < .19:
                ev.append((t, b"\xff\x05\x02la"))
            elif x < .21:
                ev.append((t, bytes([0xA0 | ch, p, 10])))
            else:
                d = r.choice([0, 1, tpq // 8, tpq // 3, tpq, tpq * 3, r.randint(0, tpq * 4)])
                ev.append((t, bytes([0x90 | ch, p, r.randint(1, 127)])))
                if r.random() < .93:
                    ev.append((t + d, bytes([0x80 | ch, p, 64]) if r.random() < .5 else bytes([0x90 | ch, p, 0])))
        tracks.append(_trk(ev, running=r.random() < .5))
    return b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), tpq) + b"".join(tracks)


def mix(names: list[str], tempo: float | None, seed: int) -> bytes:
    """Several presets as one multitrack file (each keeps its own program / drum flag; named after its file)."""
    from symusic import Score, Tempo, TimeSignature
    r = random.Random(seed)
    out = Score(480)
    out.time_signatures.append(TimeSignature(0, 4, 4))
    if tempo:
        out.tempos.append(Tempo(0, tempo))
    for name in names:
        sc = Score(str(PRESETS / name)).resample(480)
        for t in sc.tracks:
            if len(t.notes):
                t = t.copy()
                t.name = r.choice([name.removesuffix(".mid"), "", "Track 1"])
                out.tracks.append(t)
    return dump(out)


def inputs() -> dict[str, bytes]:
    files = {f"preset/{p.name}": p.read_bytes() for p in sorted(PRESETS.glob("*.mid"))}
    for s in range(40):
        files[f"fuzz/{s}"] = fuzz(1000 + s)
    presets = sorted(p.name for p in PRESETS.glob("*.mid"))
    drums = [p for p in presets if "drum" in p or p in ("ableton-909.mid", "ableton-reverse-kick-beat.mid")]
    bass = [p for p in presets if "bass" in p]
    other = [p for p in presets if p not in drums and p not in bass]
    r = random.Random(7)
    for s in range(12):
        pick = [r.choice(other)] + ([r.choice(drums)] if s % 3 != 2 else []) + ([r.choice(bass)] if s % 2 == 0 else [])
        if s % 4 == 3:
            pick.append(r.choice(other))
        r.shuffle(pick)
        files[f"mix/{s}"] = mix(pick, [None, 96.0, 128.0, 174.0][s % 4], s)
    return files


# ---------------------------------------------------------------- core: symusic + MidiTok

def core(files, tok) -> dict:
    from symusic import Score

    from midigenai.attributes import header_for_score
    out = {}
    for name, data in files.items():
        try:
            sc = Score.from_midi(data)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": type(e).__name__}
            continue
        rec = {
            "write": sha(dump(sc)),
            "end": int(sc.end()),
            "downbeats": [int(x) for x in sc.get_downbeats()],
            "resample": sha(dump(sc.resample(480, min_dur=1))),
            "clip": sha(dump(sc.clip(sc.tpq * 2, sc.tpq * 10, clip_end=True))),
        }
        try:
            rec["header"] = header_for_score(sc)
        except Exception as e:  # noqa: BLE001
            rec["header"] = {"error": type(e).__name__}
        try:
            ids = tok(sc).ids
            rec["ids"] = {"n": len(ids), "sha": ids_sha(ids)}
            rec["decode"] = sha(dump(tok.decode(ids)))
        except Exception as e:  # noqa: BLE001
            rec["ids"] = {"error": type(e).__name__}
        out[name] = rec
    return out


# ---------------------------------------------------------------- tasks: modal_serve with a canned model

def bodies_of(tok) -> list[list[int]]:
    from symusic import Score
    return [list(tok(Score(str(p))).ids) for p in sorted(PRESETS.glob("*.mid"))]


def canned(kind: str, case: int, call: int, bars: int, bodies, gen) -> list[int]:
    """The raw tokens a draw returns (tests/web/tasks.test.mjs has the same function). Before the decoder's own
    max_new_tokens cut."""
    ts = gen.tokenizer.vocab["TimeSig_4/4"]
    if kind == "accompany" and (case % 9 == 0 or (case + call) % 4 == 1):
        return [gen.bar_id, ts] * (bars + 2)                        # an empty take: retried
    body = bodies[(case * 7 + call * 3 + 1) % len(bodies)]
    if len(body) >= 200:
        return body + [gen.eos_id]
    return (body * (1 + 8192 // max(1, len(body))))[:8192]


class CannedModel:
    """Stands in for MusicTransformer in MidiGen._batched_generate: sample b gets one-hot logits for seqs[b]."""

    def __init__(self, seqs, vocab, eos):
        self.seqs, self.vocab, self.eos, self.step = seqs, vocab, eos, 0

    def __call__(self, ids, kv_caches=None):
        import torch
        B, T = ids.shape
        logits = torch.zeros(B, T, self.vocab)
        for b in range(B):
            s = self.seqs[b]
            logits[b, -1, s[self.step] if self.step < len(s) else self.eos] = 1e4
        self.step += 1
        return logits, None


def camel(k: str) -> str:
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), k)


def tasks(files, tok_path) -> list[dict]:
    import torch

    from midigenai.accompany_tracks import continuation_source
    from midigenai.generate import Generator
    from midigenai.modal_serve import MidiGen

    Serve = MidiGen._get_user_cls()
    state = {}

    def decode(spec):
        state["specs"].append({k: v for k, v in spec.items() if k != "prompt"} | {
            "prompt": {"n": len(spec["prompt"]), "sha": ids_sha(spec["prompt"])}})
        raw = canned(state["kind"], state["case"], len(state["specs"]) - 1, state["bars"], bodies, gen)
        return raw[:spec["max_new_tokens"]]

    gen = Generator(checkpoint_path=None, tokenizer_path=tok_path, backend="remote", decode=decode)
    gen.device = torch.device("cpu")
    bodies = bodies_of(gen.tokenizer)
    srv = object.__new__(Serve)
    srv.gen, srv.version = gen, "v5"
    batched = srv._batched_generate

    def record_batched(prompt_ids, n_samples, max_new_tokens, temperature, top_k, min_new_tokens=0):
        # what each take's decoder spec carries in the JS port (the torch loop has no spec of its own)
        fed = prompt_ids if prompt_ids[:1] == [gen.bos_id] else [gen.bos_id, *prompt_ids]
        state["specs"] += [{"prompt": {"n": len(fed), "sha": ids_sha(fed)}, "max_new_tokens": max_new_tokens,
                            "temperature": temperature, "top_k": top_k, "min_new_tokens": min_new_tokens,
                            "ban_ids": gen.default_ban_ids()}] * n_samples
        return batched(prompt_ids, n_samples, max_new_tokens, temperature, top_k, min_new_tokens)
    srv._batched_generate = record_batched

    names = [n for n in files if not n.startswith("fuzz/")] + [f"fuzz/{s}" for s in range(0, 40, 4)]
    instruments = [None, "bass", "drums", "piano", "auto", "synth lead", "Rhodes", "strings"]
    cases = []
    for i, name in enumerate(names):
        data = files[name]
        multi = name.startswith("mix/")
        plans = [
            ("generate", {"max_new_tokens": [512, 300, 64][i % 3], "temperature": 1.2, "top_k": 50,
                          "tempo_bpm": [None, 100.0][i % 2], "n_samples": 2,
                          "track": (["0", "1,0", "all", None][i % 4] if multi else None)}),
            ("accompany", {"bars": [8, 4, 2, 16][i % 4], "temperature": 1.0, "top_k": 50, "n_samples": 2,
                           "tempo_bpm": [None, 140.0, None][i % 3], "instrument": instruments[i % len(instruments)],
                           "track": ([None, "all", "0,1", 1][i % 4] if multi else None)}),
            ("infill", {"start": 1 + i % 3, "bars": 1 + i % 4, "temperature": 1.0, "top_k": 50, "n_samples": 2,
                        "tempo_bpm": [None, 90.0][i % 2]}),
        ]
        if i % 10 == 0:          # requests the server refuses, with its message
            plans += [("accompany", {"bars": 4, "track": 7, "n_samples": 1}),
                      ("accompany", {"bars": 4, "track": "x", "n_samples": 1}),
                      ("accompany", {"bars": 4, "instrument": "kazoo", "n_samples": 1}),
                      ("infill", {"start": 0, "bars": 2, "n_samples": 1}),
                      ("infill", {"start": 1, "bars": 5, "n_samples": 1})]
        for kind, params in plans:
            case = len(cases)
            state.update(kind=kind, case=case, specs=[], bars=params.get("bars", 0))
            rec = {"input": name, "kind": kind, "case": case, "params": params}
            try:
                if kind == "generate":
                    src, tracks, tnames = continuation_source(data, params["track"])
                    seqs = [canned(kind, case, b, 0, bodies, gen) for b in range(params["n_samples"])]
                    gen.model = CannedModel(seqs, len(gen.tokenizer.vocab), gen.eos_id)
                    res = srv._generate_batch(src, params["max_new_tokens"], params["temperature"], params["top_k"],
                                              params["tempo_bpm"], params["n_samples"])
                    res |= {"prompt_tracks": tracks, "prompt_track_names": tnames}
                elif kind == "accompany":
                    res = srv._accompany_batch(data, params["bars"], params.get("temperature", 1.0),
                                               params.get("top_k", 50), params["n_samples"], params.get("tempo_bpm"),
                                               params.get("instrument"), params.get("track"))
                else:
                    res = srv._infill_batch(data, params["start"], params["bars"], params.get("temperature", 1.0),
                                            params.get("top_k", 50), params["n_samples"], params.get("tempo_bpm"))
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)
                cases.append(rec)
                continue
            rec["result"] = {camel(k): v for k, v in res.items() if k not in ("midi", "midis")}
            rec["midis"] = [sha(m) for m in res["midis"]]
            rec["specs"] = state["specs"]
            cases.append(rec)
    return cases


def main():
    if not TOKENIZER.exists():
        from huggingface_hub import hf_hub_download
        TOKENIZER.write_bytes(Path(hf_hub_download("nicholasbien/midigenai", "tokenizer.json", subfolder="v5")).read_bytes())
    from midigenai.tokenizer import load_tokenizer
    tok = load_tokenizer(TOKENIZER)
    files = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.json").write_text(json.dumps(
        {k: base64.b64encode(v).decode() for k, v in files.items() if not k.startswith("preset/")}, indent=0) + "\n")
    (OUT / "core.json").write_text(json.dumps(core(files, tok), separators=(",", ":")) + "\n")
    cases = tasks(files, TOKENIZER)
    # gzip: the decoder specs repeat the same id lists case after case (520 KB -> 70 KB)
    (OUT / "tasks.json.gz").write_bytes(gzip.compress(json.dumps(cases, separators=(",", ":")).encode(), 9, mtime=0))
    errors = sum("error" in c for c in cases)
    print(f"{len(files)} inputs, {len(cases)} task cases ({errors} refused)")


if __name__ == "__main__":
    main()
