# midigenai

Real-time AI MIDI continuation: play a musical phrase, get the next one back before you've finished listening.

A custom GPT-style transformer (113M params) trained from scratch on ~500k MIDI files with event-based tokenization. Demo at https://nicholasbien.com/midi.

## Quickstart

```bash
pip install git+https://github.com/nicholasbien/midigenai
```

```python
from midigenai import load_from_hub

gen = load_from_hub()                       # pulls the default checkpoint from HF
prompt = gen.encode_midi_file("riff.mid")
gen.generate_to_midi(prompt, "continuation.mid", max_new_tokens=512)
```

Or from the command line:

```bash
python -m midigenai.generate \
    --checkpoint <ckpt.pt> --tokenizer <tokenizer.json> \
    --input-midi riff.mid --output-midi continuation.mid
```

Streaming, note-by-note output for real-time use:

```python
for note in gen.stream_notes(prompt, tempo_bpm=120):
    ...  # {pitch, start, end, velocity, program}, emitted as the model plays
```

## Live jamming with Ableton

`python -m midigenai.live_session` turns Live into a call-and-response partner:
session-record a phrase on any track and the model answers on its own track
within ~0.5 s, matching your phrase length in bars and staying aware of the
whole session so far. Requires Live with the
[AbletonMCP](https://github.com/nicholasbien/ableton-mcp-pro) control surface.
Tip: set Live's launch quantization (the Q dropdown) to 1/4 so answers start
on the next quarter note.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer: RoPE, SwiGLU, RMSNorm, SDPA (FlashAttention-2), tied embeddings |
| Parameters | 113M (`medium`; 25M `pilot` and 200M `production` configs in `model.py`) |
| Vocabulary | 641 event tokens ([MidiTok MIDILike](https://github.com/Natooz/MidiTok)): NoteOn/Off, 32 velocity bins, TimeShift |
| Context | 2048 tokens trained, longer at inference (RoPE extrapolates) |
| Timing | beat-relative ticks; tempo stripped at training, re-applied at decode (tempo-invariant learning) |
| Training data | Lakh + LAMD + MAESTRO + POP909 + GiantMIDI (~408k files after dedup, ~8B tokens) |

### Checkpoints

| Tag | Params | Final loss | Best sampling | HF |
|---|---|---|---|---|
| `v2-pilot` | 25M | 0.97 | t=1.0, top_k=50 | [tree/main/v2-pilot](https://huggingface.co/nicholasbien/midigenai/tree/main/v2-pilot) |
| `v2-production` | 25M | 0.93 | t=1.0, top_k=50 | [tree/main/v2-production](https://huggingface.co/nicholasbien/midigenai/tree/main/v2-production) |
| `v2-100m` | 113M | 0.71 | t=1.2, top_k=50 | [tree/main/v2-100m](https://huggingface.co/nicholasbien/midigenai/tree/main/v2-100m) |
| **`v3`** ← default | **113M** | **0.75 val** | **t=1.1, top_k=50** | [tree/main/v3](https://huggingface.co/nicholasbien/midigenai/tree/main/v3) |

Select with `load_from_hub(version=...)` or `MIDIGENAI_VERSION`.

## Performance

Inference is fast enough that playback, not generation, is the bottleneck
(v2-100m on an M3 Max, fp16):

| | decode | TTFT @ 2048-token prompt |
|---|---|---|
| MLX backend (default on Apple silicon) | **~800 tok/s** (~200 notes/s) | **50 ms** |
| PyTorch CPU (fallback) | ~270 tok/s | ~2 s |

The MLX backend (`model_mlx.py`) runs the same weights on the Mac GPU with a
preallocated KV cache, one-step-ahead pipelined decoding, and a cached
`.safetensors` sidecar for ~15 ms model loads. Outputs are token-identical to
the PyTorch path under greedy decoding. `Generator(backend=...)` /
`--backend` selects `auto` (default), `mlx`, or `torch`; `--dtype` selects
float16 (default) / bfloat16 / float32.

## Repo layout

```
midigenai/
├─ model.py           # the transformer (+ pilot/medium/production configs)
├─ model_mlx.py       # MLX (Apple-silicon GPU) implementation of the same
├─ tokenizer.py       # MidiTok MIDILike wrapper
├─ generate.py        # Generator: streaming inference, KV cache, tempo plumbing
├─ live_session.py    # interactive Ableton call-and-response
├─ hub.py             # HuggingFace checkpoint download / load_from_hub
├─ train.py           # training loop (sliding-window, AdamW, cosine LR, bf16)
├─ modal_train.py     # Modal H100/A100 training entrypoint
├─ eval.py            # quantitative metrics (any two output dirs)
├─ grade_app.py       # A/B preference grading frontend (Flask)
├─ data/              # dataset download / clean / dedup / tokenize / shard
├─ run_local_pilot.sh
```

## Training

```bash
python -m midigenai.data.download --root data/raw
python -m midigenai.data.clean ...
python -m midigenai.data.build_dataset ...
modal run midigenai/modal_train.py --size medium --max-steps 15000 --gpu H100
```

The 113M model trained on a Modal
H100; the 25M pilot cost ~$1.50. See `modal_train.py` for the flow.

## History

midigenai supersedes [openmusenet](https://github.com/nicholasbien/openmusenet),
which fine-tuned GPT-2 on text encodings of MIDI. Training a small custom
model from scratch beat the fine-tune on every axis that matters: ~2.5x fewer
tokens per note (641-token event vocab vs GPT-2 BPE shredding numbers), 60%
less repetition, wider pitch/rhythm range, and an order of magnitude faster
inference. The v1 design notes and the full v1-vs-v2 comparison live in
[docs/v1-history.md](docs/v1-history.md).

## License

MIT — see LICENSE.
