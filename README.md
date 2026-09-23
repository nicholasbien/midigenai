# midigenai

Real-time AI MIDI generation: play a phrase and get the next one back, or get a new part to play along with it.

A 113M-parameter transformer trained from scratch on ~800k MIDI files plus audio transcriptions, then fine-tuned with reinforcement learning against a preference model. Demo at https://nicholasbien.com/midi.

## Quickstart

```bash
pip install git+https://github.com/nicholasbien/midigenai
```

```python
from midigenai import load_from_hub

gen = load_from_hub()                       # default checkpoint from the Hub
prompt = gen.encode_midi_file("riff.mid")
gen.generate_to_midi(prompt, "continuation.mid", max_new_tokens=512)
```

Streaming, note by note:

```python
for note in gen.stream_notes(prompt, tempo_bpm=120):
    ...  # {pitch, start, end, velocity, program}
```

### Accompaniment

Writes a new part over the same bars as the input instead of after it. Name the
instruments the finished piece should contain, the input's plus the ones to add:

```python
cond = gen.tokenizer(condition_score).ids
header = gen.make_header(instruments=["Piano", "Bass"])
new_part = list(gen.accompany(cond, bars=8, header=header))
```

Over HTTP: `/api/accompany?instrument=bass` (or `auto`).

## Checkpoints

| Version | Params | Notes |
|---|---|---|
| **`v5-rl`** (default) | 113M | v5 after GRPO. Preferred over v5 in blind A/B: 13–1 continuation, 8–1 accompaniment |
| `v5` | 113M | base model: rebuilt corpus, audio transcriptions, drum-label fix |
| `v4` | 113M | previous default |
| `v4-large` | 202M | v4 at 202M; near-even with v4 in blind labels |

Load any of them with `load_from_hub(version=...)` or `MIDIGENAI_VERSION`. v5 checkpoints use a
598-token vocabulary, v4 checkpoints 590; each version ships its own `tokenizer.json`.
Sampling defaults: temperature 1.0, top_k 50.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer: RoPE, SwiGLU, RMSNorm, tied embeddings |
| Tokenization | [MidiTok REMI](https://github.com/Natooz/MidiTok) (bar / position / duration) plus an attribute header (instruments, density, polyphony, range, tempo, source, genre) |
| Context | 2048 tokens trained, longer at inference |
| Data | Lakh, LAMD, Aria, GigaMIDI, MAESTRO, POP909, GiantMIDI, and ~7k MuScriptor transcriptions of Free Music Archive audio |
| Fine-tuning | GRPO against a reward model fit to LLM-judge labels validated on human preferences |

On Apple silicon the MLX backend decodes at ~800 tokens/s with 50 ms to first token.

## Training

```bash
python -m midigenai.data.build_dataset --manifest ... --out ... --exclude-ids ...
python -m midigenai.modal_launch spawn --run-name <name> --corpus <corpus> --size medium --resume
```

Launch with `--resume` so a preempted run picks up from its last checkpoint. What went into v5 and
how it was measured is in [docs/v5.md](docs/v5.md); design proposals are in `docs/proposals/`.

## License

MIT, see LICENSE.
