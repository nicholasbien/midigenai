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

### Infill

Rewrites some bars and keeps the rest: the model sees the bars before and after
the gap and fills it.

```python
ids = gen.tokenizer(score).ids
prefix, suffix = gen.split_bars(ids, at_bar=2, n_bars=2)   # redo bars 3-4
middle = list(gen.infill(prefix, suffix, bars=2))
```

Over HTTP: `/api/infill?start=2&bars=2` returns the whole file with bars 3-4
replaced. Spans of 1-4 bars with at least one kept bar on each side, the shape
the model was trained on.

### In the browser

The model also runs in a web page, on [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/): WebGPU
where the browser has it, WebAssembly otherwise. Everything the server does around the model is in JS too
(`midigenai/web/`, shipped with the package), so a page needs no server at all:

```js
import * as ort from "./ort.webgpu.bundle.min.mjs";     // onnxruntime-web 1.30
import { detect } from "./midigen-web.mjs";
import { loadMidigen } from "./tasks.mjs";

const mg = await loadMidigen({ ort, engine: (await detect()).recommended, onProgress });
const cont = await mg.generate(midiBytes, { maxNewTokens: 512 });              // continuation
const acc = await mg.accompany(midiBytes, { bars: 8, instrument: "bass" });    // parts to play along
const inf = await mg.infill(midiBytes, { start: 2, bars: 2 });                 // redo bars 3-4
play(acc.midis[0]);                                                             // MIDI file bytes
mg.tracks(midiBytes);                                                           // what a track picker needs
```

Each takes the options and returns the fields of the matching `/api` route (`track`, `tempoBpm`, `nSamples`,
`temperature`, `topK`, `seed`; `onToken` and an AbortSignal `signal` for progress and cancelling). The modules:

| File | Ports |
|---|---|
| `midi.mjs` | symusic: reading and writing MIDI, resample, clip, bar lines |
| `tokenizer.mjs` | MidiTok REMI for the v4/v5 `tokenizer.json` (encode and decode) |
| `attributes.mjs` | the attribute header, instrument names, drum-track detection |
| `generator.mjs` | `Generator`: prompt layouts, context fit, bar padding, `postprocess` |
| `tasks.mjs` | `modal_serve`'s continuation / accompaniment / infill, track choice, infill splice |
| `midigen-web.mjs` | the model: download, WebGPU / WebAssembly sessions, sampling |
| `midigen-prompt.mjs` | the same prompt side on plain score data, for a page that parses MIDI itself (below) |

They give the same tokens and byte-identical MIDI files as the Python code on the presets and a set of odd files
(`tests/web/make_fixtures.py` makes the golden files; `node --test tests/web/*.test.mjs` checks them). Training,
evals, data and the Ableton tools stay in Python.

Python can also drive the browser model: a `remote` Generator hands each generation to a decoder you supply.

```python
from midigenai import remote_from_hub

gen = remote_from_hub(decode=ask_the_page)   # decode(spec) -> sampled ids; spec is plain JSON
ids = list(gen.generate_ids(gen.encode_midi_file("riff.mid"), max_new_tokens=512, stop_after_bars=4))
```

The model files are on the Hub under `VERSION/onnx/` (`python -m midigenai.export_onnx --version v5 --out DIR`
makes them; `pip install "midigenai[onnx]"`). Each is downloaded once and kept in the browser's cache:

| Engine | Weights | Download | 500-token prompt + 256 new |
|---|---|---|---|
| `gpu` (WebGPU) | 4-bit | 73 MB | 2.8 s on an M-series Mac in Chrome |
| `cpu` (WebAssembly) | 8-bit | 115 MB | 7.3 s on that Mac, 6.0 s on an iPhone (one thread) |

plus onnxruntime-web's own 27 MB `.wasm`. Both agree with the fp32 model on 99% of next tokens. A seed repeats
a result in the browser but not torch's (each has its own random numbers).

Pages that already hold a parsed score (rather than MIDI bytes) can use
[`midigenai/web/midigen-prompt.mjs`](midigenai/web/midigen-prompt.mjs) instead: the `remote` Generator's side on
plain score data (symusic's field names), checked against MidiTok and symusic by `tests/test_web_prompt.py`. Either
layer's decoder can be the page's model or a GPU: the deployed `MidiGen.decode(spec)` samples a spec with the full
model (`Generator.decode_spec`), so one prompt runs either place.

```js
import { loadGenerator, headerForScore } from "./midigen-prompt.mjs";
const g = await loadGenerator({ version: "v5", decode: (spec) => model.decode(spec) });   // or a server's decode
const ids = await g.continueIds(g.tokenizer.encode(score), { header: g.headerIdsFor(headerForScore(score, { tempo: 120 })), bars: 4 });
const answer = g.tokenizer.decode(ids);                 // a Score (plain data, symusic's fields)
```

## Checkpoints

| Version | Params | Notes |
|---|---|---|
| **`v5`** (default) | 113M | v5 after GRPO (called `v5-rl` until 2026-10; that name still works). Preferred over the base in blind A/B: 13–1 continuation, 9–1 accompaniment |
| `v5-base` | 113M | v5 before GRPO (called `v5` until 2026-10): rebuilt corpus, audio transcriptions, drum-label fix |
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
