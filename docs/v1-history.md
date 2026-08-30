# v1 history: the GPT-2 fine-tune, and why v2 replaced it

Preserved from the original README when the repo dropped v1 support.
v1 code lives in the git history and in the archived
[openmusenet](https://github.com/nicholasbien/openmusenet) repo.

## v1 → v2 at a glance

| | v1 | v2 (current: 113M) |
|---|---|---|
| Base model | GPT-2 (HuggingFace fine-tune) | Custom GPT-style transformer (from scratch) |
| Parameters | 124M | **113M** |
| Vocabulary | GPT-2 BPE (50,257) | Custom event vocab (**641**) |
| Tokenization | BPE on `pitch_start_dur_vel` text | [MidiTok MIDILike](https://github.com/Natooz/MidiTok) (event-based) |
| Tokens per note | ~6–10 | **~4** |
| Positional encoding | Learned absolute | RoPE (extrapolates beyond training length) |
| Activation / norm | GELU + LayerNorm | SwiGLU + RMSNorm |
| Attention | HF default | FlashAttention-2 via SDPA |
| Tempo handling | Encoded in text | Stripped at training, re-applied at decode (tempo-invariant learning) |
| Training data | Lakh full | Lakh + MAESTRO + POP909 + GiantMIDI + LAMD, deduped |
| Training infra | manual | one-shot Lambda Labs bootstrap (`v2/setup_lambda.sh`) |
| Eval | informal | quantitative metrics + A/B grading UI |

### Quantitative comparison (n=5 prompts, t=1.0)

Generated continuations from the same 5 prompt MIDIs, measured on the output:

| Metric | v1 | v2 (25M pilot) | Δ |
|---|---|---|---|
| Notes generated | 78 | 193 | +148% |
| Pitch class entropy | 1.91 | 2.63 | +37% |
| Pitch range (semitones) | 28 | 39 | +39% |
| Polyphony rate | 2.68 | 3.03 | +13% |
| **Repetition rate** (4-gram) | 0.48 | **0.19** | **−60%** |
| Inter-onset interval entropy | 1.51 | 2.26 | +50% |
| Scale consistency | 0.98 | 0.94 | −4% |

**Net:** v2 produces meaningfully more music with much less repetition, broader expressive range, and more rhythmic variety, while remaining tonally coherent.

### Inference latency (Mac)

| | tok/s | notes/s |
|---|---|---|
| v1 (124M) on CPU | 69 | ~9 |
| v1 (124M) on MPS | 93 | ~12 |
| v2 (25M) on CPU | **278** | **~70** |
| v2 (25M) on MPS | 172 | ~43 |

v2 is **~8x faster than v1 in actual notes/sec on CPU** (smaller model + 2x more efficient tokenization). MPS hurts v2 because per-kernel dispatch overhead dominates at this size — set `OMN_USE_MPS=1` to override.

---


## Detailed design and next steps (v1)

The v1 design notes from the original README are preserved below for reference. Items resolved by v2 are annotated.

### Dataset
- There are lots of other large MIDI datasets out there. Add these to the training data. In experiments with finetuning on the LMD-matched (45,129 MIDI files) vs. the LMD-full (176,581 MIDI files) it is pretty clear that data is the most important lever for improving generations. *(Resolved in v2: corpus expanded to ~500k MIDIs across 5 sources with content-hash dedup.)*
- Currently only the first max_sequence_length tokens of each MIDI file are parsed. Chunking MIDI files up into max_sequence_length-size chunks would increase the training data volume significantly (~100x). *(Resolved in v2: sliding-window training over packed shards.)*
- Support multiple tracks. Many of the MIDI files in the dataset have multiple tracks. Currently only the first track in the file is processed. *(Resolved in v2: `use_programs=True` interleaves multi-instrument tokens into a single autoregressive stream.)*
- There is probably some leakage due to near-duplicates MIDI files in both the training and validation sets. *(Partially resolved in v2: exact-content dedup via SHA1 over `(pitch, start, duration)` tuples. Near-duplicate detection is still future work.)*

### Encoding
- There is a minor mistake in the encoding: note velocity is encoded as a float instead of an int. *(Resolved in v2: 32 velocity bins as discrete tokens.)*
- Timing is encoded in seconds. The model learns timing pretty well, but it is at times imprecise. Encoding the start of each measure or beat could help. *(Resolved in v2: timing in beat-relative ticks via TimeShift tokens, tempo stripped at training time and re-applied at decode.)*
- A preferred method of encoding timing might be to use one beat as the timing unit instead of one second and additionally encode the BPM. *(Resolved in v2: exactly this scheme.)*

### Tokenizer
- v1 uses the default GPT-2 tokenizer. Experiments with custom tokenizers gave worse results, likely due to small dataset size. *(Resolved in v2: 641-token event-based MIDILike vocab via MidiTok, ~2x more efficient than v1's BPE.)*
- GPT-3+ tokenizers have improvements to numerical encoding that would help with timing. *(Sidestepped in v2: timing is no longer represented as floating-point text, so tokenizer numerical handling doesn't matter.)*

### Model
- v1 uses GPT-2 small (~117M parameters) because it's open source and easy to train/inference on 1 modest GPU. Llama-7B was too cost-prohibitive. *(v2 trains a 25M custom model from scratch — smaller, faster, better musical quality, ~$1.50 to train on Lambda.)*

### Finetuning
- Finetuning is better than just prompting because input/output is not similar to natural language. The base GPT-2 model gives funny outputs like:
    - ```54.19_04.15_99.8 53_19.44_05.28_94.16 ... Now, let's take a look at the numbers ...```
    - ```26.42_97.23_84.64 ... Quotes are not sourced from all markets...```
- v1 finetunes from GPT-2; v2 trains from scratch since the music vocab is small enough that we don't need a strong language prior. The text prior actively wastes capacity in v1.

### Generation
- Generated MIDI = user prompt + model response. Response is usually after the prompt but sometimes is overlaid on top of the prompt.
- Sequence length is limited by GPT-2 model size (1024 tokens) — long enough for 1-2 minutes of single-line music, but only a few measures of chords. *(Resolved in v2: 2048 context, with RoPE that extrapolates beyond training length.)*
- GPU inference via Modal: ~10 seconds end-to-end on v1. *(v2 is ~8x faster in notes/sec, see latency table above.)*
- Generation parameters
    - Temperature: ~1.0–1.2 sweet spot for v1 melodies; ~1.0 with top-k 50 is the v2 sweet spot.
    - Top_k: ~10 for v1 (smaller vocab made larger top-k unstable). v2 handles top-k 50 fine.

### RLHF
- In the web interface, 2 responses are generated for each user prompt. The user has the option to pick which generation they prefer. The generations and user preferences are stored and can be used later to train a reward model as in equation 1 from "Training language models to follow instructions with human feedback" (https://arxiv.org/abs/2203.02155). *(v2 keeps this loop and adds a dedicated A/B grading frontend at `v2/grade_app.py` that randomizes left/right and writes a CSV compatible with reward-model training.)*
