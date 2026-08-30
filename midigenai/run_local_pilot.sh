#!/usr/bin/env bash
# End-to-end local pilot run on Mac MPS.
#
# Goal: validate the full pipeline (data → clean → tokenize → train → checkpoint)
# end-to-end before paying for Lambda. Produces a working 25M checkpoint we
# can exercise via generate.py for sanity-listening.
#
# Defaults tuned for Apple Silicon with ~16-32GB unified memory.
# Override with env vars: MAX_STEPS=5000 BATCH_SIZE=4 bash run_local_pilot.sh

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

DATA_ROOT="${DATA_ROOT:-data}"
RUNS_DIR="${RUNS_DIR:-runs}"
DATASETS="${DATASETS:-lakh}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "venv missing — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

echo "==> download datasets: $DATASETS (skips already-downloaded)"
mkdir -p "$DATA_ROOT/raw"
$PY -m midigenai.data.download --root "$DATA_ROOT/raw" --datasets $DATASETS

echo "==> clean + dedup"
$PY -m midigenai.data.clean \
  --input "$DATA_ROOT/raw" \
  --manifest "$DATA_ROOT/manifest.jsonl"

echo "==> tokenize + shard"
$PY -m midigenai.data.build_dataset \
  --manifest "$DATA_ROOT/manifest.jsonl" \
  --out "$DATA_ROOT/v2_corpus"

echo "==> train pilot ($MAX_STEPS steps, batch=$BATCH_SIZE, accum=$GRAD_ACCUM, block=$BLOCK_SIZE)"
mkdir -p "$RUNS_DIR"
RUN_NAME="pilot_local_$(date +%Y%m%d_%H%M%S)"
$PY -m midigenai.train \
  --data "$DATA_ROOT/v2_corpus" \
  --out "$RUNS_DIR/$RUN_NAME" \
  --size pilot \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --block-size "$BLOCK_SIZE" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 100 \
  --eval-interval 200 \
  --save-interval 500 \
  --dtype float32

echo "==> done. Checkpoints in $RUNS_DIR/$RUN_NAME"
