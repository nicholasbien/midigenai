#!/usr/bin/env bash
# Bootstrap a fresh Lambda Labs H100 instance for openmusenet2 v2 training.
#
# Usage (after SSHing in):
#   curl -L https://raw.githubusercontent.com/nicholasbien/openmusenet/v2-scaffold/v2/setup_lambda.sh | bash
# Or:
#   git clone -b v2-scaffold git@github.com:nicholasbien/openmusenet.git
#   cd openmusenet && bash v2/setup_lambda.sh
#
# Defaults:
#   - clones repo to ~/openmusenet
#   - downloads Lakh + MAESTRO + POP909 + GiantMIDI into ~/data/raw
#   - cleans, tokenizes, shards into ~/data/v2_corpus
#   - kicks off pilot training to ~/runs/pilot
#
# Override with env vars:
#   REPO_DIR=/workspace/openmusenet  DATASETS="lakh maestro" SIZE=production bash setup_lambda.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/openmusenet}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
RUNS_DIR="${RUNS_DIR:-$HOME/runs}"
DATASETS="${DATASETS:-lakh maestro pop909 giantmidi}"
BRANCH="${BRANCH:-v2-scaffold}"
SIZE="${SIZE:-pilot}"            # "pilot" or "production"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"

echo "==> system info"
nvidia-smi || echo "WARNING: nvidia-smi failed — are you on a GPU instance?"
python3 --version

echo "==> ensure repo at $REPO_DIR"
if [ ! -d "$REPO_DIR" ]; then
  git clone -b "$BRANCH" https://github.com/nicholasbien/openmusenet.git "$REPO_DIR"
elif [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR" && git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull
else
  echo "[skip] $REPO_DIR exists without .git — assuming code was rsynced in"
fi
cd "$REPO_DIR"

echo "==> create venv"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> download datasets: $DATASETS"
mkdir -p "$DATA_ROOT/raw"
python -m v2.data.download --root "$DATA_ROOT/raw" --datasets $DATASETS

echo "==> clean (quality filters + exact dedup)"
python -m v2.data.clean \
  --input "$DATA_ROOT/raw" \
  --manifest "$DATA_ROOT/manifest.jsonl"

echo "==> near-duplicate dedup (MinHash over pitch-interval n-grams)"
python -m v2.data.dedup \
  --manifest "$DATA_ROOT/manifest.jsonl" \
  --out "$DATA_ROOT/manifest_deduped.jsonl" \
  --clusters "$DATA_ROOT/dup_clusters.json"

echo "==> tokenize + shard"
python -m v2.data.build_dataset \
  --manifest "$DATA_ROOT/manifest_deduped.jsonl" \
  --out "$DATA_ROOT/v2_corpus"

echo "==> launch training in tmux (size=$SIZE max_steps=$MAX_STEPS)"
mkdir -p "$RUNS_DIR"
RUN_NAME="${SIZE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/train.log"

# Make sure tmux is available
if ! command -v tmux >/dev/null; then
  sudo apt-get install -y tmux
fi

# Detach training inside tmux so it survives SSH drops.
TMUX_SESSION="${TMUX_SESSION:-train}"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" -- bash -c "
  cd $REPO_DIR && source .venv/bin/activate && \
  python -m v2.train_v2 \
    --data '$DATA_ROOT/v2_corpus' \
    --out '$RUN_DIR' \
    --size '$SIZE' \
    --batch-size '$BATCH_SIZE' \
    --grad-accum '$GRAD_ACCUM' \
    --block-size '$BLOCK_SIZE' \
    --max-steps '$MAX_STEPS' \
    --dtype bfloat16 \
    --compile 2>&1 | tee '$LOG_FILE'
"

echo "==> training started in tmux session '$TMUX_SESSION'"
echo "    attach:    tmux attach -t $TMUX_SESSION"
echo "    log tail:  tail -f $LOG_FILE"
echo "    run dir:   $RUN_DIR"
