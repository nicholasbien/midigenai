#!/usr/bin/env bash
# Build the pilot corpus with the full modern pipeline:
#   download -> clean (quality filters) -> cross-source near-dup dedup ->
#   tokenize/shard (EOS endings, track views, silence trim, per-source tags)
#
# Defaults to Lakh + the curated sets (small enough to build locally in a few
# hours). For the big run, add: lamd gigamidi aria — and run it on the
# training box instead.
#
#   DATA_ROOT=~/midigenai_data SOURCES="lakh maestro pop909 giantmidi" \
#       bash midigenai/data/build_pilot_corpus.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/midigenai_data}"
SOURCES="${SOURCES:-lakh maestro pop909 giantmidi}"
PY="${PYTHON:-python}"
OUT="${OUT:-$DATA_ROOT/corpus_pilot}"

mkdir -p "$DATA_ROOT"
echo "==> [1/5] download: $SOURCES"
$PY -m midigenai.data.download --root "$DATA_ROOT/raw" --datasets $SOURCES

echo "==> [2/5] clean per source"
for s in $SOURCES; do
  $PY -m midigenai.data.clean \
    --input "$DATA_ROOT/raw/$s" \
    --manifest "$DATA_ROOT/manifest_$s.jsonl"
done

echo "==> [3/5] cross-source near-dup dedup"
cat $(for s in $SOURCES; do echo "$DATA_ROOT/manifest_$s.jsonl"; done) \
  > "$DATA_ROOT/manifest_all.jsonl"
$PY -m midigenai.data.dedup \
  --manifest "$DATA_ROOT/manifest_all.jsonl" \
  --out "$DATA_ROOT/manifest_all_dedup.jsonl" \
  --clusters "$DATA_ROOT/dup_clusters.json"

echo "==> [4/5] split deduped manifest back per source"
DATA_ROOT="$DATA_ROOT" SOURCES="$SOURCES" $PY - <<'PYEOF'
import json, os
root = os.environ["DATA_ROOT"]
srcs = os.environ["SOURCES"].split()
outs = {s: open(f"{root}/manifest_{s}_dedup.jsonl", "w") for s in srcs}
for line in open(f"{root}/manifest_all_dedup.jsonl"):
    path = json.loads(line)["path"]
    for s in srcs:
        if f"/raw/{s}/" in path:
            outs[s].write(line)
            break
PYEOF

echo "==> [5/5] tokenize + shard (tagged per source)"
for s in $SOURCES; do
  $PY -m midigenai.data.build_dataset \
    --manifest "$DATA_ROOT/manifest_${s}_dedup.jsonl" \
    --out "$OUT" --tag "$s"
done

echo "==> done: $OUT"
ls -lh "$OUT/shards" | head -20
