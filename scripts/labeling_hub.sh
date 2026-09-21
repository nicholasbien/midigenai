#!/usr/bin/env bash
# Bring the blind labeling hub back up on localhost: every evals/labeling_*/
# set that has a manifest, served from one process.
#
#   scripts/labeling_hub.sh
#   PORT=7790 scripts/labeling_hub.sh
#
# Sets are discovered on disk and re-scanned while the server runs, so a
# pairgen that finishes later joins the hub without a restart. Listing order
# and hidden sets come from evals/labeling_sets.json. Pair MIDI is gitignored
# and lives only in the worktree the pairs were generated in, so this has to
# run on the machine that holds them.
#
# Exposing the hub beyond localhost (a cloudflared quick tunnel, say) is left
# to the operator: it is a public URL onto local files, so it should be a
# deliberate command you type, not something a script does behind you.
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${PORT:-7789}"
PYTHON="${PYTHON:-python}"
HUB_LOG="evals/hub.log"              # evals/*.log is gitignored

# --- what is on disk -------------------------------------------------------
found=0
for d in evals/labeling_*/; do
  [ -f "$d/manifest.jsonl" ] || continue
  found=1
done
if [ "$found" = 0 ]; then
  echo "no labeled sets found under evals/ (need labeling_*/manifest.jsonl)." >&2
  echo "generate a set first: pairgen, then relabel_app select-dir --pairs <dir> --out evals/labeling_<name>" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import json, pathlib
for d in sorted(pathlib.Path("evals").glob("labeling_*")):
    man = d / "manifest.jsonl"
    if not man.exists():
        continue
    total = sum(1 for l in man.read_text().splitlines() if l.strip())
    decided = 0
    labels = d / "labels.jsonl"
    if labels.exists():
        for l in labels.read_text().splitlines():
            if l.strip() and json.loads(l).get("choice") in ("left", "right"):
                decided += 1
    print(f"  {d.name:<34} {decided:>4} decided / {total} pairs")
PY

# --- the server ------------------------------------------------------------
if curl -fsS -m 2 "http://127.0.0.1:$PORT/api/sources" >/dev/null 2>&1; then
  echo "hub already serving on :$PORT — reusing it"
  echo "hub: http://localhost:$PORT"
  exit 0
fi

nohup "$PYTHON" -m midigenai.relabel_app serve --sets-dir evals --port "$PORT" \
  >"$HUB_LOG" 2>&1 &

for _ in $(seq 30); do
  curl -fsS -m 2 "http://127.0.0.1:$PORT/api/sources" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -fsS -m 2 "http://127.0.0.1:$PORT/api/sources" >/dev/null 2>&1; then
  echo "hub did not come up on :$PORT — last lines of $HUB_LOG:" >&2
  tail -20 "$HUB_LOG" >&2
  exit 1
fi

echo "hub:  http://localhost:$PORT"
echo "logs: $HUB_LOG"
