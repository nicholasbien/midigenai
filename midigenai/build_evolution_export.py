"""
Build a fully self-contained, shareable training-evolution page: one HTML
file with every MIDI inlined as a base64 data-URI and the grid manifest
embedded. Needs internet only for the player/visualizer JS (jsDelivr CDN);
visuals work regardless, realistic piano soundfont streams when online.

    python -m midigenai.build_evolution_export \\
        --evolution evals/dataset_samples/evolution \\
        --out evals/evolution_export.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>midigenai — training evolution</title>
<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }
  h1 { font-size: 20px; margin-bottom: 2px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 14px; }
  .controls { margin-bottom: 14px; display: flex; gap: 10px; align-items: center; }
  button { padding: 7px 12px; border: 1px solid #ccc; background: white; border-radius: 6px; cursor: pointer; font-size: 13px; }
  button.on { background: #4f46e5; color: white; border-color: #4f46e5; }
  .grid { display: grid; gap: 10px; overflow-x: auto; }
  .cell { border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fafafa; min-width: 300px; }
  .cell h4 { margin: 0 0 6px; font-size: 12px; color: #444; }
  .cell.prompt-cell { background: #eef2ff; }
  midi-player { width: 100%; }
  midi-visualizer { display: block; background: white; border-radius: 4px; margin-top: 6px; overflow: auto; max-height: 140px; }
  .meta { font-size: 11px; color: #777; margin-top: 4px; }
  .eos { color: #047857; font-weight: 600; }
  .colhead { font-size: 12px; font-weight: 700; color: #333; align-self: end; padding: 4px; }
</style>
</head>
<body>
  <h1>midigenai — training evolution</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="controls">
    view: <button id="btn-roll" class="on" onclick="setMode('piano-roll')">piano roll</button>
    <button id="btn-staff" onclick="setMode('staff')">notation</button>
  </div>
  <div id="grid" class="grid"></div>
<script>
const DATA = __DATA__;
let mode = 'piano-roll';
function setMode(m) {
  mode = m;
  document.getElementById('btn-roll').classList.toggle('on', m === 'piano-roll');
  document.getElementById('btn-staff').classList.toggle('on', m === 'staff');
  document.querySelectorAll('midi-visualizer').forEach(v => v.setAttribute('type', m));
}
function cellHtml(key, title, meta, id) {
  const src = 'data:audio/midi;base64,' + DATA.midi[key];
  return `<div class="cell${title === 'prompt' ? ' prompt-cell' : ''}">
    <h4>${title}</h4>
    <midi-player src="${src}" sound-font visualizer="#viz-${id}"></midi-player>
    <midi-visualizer id="viz-${id}" type="piano-roll" src="${src}"></midi-visualizer>
    <div class="meta">${meta}</div>
  </div>`;
}
function render() {
  const steps = DATA.steps;
  const grid = document.getElementById('grid');
  grid.style.gridTemplateColumns = `repeat(${steps.length + 1}, minmax(300px, 1fr))`;
  let html = `<div class="colhead">prompt (val set)</div>` +
    steps.map(s => `<div class="colhead">step ${(+s).toLocaleString()} · ${(s*131072/1e9).toFixed(2)}B tokens seen</div>`).join('');
  let cid = 0;
  for (const p of DATA.prompts) {
    html += cellHtml(p.name + '_prompt', 'prompt', `${p.source} · ${p.file}`, cid++);
    for (const s of steps) {
      const cell = (DATA.cells[s] || {})[p.name];
      if (cell) {
        const eos = cell.ended_via_eos ? ' · <span class="eos">ended itself (EOS)</span>' : '';
        html += cellHtml(cell.file.replace('.mid',''), `step ${(+s).toLocaleString()}`,
                         `${cell.n_tokens} tokens${eos}`, cid++);
      } else {
        html += `<div class="cell"><h4>step ${(+s).toLocaleString()}</h4><div class="meta">not generated</div></div>`;
      }
    }
  }
  grid.innerHTML = html;
}
render();
</script>
</body>
</html>
"""


def build(evo_dir: Path, out_path: Path, run_name: str = "medium_full_v1") -> None:
    prompts = json.loads((evo_dir / "prompts.json").read_text())
    prompts = [{k: p[k] for k in ("name", "source", "file")} for p in prompts]

    steps, cells, midi = [], {}, {}
    for mf in sorted(evo_dir.glob("manifest_*.json")):
        step = str(int(mf.stem.split("_")[1]))
        steps.append(step)
        cells[step] = {r["prompt"]: r for r in json.loads(mf.read_text())}
    for f in evo_dir.glob("*.mid"):
        midi[f.stem] = base64.b64encode(f.read_bytes()).decode()

    data = {"run": run_name, "prompts": prompts, "steps": steps,
            "cells": cells, "midi": midi}
    subtitle = (f"Run {run_name}: {len(prompts)} validation prompts x "
                f"{len(steps)} checkpoints (steps {', '.join(steps)}). Same prompt, "
                "same sampling seed per column - only the amount of training changes. "
                "Self-contained: all audio embedded.")
    html = TEMPLATE.replace("__DATA__", json.dumps(data)) \
                   .replace("__SUBTITLE__", subtitle)
    out_path.write_text(html)
    print(f"[export] {out_path} ({out_path.stat().st_size/1024:.0f} KB, "
          f"{len(midi)} MIDIs, {len(steps)} checkpoint columns)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evolution", type=Path,
                   default=Path("evals/dataset_samples/evolution"))
    p.add_argument("--out", type=Path, default=Path("evals/evolution_export.html"))
    p.add_argument("--run-name", default="medium_full_v1")
    args = p.parse_args()
    build(args.evolution, args.out, args.run_name)
