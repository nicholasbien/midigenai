// Runs midigen-prompt.mjs on cases from tests/test_web_prompt.py: a JSON list of {op, ...} on stdin, one result each
// on stdout, so the test can compare them with what MidiTok, symusic and Generator give in Python.
import { readFileSync } from "node:fs";
import {
  Tokenizer, Generator, barEdges, windowOf, headerForScore, infillPlan, promptSegments, splice, normalizeDrums,
  copyScore, withInstruments, sortHeader, fitToContext,
} from "../../midigenai/web/midigen-prompt.mjs";

const cases = JSON.parse(readFileSync(0, "utf8"));
const toks = new Map();
const tokenizer = (params) => {
  const key = JSON.stringify(params);
  if (!toks.has(key)) toks.set(key, new Tokenizer(params));
  return toks.get(key);
};
const gen = (params) => new Generator(tokenizer(params), () => { throw new Error("no decoder here"); });

function run(c) {
  switch (c.op) {
    case "vocab": return tokenizer(c.tokenizer).tokens;
    case "encode": return tokenizer(c.tokenizer).encode(c.score);
    case "decode": return tokenizer(c.tokenizer).decode(c.ids);
    case "postprocess": return gen(c.tokenizer).postprocess(c.prompt, c.raw, { stopAfterBars: c.stop, trimLeadingBars: c.trim });
    case "pad": return gen(c.tokenizer).padToBars(c.ids, c.bars);
    case "ends_on_bar_line": return gen(c.tokenizer).endsOnBarLine(c.ids);
    case "fit": return fitToContext(c.ids, c.max_new, c.max_seq);
    case "bar_edges": return barEdges(c.score);
    case "window": return windowOf(c.score, c.start, c.end);
    case "header": return headerForScore(c.score, { tempo: c.tempo });
    case "with_instruments": return withInstruments(c.names, c.families);
    case "sort_header": return sortHeader(c.names);
    case "normalize": { const sc = copyScore(c.score); normalizeDrums(sc, c.hint); return sc.tracks.map((t) => t.is_drum); }
    case "infill": {
      const g = gen(c.tokenizer), sc = copyScore(c.score);
      normalizeDrums(sc, "upload.mid");
      const p = infillPlan(sc, c.start, c.bars);
      const [prefix, suffix] = promptSegments(g, p);
      const header = g.headerIdsFor(headerForScore(p.window(), { tempo: c.tempo }));
      return { prefix, suffix, header, ctx: [p.ctxStart, p.ctxEnd], prompt: g.infillPrompt(header, prefix, suffix),
        spliced: c.answer ? splice(p, g.tokenizer.decode(c.answer)) : null };
    }
    default: throw new Error(`unknown op ${c.op}`);
  }
}

const out = cases.map((c) => {
  try { return { ok: run(c) }; } catch (e) { return { error: `${e.name}: ${e.message}` }; }
});
process.stdout.write(JSON.stringify(out));
