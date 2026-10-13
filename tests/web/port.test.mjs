// The JS port (midi.mjs, tokenizer.mjs, attributes.mjs, generator.mjs, tasks.mjs) against golden fixtures made
// by the Python code it ports: symusic, MidiTok and modal_serve. Regenerate with tests/web/make_fixtures.py.
// Run: node --test tests/web/*.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { clip, downbeats, parseMidi, resample, scoreEnd, writeMidi } from "../../midigenai/web/midi.mjs";
import { loadTokenizer } from "../../midigenai/web/tokenizer.mjs";
import { headerForScore } from "../../midigenai/web/attributes.mjs";
import { Generator } from "../../midigenai/web/generator.mjs";
import { Midigen } from "../../midigenai/web/tasks.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const FIX = join(HERE, "fixtures");
const json = name => JSON.parse(name.endsWith(".gz") ? gunzipSync(readFileSync(join(FIX, name))).toString("utf8") : readFileSync(join(FIX, name), "utf8"));

const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const idsSha = ids => sha(JSON.stringify(ids));

const presetNames = readdirSync(join(ROOT, "preselected_midi")).filter(n => n.endsWith(".mid")).sort();
const files = new Map(presetNames.map(n => [`preset/${n}`, new Uint8Array(readFileSync(join(ROOT, "preselected_midi", n)))]));
for (const [k, v] of Object.entries(json("inputs.json"))) files.set(k, new Uint8Array(Buffer.from(v, "base64")));
const tok = loadTokenizer(readFileSync(join(FIX, "tokenizer.json"), "utf8"));

test("MIDI files read, write, resample, clip and bar the way symusic does", () => {
  const core = json("core.json");
  let n = 0;
  for (const [name, want] of Object.entries(core)) {
    const data = files.get(name);
    assert.ok(data, `missing input ${name}`);
    if (want.error) { assert.throws(() => parseMidi(data), undefined, name); continue; }
    const sc = parseMidi(data);
    assert.equal(sha(writeMidi(sc)), want.write, `${name}: written back`);
    assert.equal(scoreEnd(sc), want.end, `${name}: end`);
    assert.deepEqual(downbeats(sc), want.downbeats, `${name}: downbeats`);
    assert.equal(sha(writeMidi(resample(sc, 480, 1))), want.resample, `${name}: resample`);
    assert.equal(sha(writeMidi(clip(sc, sc.tpq * 2, sc.tpq * 10, true))), want.clip, `${name}: clip`);
    n++;
  }
  assert.ok(n > 100);
});

test("the tokenizer and the header match MidiTok and attributes.py", () => {
  const core = json("core.json");
  for (const [name, want] of Object.entries(core)) {
    if (want.error) continue;
    const sc = parseMidi(files.get(name));
    if (want.header.error) assert.throws(() => headerForScore(sc), undefined, name);
    else assert.deepEqual(headerForScore(sc), want.header, `${name}: header`);
    if (want.ids.error) { assert.throws(() => tok.encode(sc), undefined, name); continue; }
    const ids = tok.encode(sc);
    assert.equal(ids.length, want.ids.n, `${name}: token count`);
    assert.equal(idsSha(ids), want.ids.sha, `${name}: ids`);
    assert.equal(sha(writeMidi(tok.decode(ids))), want.decode, `${name}: decode`);
  }
});

// make_fixtures.canned: the raw tokens one draw returns
const bodies = presetNames.map(n => tok.encode(parseMidi(files.get(`preset/${n}`))));
function canned(kind, kase, call, bars, gen) {
  const ts = tok.vocab.get("TimeSig_4/4");
  if (kind === "accompany" && (kase % 9 === 0 || (kase + call) % 4 === 1)) {
    const out = [];
    for (let i = 0; i < bars + 2; i++) out.push(gen.barId, ts);
    return out;
  }
  const body = bodies[(kase * 7 + call * 3 + 1) % bodies.length];
  if (body.length >= 200) return [...body, gen.eosId];
  const out = [];
  for (let k = 0; k < 1 + Math.floor(8192 / Math.max(1, body.length)); k++) out.push(...body);
  return out.slice(0, 8192);
}

const OPTS = {
  generate: p => ({ maxNewTokens: p.max_new_tokens, temperature: p.temperature, topK: p.top_k, tempoBpm: p.tempo_bpm, nSamples: p.n_samples, track: p.track }),
  accompany: p => ({ bars: p.bars, temperature: p.temperature ?? 1.0, topK: p.top_k ?? 50, nSamples: p.n_samples, tempoBpm: p.tempo_bpm ?? null, instrument: p.instrument ?? null, track: p.track ?? null }),
  infill: p => ({ start: p.start, bars: p.bars, temperature: p.temperature ?? 1.0, topK: p.top_k ?? 50, nSamples: p.n_samples, tempoBpm: p.tempo_bpm ?? null }),
};

test("continuation, accompaniment and infill do what modal_serve does", async () => {
  const cases = json("tasks.json.gz");
  let state = null;
  const gen = new Generator(tok, async spec => {
    state.specs.push({ ...spec, prompt: { n: spec.prompt.length, sha: idsSha(spec.prompt) } });
    return canned(state.kind, state.kase, state.specs.length - 1, state.bars, gen).slice(0, spec.max_new_tokens);
  });
  const mg = new Midigen(gen);
  const counts = { generate: 0, accompany: 0, infill: 0, refused: 0 };
  for (const c of cases) {
    const where = `case ${c.case} (${c.kind} ${c.input} ${JSON.stringify(c.params)})`;
    state = { kind: c.kind, kase: c.case, bars: c.params.bars ?? 0, specs: [] };
    const run = () => mg[c.kind](files.get(c.input), OPTS[c.kind](c.params));
    if (c.error !== undefined) {
      await assert.rejects(run, err => { assert.equal(err.message, c.error, where); return true; });
      counts.refused++;
      continue;
    }
    const res = await run();
    for (const [k, v] of Object.entries(c.result)) assert.deepEqual(res[k], v, `${where}: ${k}`);
    assert.deepEqual(res.midis.map(sha), c.midis, `${where}: MIDI bytes`);
    const specs = c.kind !== "generate" ? state.specs
      : state.specs.map(s => Object.fromEntries(Object.keys(c.specs[0] || {}).map(k => [k, s[k]])));
    assert.deepEqual(specs, c.specs, `${where}: decoder specs`);
    counts[c.kind]++;
  }
  assert.ok(counts.generate > 50 && counts.accompany > 50 && counts.infill > 50 && counts.refused > 20, JSON.stringify(counts));
});
