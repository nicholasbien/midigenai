// The browser runner's pure parts: sampling, the stop rule and the decode loop, on stand-in logits.
// Run: node --test tests/web/
import test from "node:test";
import assert from "node:assert/strict";
import { decodeLoop, rng, sampler, stopper, modelBase, sizes } from "../../midigenai/web/midigen-web.mjs";

const V = 16;
const flat = () => new Float32Array(V);
const spec = over => ({
  prompt: [1, 5], max_new_tokens: 50, max_seq_len: 8192, temperature: 1.0, top_k: null, min_new_tokens: 0,
  eos_id: 2, ban_ids: [], stop_ids: [2], bar_id: 9, timesig_ids: [10, 11], stop_after_bars: null,
  trim_leading_bars: true, seed: 1, ...over,
});

test("a seed repeats its draws, and differs from another seed", () => {
  const a = rng(7), b = rng(7), c = rng(8);
  const xs = Array.from({ length: 5 }, () => a());
  assert.deepEqual(Array.from({ length: 5 }, () => b()), xs);
  assert.notDeepEqual(Array.from({ length: 5 }, () => c()), xs);
  assert.ok(xs.every(x => x >= 0 && x < 1));
});

test("banned ids are never drawn", () => {
  const s = sampler(spec({ ban_ids: [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] }));
  const r = rng(1);
  for (let i = 0; i < 200; i++) assert.equal(s(flat(), i, r), 2);
});

test("EOS is masked for the first min_new_tokens steps only", () => {
  const logits = flat(); logits[2] = 50;                     // EOS overwhelmingly likely
  const s = sampler(spec({ min_new_tokens: 3 }));
  const r = rng(2);
  for (let i = 0; i < 3; i++) assert.notEqual(s(logits, i, r), 2);
  assert.equal(s(logits, 3, r), 2);
});

test("top-k keeps only the k most likely", () => {
  const logits = Float32Array.from({ length: V }, (_, j) => j);      // 15 is likeliest
  const s = sampler(spec({ top_k: 3 }));
  const r = rng(3), seen = new Set();
  for (let i = 0; i < 500; i++) seen.add(s(logits, i, r));
  assert.deepEqual([...seen].sort((a, b) => a - b), [13, 14, 15]);
});

test("a low temperature sharpens toward the argmax", () => {
  const logits = flat(); logits[6] = 1;
  const s = sampler(spec({ temperature: 0.01 }));
  const r = rng(4);
  for (let i = 0; i < 100; i++) assert.equal(s(logits, i, r), 6);
});

test("stop ids stop the decoder after the token", () => {
  const stop = stopper(spec({ stop_ids: [2, 3] }));
  assert.equal(stop(5), false);
  assert.equal(stop(3), true);
});

test("stop_after_bars counts from the first non-Bar, non-TimeSig token", () => {
  const stop = stopper(spec({ stop_after_bars: 2 }));
  // leading bars and a time signature are not counted
  for (const t of [9, 10, 9, 11]) assert.equal(stop(t), false);
  for (const t of [5, 9, 5, 9, 5]) assert.equal(stop(t), false);
  assert.equal(stop(9), true);                              // the third Bar after the start
});

test("without trimming every Bar counts", () => {
  const stop = stopper(spec({ stop_after_bars: 1, trim_leading_bars: false }));
  assert.equal(stop(9), false);
  assert.equal(stop(9), true);
});

test("the loop feeds the prompt once, then one token per step, and stops at EOS", async () => {
  const fed = [];
  const plan = [7, 8, 2, 5];
  let i = 0;
  const step = async toks => {
    fed.push([...toks]);
    const l = new Float32Array(V).fill(-1e9); l[plan[i++]] = 0;
    return l;
  };
  const out = await decodeLoop(step, spec({ prompt: [1, 4, 6] }));
  assert.deepEqual(out, [7, 8, 2]);
  assert.deepEqual(fed, [[1, 4, 6], [7], [8]]);
});

test("the loop stops at max_new_tokens and at the context limit", async () => {
  const step = async () => { const l = new Float32Array(V).fill(-1e9); l[5] = 0; return l; };
  assert.equal((await decodeLoop(step, spec({ max_new_tokens: 4 }))).length, 4);
  // prompt of 2 in a context of 5: positions 2, 3, 4 can still be fed, so 4 ids come out (as generate)
  assert.equal((await decodeLoop(step, spec({ max_new_tokens: 50, max_seq_len: 5 }))).length, 4);
});

test("an aborted signal stops between steps", async () => {
  const ctl = new AbortController();
  const step = async () => { ctl.abort(new Error("stop")); return flat(); };
  await assert.rejects(decodeLoop(step, spec(), { signal: ctl.signal }), /stop/);
});

test("model paths and sizes come from the manifest", () => {
  assert.equal(modelBase("v5-rl", "https://example.org/r/"), "https://example.org/r/v5-rl/onnx/");
  assert.deepEqual(sizes({ files: { gpu: { bytes: 3 }, cpu: { bytes: 4 } } }), { gpu: 3, cpu: 4 });
});
