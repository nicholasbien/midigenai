// midigen-web: midigenai's decoder in a web page, on onnxruntime-web (WebGPU or WebAssembly).
//
// The model files come from `python -m midigenai.export_onnx` and live on the hub next to each checkpoint, in
// VERSION/onnx/ with a manifest.json. Prompts are still built in Python: a `Generator(backend="remote")` hands
// out one spec per generation (`Generator.remote_spec`: prompt ids and the sampling settings, as JSON), this
// module samples the answer on the device, and the ids go back to Python, which decodes them as usual.
//
//   import * as ort from "./ort.webgpu.bundle.min.mjs";        // onnxruntime-web 1.30; the page serves its .wasm
//   import { detect, loadModel } from "./midigen-web.mjs";
//   const device = await detect();                              // what this browser can run, and which to pick
//   const model = await loadModel({ ort, engine: device.recommended, onProgress: p => show(p.loaded, p.total) });
//   const ids = await model.decode(spec);                       // sampled ids, as Generator's torch backend yields them
//
// Two engines, one file each (sizes in the manifest):
// - "gpu": WebGPU with 4-bit weights (about 73 MB). The KV cache stays on the GPU between steps.
// - "cpu": WebAssembly with 8-bit weights (about 115 MB); uses threads when the page is cross-origin isolated.
// A download is checked against the manifest's sha256 and kept in the Cache API, so it happens once per browser
// (`cached` says which are there, `forget` drops them). onnxruntime-web's own .wasm (about 27 MB) is the page's.
//
// Sampling follows MusicTransformer.generate: logits / temperature, banned ids and (for the first
// min_new_tokens steps) EOS masked, top-k, a draw. A seed repeats a result on this module's RNG, not torch's.

export const HUB = "https://huggingface.co/nicholasbien/midigenai/resolve/main";
export const DEFAULT_VERSION = "v5";
export const ALIASES = { "v5-rl": "v5" };          // earlier names (midigenai.hub.ALIASES)
export const CACHE_NAME = "midigenai-onnx";
export const ENGINES = {
  gpu: { label: "GPU (WebGPU, 4-bit)", quant: "q4" },
  cpu: { label: "CPU (WebAssembly, 8-bit)", quant: "int8" },
};

export function modelBase(version = DEFAULT_VERSION, hub = HUB) {
  return `${hub.replace(/\/+$/, "")}/${ALIASES[version] || version}/onnx/`;
}

export async function fetchManifest(base = modelBase(), fetchImpl = (...a) => globalThis.fetch(...a)) {
  const res = await fetchImpl(new URL("manifest.json", base).href);
  if (!res.ok) throw new Error(`no midigenai model at ${base} (manifest.json: HTTP ${res.status})`);
  return res.json();
}

export function fileFor(manifest, engine) {
  const f = manifest && manifest.files && manifest.files[engine];
  if (!f) throw new Error(`the model has no ${engine} file (engines: ${Object.keys(manifest?.files || {}).join(", ")})`);
  return f;
}

/** Download size in bytes of each engine's model file: {gpu, cpu}. */
export function sizes(manifest) {
  return Object.fromEntries(Object.entries(manifest.files).map(([k, f]) => [k, f.bytes]));
}

// ---------------------------------------------------------------- what this device can run

/** {webgpu, adapter, threads, isolated, appleMobile, recommended}. WebGPU is the faster engine where it holds up;
 *  on iPhone and iPad the GPU run was seen to reload the page partway (2026-10, iOS 26) while the CPU engine
 *  finished in 6 s, so those default to "cpu". */
export async function detect(nav = globalThis.navigator) {
  let webgpu = false, adapter = "not available";
  try {
    if (nav && nav.gpu) {
      const a = await nav.gpu.requestAdapter();
      if (a) {
        const i = a.info || {};
        webgpu = true;
        adapter = [i.vendor, i.architecture || i.description].filter(Boolean).join(" ") || "available";
      } else adapter = "no adapter";
    }
  } catch (e) {
    adapter = "blocked";
  }
  const ua = (nav && nav.userAgent) || "";
  const appleMobile = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints || 0) > 1);
  const isolated = !!globalThis.crossOriginIsolated;
  const threads = defaultThreads(nav);
  return { webgpu, adapter, threads, isolated, appleMobile, recommended: webgpu && !appleMobile ? "gpu" : "cpu" };
}

function defaultThreads(nav = globalThis.navigator) {
  return globalThis.crossOriginIsolated ? Math.max(1, Math.min(4, (nav && nav.hardwareConcurrency) || 4)) : 1;
}

// ---------------------------------------------------------------- downloads, checked and cached

function cacheKey(base, f) {
  return `${new URL(f.file, base).href}?sha256=${f.sha256}`;
}

async function openCache() {
  try {
    return globalThis.caches ? await globalThis.caches.open(CACHE_NAME) : null;
  } catch (e) {
    return null;                      // private windows and some webviews refuse storage: download each time
  }
}

/** Which engines' files are already in this browser: {gpu: bool, cpu: bool}. */
export async function cached(base = modelBase(), manifest = null) {
  manifest = manifest || await fetchManifest(base);
  const cache = await openCache();
  const out = {};
  for (const [k, f] of Object.entries(manifest.files)) {
    try { out[k] = !!(cache && await cache.match(cacheKey(base, f))); } catch (e) { out[k] = false; }
  }
  return out;
}

/** Drop the cached model files (all of them, or those under `base`). */
export async function forget(base = null) {
  if (!globalThis.caches) return;
  if (!base) { await globalThis.caches.delete(CACHE_NAME); return; }
  const cache = await openCache();
  if (!cache) return;
  for (const req of await cache.keys()) if (req.url.startsWith(base)) await cache.delete(req);
}

async function sha256Hex(bytes) {
  const subtle = globalThis.crypto && globalThis.crypto.subtle;
  if (!subtle) return null;           // not a secure context: skip the check rather than fail
  const h = new Uint8Array(await subtle.digest("SHA-256", bytes));
  return Array.from(h, b => b.toString(16).padStart(2, "0")).join("");
}

/** The engine's model bytes, from the cache or the network. onProgress({loaded, total, cached}). */
export async function download(base, manifest, engine, { onProgress, signal, cache: useCache = true, fetchImpl = (...a) => globalThis.fetch(...a) } = {}) {
  const f = fileFor(manifest, engine);
  const key = cacheKey(base, f);
  const cache = useCache ? await openCache() : null;
  let res = null, fromCache = false;
  try { res = cache ? await cache.match(key) : null; } catch (e) { res = null; }
  if (res) fromCache = true;
  else {
    res = await fetchImpl(new URL(f.file, base).href, { signal });
    if (!res.ok) throw new Error(`model download failed: ${f.file} (HTTP ${res.status})`);
  }
  const total = f.bytes;
  const out = new Uint8Array(total);
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > total) throw new Error(`model download is larger than the manifest says (${f.file})`);
    out.set(value, loaded);
    loaded += value.length;
    if (onProgress) onProgress({ loaded, total, cached: fromCache });
  }
  if (loaded !== total) throw new Error(`model download stopped at ${loaded} of ${total} bytes (${f.file})`);
  const sum = await sha256Hex(out);
  if (sum && sum !== f.sha256) {
    if (fromCache) await cache.delete(key);
    throw new Error(`model file ${f.file} does not match its checksum; try again`);
  }
  if (cache && !fromCache) {
    try { await cache.put(key, new Response(out, { headers: { "content-type": "application/octet-stream" } })); }
    catch (e) { /* storage full or refused: it still runs, and downloads again next time */ }
  }
  return out;
}

// ---------------------------------------------------------------- sampling, as MusicTransformer.generate

/** A [0, 1) generator: seeded (mulberry32 over the seed's low and high words) or Math.random. */
export function rng(seed) {
  if (seed === null || seed === undefined) return Math.random;
  let a = (Number(seed) >>> 0) ^ (Math.floor(Number(seed) / 4294967296) >>> 0) ^ 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** sample(logits, step, rand) -> id, for one spec. */
export function sampler(spec) {
  let ban = null, eos = spec.eos_id ?? null;
  const temp = Math.max(spec.temperature ?? 1.0, 1e-6), topK = spec.top_k ?? null, minNew = spec.min_new_tokens || 0;
  let vals = null, sorted = null;
  return (logits, step, rand) => {
    const V = logits.length;
    if (!vals || vals.length !== V) {
      vals = new Float32Array(V); sorted = new Float32Array(V);
      ban = new Uint8Array(V);
      for (const t of spec.ban_ids || []) if (t >= 0 && t < V) ban[t] = 1;
    }
    for (let j = 0; j < V; j++) vals[j] = ban[j] ? -Infinity : logits[j] / temp;
    if (eos !== null && step < minNew && eos < V) vals[eos] = -Infinity;
    let floor = -Infinity;
    if (topK !== null && topK < V) {
      sorted.set(vals);
      sorted.sort();                                   // ascending; the k-th largest sits at V - k
      floor = sorted[V - Math.max(1, topK)];
    }
    let m = -Infinity;
    for (let j = 0; j < V; j++) if (vals[j] >= floor && vals[j] > m) m = vals[j];
    if (m === -Infinity) throw new Error("every token is banned: nothing to sample");
    let sum = 0;
    for (let j = 0; j < V; j++) {
      const w = vals[j] >= floor ? Math.exp(vals[j] - m) : 0;
      vals[j] = w; sum += w;
    }
    let r = rand() * sum, last = -1;
    for (let j = 0; j < V; j++) {
      if (vals[j] <= 0) continue;
      last = j;
      r -= vals[j];
      if (r < 0) return j;
    }
    return last;
  };
}

/** stop(token) -> true once the decoder may stop after this token: one of stop_ids (EOS included), or one Bar
 *  more than stop_after_bars, counting from the first token that is not a Bar or TimeSig when
 *  trim_leading_bars (Generator.postprocess has stopped reading by then; see Generator.remote_spec). */
export function stopper(spec) {
  const stops = new Set(spec.stop_ids || []);
  if (spec.eos_id !== null && spec.eos_id !== undefined) stops.add(spec.eos_id);
  const ts = new Set(spec.timesig_ids || []);
  const limit = spec.stop_after_bars || 0;
  let started = !spec.trim_leading_bars, bars = 0;
  return tok => {
    if (stops.has(tok)) return true;
    if (!limit) return false;
    if (!started) {
      if (tok === spec.bar_id || ts.has(tok)) return false;
      started = true;
    }
    return tok === spec.bar_id && ++bars > limit;
  };
}

/** The decode loop over any `step(ids) -> logits of the last position` (an engine's, or a stand-in in tests). */
export async function decodeLoop(step, spec, { onToken, signal, rand } = {}) {
  const prompt = spec.prompt;
  if (!prompt || !prompt.length) throw new Error("empty prompt");
  const sample = sampler(spec), stop = stopper(spec), draw = rand || rng(spec.seed);
  const maxSeq = spec.max_seq_len || Infinity;
  const out = [];
  let feed = prompt;
  for (let i = 0; i < spec.max_new_tokens; i++) {
    if ((i === 0 ? prompt.length : prompt.length + i) > maxSeq) break;      // as generate: no position past the context
    if (signal && signal.aborted) throw signal.reason || new Error("aborted");
    const logits = await step(feed);
    if (i === 0 && spec.vocab_size && logits.length !== spec.vocab_size)
      throw new Error(`the model has ${logits.length} tokens and the prompt's tokenizer ${spec.vocab_size}: another version`);
    const tok = sample(logits, i, draw);
    out.push(tok);
    if (onToken) onToken(tok, i);
    if (stop(tok)) break;
    feed = [tok];
  }
  return out;
}

// ---------------------------------------------------------------- the two engines

function ioNames(n, outPrefix) {
  const past = [], out = [];
  for (let i = 0; i < n; i++) for (const c of "kv") { past.push(`past_${c}${i}`); out.push(`${outPrefix}_${c}${i}`); }
  return [past, out];
}

function idsTensor(ort, toks) {
  return new ort.Tensor("int64", BigInt64Array.from(toks, t => BigInt(t)), [1, toks.length]);
}

// GPU: (1, H, P, D) caches in, the whole (1, H, P+S, D) presents out, left on the GPU and fed back next step.
function gpuStep(ort, session, m) {
  const H = m.n_heads, D = m.head_dim;
  const [pastNames, presNames] = ioNames(m.n_layers, "present");
  let past = Object.fromEntries(pastNames.map(n => [n, new ort.Tensor("float32", new Float32Array(0), [1, H, 0, D])]));
  const free = ts => { for (const t of ts) if (t && t.location === "gpu-buffer") t.dispose(); };
  const step = async toks => {
    const out = await session.run({ ids: idsTensor(ort, toks), ...past });
    const logits = await out.logits.getData();
    const next = {};
    pastNames.forEach((n, j) => { next[n] = out[presNames[j]]; });
    free(Object.values(past));
    past = next;
    return logits;
  };
  step.done = () => { free(Object.values(past)); past = {}; };
  return step;
}

// CPU: time-major (1, P, H, D) caches in, only the new rows out; they are appended into buffers kept here,
// grown as needed, so each step copies the cache into wasm once and nothing back.
function cpuStep(ort, session, m, expect) {
  const ROW = m.n_heads * m.head_dim, H = m.n_heads, D = m.head_dim;
  const [pastNames, newNames] = ioNames(m.n_layers, "new");
  let cap = Math.max(64, expect), P = 0;
  let bufs = pastNames.map(() => new Float32Array(cap * ROW));
  const step = async toks => {
    if (P + toks.length > cap) {
      cap = Math.max(P + toks.length, Math.ceil(cap * 1.5));
      bufs = bufs.map(b => { const nb = new Float32Array(cap * ROW); nb.set(b.subarray(0, P * ROW)); return nb; });
    }
    const feeds = { ids: idsTensor(ort, toks) };
    pastNames.forEach((n, j) => { feeds[n] = new ort.Tensor("float32", bufs[j].subarray(0, P * ROW), [1, P, H, D]); });
    const out = await session.run(feeds);
    newNames.forEach((n, j) => { bufs[j].set(out[n].data, P * ROW); });
    P += toks.length;
    return out.logits.data;
  };
  step.done = () => { bufs = []; };
  return step;
}

export class MidigenModel {
  constructor(ort, session, engine, manifest) {
    this.ort = ort;
    this.session = session;
    this.engine = engine;
    this.manifest = manifest;
    this._queue = Promise.resolve();
  }

  /** Sample one generation for `spec` (Generator.remote_spec). Calls run one at a time on a model.
   *  onToken(id, step) sees each id as it lands; `signal` (an AbortSignal) stops between steps. */
  decode(spec, opts = {}) {
    const run = async () => {
      const m = this.manifest;
      const expect = spec.prompt.length + Math.min(spec.max_new_tokens, 512);
      const step = this.engine === "gpu" ? gpuStep(this.ort, this.session, m) : cpuStep(this.ort, this.session, m, expect);
      try {
        return await decodeLoop(step, spec, opts);
      } finally {
        step.done();
      }
    };
    const p = this._queue.then(run, run);
    this._queue = p.catch(() => {});
    return p;
  }

  async release() {
    await this._queue;
    try { await this.session.release(); } catch (e) { /* already gone */ }
  }
}

/** Download (or read from the cache) and start one engine's model. onProgress({phase: "download"|"load",
 *  loaded, total, cached}). `threads` sets wasm threads for the cpu engine (default: 1, or up to 4 when the
 *  page is cross-origin isolated). */
export async function loadModel({ ort, engine = "cpu", base = modelBase(), manifest = null, threads, onProgress, signal, cache = true, fetchImpl } = {}) {
  if (!ort) throw new Error("loadModel needs onnxruntime-web: loadModel({ort, ...})");
  if (!ENGINES[engine]) throw new Error(`engine must be ${Object.keys(ENGINES).join(" or ")}`);
  if (engine === "gpu" && !(globalThis.navigator && globalThis.navigator.gpu)) throw new Error("WebGPU is not available in this browser");
  manifest = manifest || await fetchManifest(base, fetchImpl);
  if (ort.env && ort.env.wasm) ort.env.wasm.numThreads = threads ?? defaultThreads();
  let bytes = await download(base, manifest, engine, {
    signal, cache, fetchImpl,
    onProgress: onProgress && (p => onProgress({ phase: "download", ...p })),
  });
  if (onProgress) onProgress({ phase: "load", loaded: bytes.length, total: bytes.length, cached: true });
  const presents = ioNames(manifest.n_layers, "present")[1];
  const opts = engine === "gpu"
    ? {
      // lazyRelease: buffers freed during a step go back after it rather than into a pool, which keeps memory
      // flat while the cache grows a row per token
      executionProviders: [{ name: "webgpu", storageBufferCacheMode: "lazyRelease" }],
      preferredOutputLocation: Object.fromEntries([["logits", "cpu"], ...presents.map(n => [n, "gpu-buffer"])]),
    }
    : { executionProviders: ["wasm"] };
  const session = await ort.InferenceSession.create(bytes, opts);
  bytes = null;                       // the session keeps its own copy
  return new MidigenModel(ort, session, engine, manifest);
}
