// generator: midigenai's Generator (generate.py) for a model that runs in the page. Prompt layouts, the
// context fit, bar padding and `postprocess` are ported line for line; the sampling itself is a `decode(spec)`
// function, normally MidigenModel.decode from midigen-web.mjs, which gets the same JSON spec the Python
// `remote` backend hands out (Generator.remote_spec).
//
//   const gen = new Generator(tokenizer, (spec, opts) => model.decode(spec, opts));
//   const ids = await gen.accompany(condIds, 8, { header });
//
// v4 tokenizers only (REMI + header + SEP/MASK), which is every checkpoint the browser runs.

import { newScore, resample } from "./midi.mjs";
import { HEADER_PREFIXES, isHeaderToken } from "./attributes.mjs";

export const MIN_PROMPT_TOKENS = 256;
export const OVERLAY_TPQ = 480;
export const ACCOMP_TOKENS_PER_BAR = 160;

/** [prompt, maxNewTokens] with prompt + maxNewTokens <= maxSeqLen; a long prompt is cut at the end. */
export function fitToContext(promptIds, maxNewTokens, maxSeqLen, minPromptTokens = MIN_PROMPT_TOKENS) {
  let ids = [...promptIds];
  if (maxSeqLen <= 0) return [ids, maxNewTokens];
  const minPrompt = Math.min(minPromptTokens, ids.length, Math.floor(maxSeqLen / 2));
  maxNewTokens = Math.max(0, Math.min(maxNewTokens, maxSeqLen - minPrompt));
  const keep = maxSeqLen - maxNewTokens;
  if (ids.length > keep) ids = ids.slice(0, keep);
  return [ids, maxNewTokens];
}

/** A generated accompaniment stacked on the part it answers, both resampled to one tick rate first. */
export function overlay(condition, generated, tpq = OVERLAY_TPQ) {
  const out = newScore(tpq);
  const cond = resample(condition, tpq), gen = resample(generated, tpq);
  out.timeSignatures = (cond.timeSignatures.length ? cond.timeSignatures : gen.timeSignatures).map(x => ({ ...x }));
  out.tempos = (cond.tempos.length ? cond.tempos : gen.tempos).map(x => ({ ...x }));
  for (const src of [cond, gen]) for (const t of src.tracks) if (t.notes.length) out.tracks.push(t);
  return out;
}

/** max_new_tokens for an accompaniment of `bars` bars over a condition of `condTokens` tokens. */
export function accompanimentBudget(condTokens, bars) {
  const perBar = Math.max(ACCOMP_TOKENS_PER_BAR, (2 * condTokens) / Math.max(1, bars));
  return Math.trunc(perBar * bars) + 64;
}

const sortedIds = xs => [...xs].map(Number).sort((a, b) => a - b);

export class Generator {
  /** `decode(spec, opts) -> Promise<ids>` samples one generation; `maxSeqLen` is the serving context. */
  constructor(tokenizer, decode, { maxSeqLen = 8192 } = {}) {
    if (typeof decode !== "function") throw new Error("Generator needs decode(spec) (MidigenModel.decode)");
    const v = tokenizer.vocab;
    if (!v.has("MASK_None") || !v.has("Bar_None")) throw new Error("the browser runs v4 (REMI + header) tokenizers only");
    this.tokenizer = tokenizer;
    this.decode = decode;
    this.maxSeqLen = maxSeqLen;
    const sid = name => v.get(`${name}_None`) ?? v.get(name) ?? null;
    this.specialId = sid;
    this.bosId = sid("BOS");
    this.eosId = sid("EOS");
    this.sepId = sid("SEP");
    this.maskId = sid("MASK");
    this.barId = v.get("Bar_None");
    this.taskAccomp = sid("Task_accomp");
    this.taskInfill = sid("Task_infill");
    this.headerFamily = new Map();
    for (const [name, id] of v) if (isHeaderToken(name)) this.headerFamily.set(id, HEADER_PREFIXES.find(p => name.startsWith(p)));
    this.headerIds = new Set(this.headerFamily.keys());
    this.timesigIds = new Set([...v].filter(([k]) => k.startsWith("TimeSig_")).map(([, i]) => i));
    // SEP / MASK are never valid output, and BOS is what follows a target in training: all end generation
    this.stopIds = new Set([this.eosId, this.sepId, this.maskId, this.bosId].filter(t => t !== null));
  }

  /** Header names -> ids. A whole family this tokenizer predates is skipped; any other unknown is an error. */
  headerIdsFor(names) {
    const known = new Set(this.headerFamily.values());
    const out = [];
    for (const n of names) {
      const id = this.specialId(n);
      if (id === null) {
        const fam = HEADER_PREFIXES.find(p => n.startsWith(p));
        if (fam !== undefined && !known.has(fam)) continue;
        throw new Error(`not a header token: ${n}`);
      }
      out.push(id);
    }
    return out;
  }

  needsBos(ids) { return this.bosId !== null && (!ids.length || ids[0] !== this.bosId); }

  /** fitToContext counting the BOS that generation prepends. */
  fitToContext(promptIds, maxNewTokens) {
    const added = this.needsBos(promptIds);
    const ids = added ? [this.bosId, ...promptIds] : [...promptIds];
    const [fit, n] = fitToContext(ids, maxNewTokens, this.maxSeqLen);
    return [added ? fit.slice(1) : fit, n];
  }

  defaultBanIds() { return sortedIds([this.sepId, this.maskId, this.bosId].filter(t => t !== null)); }

  countBars(ids) { let n = 0; for (const t of ids) if (t === this.barId) n++; return n; }

  /** Append empty `Bar TimeSig` pairs so `ids` spans exactly `nBars`. */
  padToBars(ids, nBars) {
    const have = this.countBars(ids);
    if (have > nBars) throw new Error(`prompt spans ${have} bars > ${nBars}`);
    const ts = ids.find(t => this.timesigIds.has(t)) ?? this.tokenizer.vocab.get("TimeSig_4/4");
    const out = [...ids];
    for (let i = have; i < nBars; i++) out.push(this.barId, ts);
    return out;
  }

  closeBar(ids) { return this.endsOnBarLine(ids) ? [...ids] : this.padToBars(ids, this.countBars(ids) + 1); }

  endsOnBarLine(ids) {
    if (!ids.length) return false;
    let tail = ids.filter(t => !this.headerIds.has(t) && t !== this.bosId);
    if (tail.length && this.timesigIds.has(tail[tail.length - 1])) tail = tail.slice(0, -1);
    return tail.length > 0 && tail[tail.length - 1] === this.barId;
  }

  /** Raw sampled ids -> the continuation a caller gets: stop at EOS / SEP / MASK / BOS, drop leading empty
   *  bars, honor stopAfterBars (Generator.postprocess). */
  postprocess(promptIds, raw, { stopAfterBars = null, trimLeadingBars = true } = {}) {
    const out = [];
    const onBar = this.endsOnBarLine(promptIds);
    let bars = stopAfterBars && onBar ? 1 : 0;
    let started = !trimLeadingBars;
    let held = [];
    const heldOpen = !onBar;
    for (const t of raw) {
      if (t === this.eosId) { out.push(t); return out; }
      if (this.stopIds.has(t)) return out;
      if (!started) {
        if (t === this.barId) {
          if (heldOpen && !held.length) held = [t];
          continue;
        }
        if (this.timesigIds.has(t)) {
          if (held.length === 1) held.push(t);
          continue;
        }
        started = true;
        for (const h of held) {
          if (stopAfterBars && h === this.barId) bars++;
          out.push(h);
        }
        held = [];
      }
      if (stopAfterBars && t === this.barId) {
        bars++;
        if (bars > stopAfterBars) return out;
      }
      out.push(t);
    }
    return out;
  }

  /** The JSON spec one generation hands to `decode` (Generator.remote_spec). */
  remoteSpec(promptIds, { maxNewTokens, temperature, topK, minNewTokens = 0, seed = null, banIds = null, stopAfterBars = null, trimLeadingBars = true }) {
    return {
      prompt: promptIds.map(Number),
      vocab_size: this.tokenizer.vocabSize,
      max_new_tokens: Math.trunc(maxNewTokens),
      max_seq_len: this.maxSeqLen,
      temperature: Number(temperature),
      top_k: topK === null || topK === undefined ? null : Math.trunc(topK),
      min_new_tokens: Math.trunc(minNewTokens),
      eos_id: this.eosId,
      ban_ids: sortedIds(banIds || []),
      stop_ids: sortedIds(this.stopIds),
      bar_id: this.barId,
      timesig_ids: sortedIds(this.timesigIds),
      stop_after_bars: stopAfterBars ? Math.trunc(stopAfterBars) : null,
      trim_leading_bars: !!trimLeadingBars,
      seed: seed === null || seed === undefined ? null : Math.trunc(seed),
    };
  }

  /** Generator.generate_ids: fit, prepend BOS, sample, postprocess. Resolves to the ids. `onToken`/`signal`
   *  go to the decoder. */
  async generateIds(promptIds, { maxNewTokens = 512, temperature = 1.0, topK = 50, minNewTokens = 0, seed = null, stopAfterBars = null, banIds = null, trimLeadingBars = true, onToken, signal } = {}) {
    if (banIds === null) banIds = this.defaultBanIds();
    let [prompt, maxNew] = this.fitToContext(promptIds, maxNewTokens);
    if (this.needsBos(prompt)) prompt = [this.bosId, ...prompt];
    const spec = this.remoteSpec(prompt, { maxNewTokens: maxNew, temperature, topK, minNewTokens, seed, banIds, stopAfterBars, trimLeadingBars });
    const raw = await this.decode(spec, { onToken, signal });
    return this.postprocess(prompt, raw.map(Number), { stopAfterBars, trimLeadingBars });
  }

  /** Continuation with a header; `bars` closes the prompt's bar and stops after that many bars. */
  async continueIds(bodyIds, { header = [], bars = null, ...opts } = {}) {
    let prompt = [...bodyIds];
    if (bars) prompt = this.closeBar(prompt);
    prompt = [this.bosId, ...header, ...prompt];
    return this.generateIds(prompt, { stopAfterBars: bars, ...opts });
  }

  /** The other parts for `condIds` over exactly `bars` bars, as a segment starting at bar 0 of the window. */
  async accompany(condIds, bars, { header = [], ...opts } = {}) {
    const task = this.taskAccomp !== null ? [this.taskAccomp] : [];
    const prompt = [this.bosId, ...task, ...header, ...this.padToBars(condIds, bars), this.sepId];
    return this.generateIds(prompt, {
      maxNewTokens: accompanimentBudget(condIds.length, bars),
      trimLeadingBars: false,
      banIds: [this.sepId, this.maskId],
      ...opts,
      stopAfterBars: bars,
    });
  }

  /** The `bars` bars between `prefixIds` and `suffixIds` (each bar-aligned), as a segment from bar 0 of the gap. */
  async infill(prefixIds, suffixIds, bars, { header = [], ...opts } = {}) {
    if (this.maskId === null) throw new Error("infill needs a MASK token (v4 tokenizer)");
    const task = this.taskInfill !== null ? [this.taskInfill] : [];
    const prompt = [this.bosId, ...task, ...header, ...prefixIds, this.maskId, ...suffixIds, this.sepId];
    const ctxBars = this.countBars(prefixIds) + this.countBars(suffixIds);
    const perBar = (prefixIds.length + suffixIds.length) / Math.max(1, ctxBars);
    return this.generateIds(prompt, {
      maxNewTokens: Math.trunc(Math.max(64, 2 * perBar) * bars) + 64,
      banIds: sortedIds([this.sepId, this.maskId, this.bosId, this.eosId].filter(t => t !== null)),
      trimLeadingBars: false,
      ...opts,
      stopAfterBars: bars,
    });
  }

  /** (prefix, suffix) around bars [atBar, atBar + nBars) of a segment: the infill inputs for "redo these". */
  splitBars(ids, atBar, nBars) {
    const body = ids.filter(t => !this.headerIds.has(t) && t !== this.bosId);
    const edges = [];
    body.forEach((t, i) => { if (t === this.barId) edges.push(i); });
    if (atBar + nBars > edges.length) throw new Error(`segment has ${edges.length} bars`);
    const end = atBar + nBars < edges.length ? edges[atBar + nBars] : body.length;
    return [body.slice(0, edges[atBar]), body.slice(end)];
  }
}
