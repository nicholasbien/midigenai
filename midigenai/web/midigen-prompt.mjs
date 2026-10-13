// midigen-prompt: the prompt side (tokenizer.mjs, attributes.mjs, generator.mjs, tasks.mjs) on scores with
// symusic's Python field names, for a page that builds its scores itself rather than parsing MIDI files.
//
// It is a thin layer: the tokenizer, the header, the prompt layouts, `postprocess` and span infill's plan and
// splice are those modules' own. What it adds is the score shape at the edges, a version on each spec (so a
// server knows which checkpoint to sample), the prompt-layout helpers and the header families `remote` callers
// set by hand (source, genres, quality). The decoder is any function from a spec to sampled ids: midigen-web's
// model in the page, or a server that samples on a GPU (modal_serve.py MidiGen.decode, which runs
// Generator.decode_spec).
//
//   import { loadGenerator } from "./midigen-prompt.mjs";
//   const g = await loadGenerator({ version: "v5", decode: spec => model.decode(spec) });
//   const body = g.tokenizer.encode(score);           // a Score: see below; the ids Python's tokenizer gives it
//   const ids = await g.continueIds(body, { header, bars: 4, maxNewTokens: 400 });
//   const answer = g.tokenizer.decode(ids);           // a Score at the tokenizer's 16 ticks per quarter
//
// A Score here is symusic's, as plain data:
//   {tpq, tracks: [{name, program, is_drum, notes: [{time, duration, pitch, velocity}], controls: [{time, number, value}]}],
//    time_signatures: [{time, numerator, denominator}], tempos: [{time, qpm}]}
// with times in ticks (midi.mjs names them isDrum, timeSignatures and {time, mspq}). tests/test_web_prompt.py
// checks the ids, windows and splices against MidiTok, symusic and Generator on random scores.

import * as midi from "./midi.mjs";
import * as attrs from "./attributes.mjs";
import { RemiTokenizer } from "./tokenizer.mjs";
import { Generator as PortGenerator } from "./generator.mjs";
import { InfillError, planScore, splice as portSplice, window as portWindow, barEdges as portBarEdges } from "./tasks.mjs";
import { ALIASES, DEFAULT_VERSION, HUB } from "./midigen-web.mjs";

export { ALIASES, DEFAULT_VERSION, HUB, InfillError };
export { fitToContext, accompanimentBudget, MIN_PROMPT_TOKENS, ACCOMP_TOKENS_PER_BAR } from "./generator.mjs";
export { CONTEXT_BARS, MAX_SPAN_BARS, contextWindow } from "./tasks.mjs";
export {
  INSTRUMENT_FAMILIES, DRUMS, MAX_INST_TOKENS, HEADER_PREFIXES, isHeaderToken, familyOf, sortHeader, withInstruments,
  nameSaysDrums, looksLikeDrums,
} from "./attributes.mjs";

// header families attributes.mjs leaves to the caller (attributes.py SOURCES, GENRES)
export const SOURCES = ["lakh", "lamd", "aria", "gigamidi", "maestro", "pop909", "giantmidi", "user", "fma"];
export const GENRES = ["rock", "pop", "jazz", "classical", "electronic", "hiphop", "rnb", "country", "folk", "latin",
  "blues", "metal", "reggae", "soul", "world", "other"];

// ---------------------------------------------------------------- the score shape at the edges

const events = (xs) => (xs || []).map((x) => ({ ...x }));

/** A Score here -> midi.mjs's. */
export function toPort(sc) {
  const out = midi.newScore(sc.tpq);
  out.tracks = sc.tracks.map((t) => ({ ...midi.newTrack(t.name || "", t.program, !!t.is_drum), notes: events(t.notes), controls: events(t.controls) }));
  out.timeSignatures = events(sc.time_signatures);
  out.tempos = (sc.tempos || []).map((x) => ({ time: x.time, mspq: midi.mspqOf(x.qpm) }));
  return out;
}

/** midi.mjs's Score -> one here. */
export function fromPort(sc) {
  return {
    tpq: sc.tpq,
    tracks: sc.tracks.map((t) => ({ name: t.name, program: t.program, is_drum: !!t.isDrum, notes: events(t.notes), controls: events(t.controls) })),
    time_signatures: events(sc.timeSignatures),
    tempos: sc.tempos.map((x) => ({ time: x.time, qpm: midi.qpm(x) })),
  };
}

export const emptyScore = (tpq) => ({ tpq, tracks: [], time_signatures: [], tempos: [] });
export const copyScore = (sc) => fromPort(toPort(sc));
export const scoreEnd = (sc) => midi.scoreEnd(toPort(sc));
export const resample = (sc, tpq, minDur = 0) => fromPort(midi.resample(toPort(sc), tpq, minDur));
export const clip = (sc, start, end, clipEnd = false) => fromPort(midi.clip(toPort(sc), start, end, clipEnd));
export const shiftTime = (sc, dt) => fromPort(midi.shiftTime(toPort(sc), dt));
/** data/v4_docs.bar_edges: every bar line from 0 through the end of the last bar. */
export const barEdges = (sc) => portBarEdges(toPort(sc));
/** data/v4_docs._window: a bar-aligned clip re-based to tick 0, its last half-position left out, the meter restated. */
export const windowOf = (sc, startTick, endTick) => fromPort(portWindow(toPort(sc), startTick, endTick));

// ---------------------------------------------------------------- attributes

/** attributes.tempo_tokens with the caller's tempo (the clock the answer plays at): [] without one. */
export function tempoTokens(tempo) {
  return tempo > 0 ? attrs.tempoTokens(null, tempo) : [];
}

/** attributes.header_for_score with an explicit tempo (the score's own tempo events are not read here). */
export function headerForScore(score, { tempo = null, source = null, genres = [], quality = null } = {}) {
  const sc = toPort(score);
  const names = [...attrs.instrumentTokens(sc), ...attrs.contentTokens(sc), ...tempoTokens(tempo)];
  if (source && SOURCES.includes(source)) names.push(`Source_${source}`);
  for (const g of genres || []) if (GENRES.includes(g)) names.push(`Genre_${g}`);
  if (quality !== null && quality !== undefined && Math.trunc(quality) >= 0 && Math.trunc(quality) < 4) names.push(`Quality_${Math.trunc(quality)}`);
  return names;
}

/** tokenizer.normalize_drums: promote drum-like tracks to is_drum, in place; how many it promoted. */
export function normalizeDrums(score, filenameHint = "") {
  const tracks = score.tracks.map((t) => ({ name: t.name, notes: t.notes, isDrum: !!t.is_drum }));
  const changed = attrs.normalizeDrums({ tracks }, filenameHint);
  tracks.forEach((t, i) => { if (t.isDrum) score.tracks[i].is_drum = true; });
  return changed;
}

// ---------------------------------------------------------------- the tokenizer and the Generator

/** tokenizer.mjs's REMI tokenizer, taking and giving Scores here. */
export class Tokenizer extends RemiTokenizer {
  /** tokenizer.special_id: the id of BOS / MASK / Inst_Piano ..., or null. */
  specialId(name) {
    for (const t of [`${name}_None`, name]) if (this.vocab.has(t)) return this.vocab.get(t);
    return null;
  }

  encodeTokens(score) { return super.encodeTokens(toPort(score)); }
  encode(score) { return this.encodeTokens(score).map((t) => this.id(t)); }
  /** A Score (tpq = the tokenizer's time division) from ids, as tokenizer.decode(ids) makes it. */
  decode(ids) { return fromPort(super.decode(ids)); }
}

/** generator.mjs's Generator, with the version on each spec and the prompt layouts by name. */
export class Generator extends PortGenerator {
  /** `decode(spec) -> ids` (or a promise of them) samples one generation: midigen-web's model.decode, or a server. */
  constructor(tokenizer, decode, { maxSeqLen = 8192, version = null } = {}) {
    super(tokenizer, decode, { maxSeqLen });
    this.version = version;
    this.sp = { bos: this.bosId, eos: this.eosId, sep: this.sepId, mask: this.maskId, bar: this.barId,
      taskAccomp: this.taskAccomp, taskInfill: this.taskInfill, headerIds: this.headerIds, headerFamily: this.headerFamily };
  }

  remoteSpec(promptIds, opts) {
    const spec = super.remoteSpec(promptIds, opts);
    return this.version ? { ...spec, version: this.version } : spec;
  }

  /** Generator.make_header without a file: the given families only. */
  makeHeader({ instruments = null, density = null, poly = null, pitchRange = null, source = null, genres = [], tempo = null } = {}) {
    let names = [];
    if (source) names.push(`Source_${source}`);
    names.push(...(genres || []).map((g) => `Genre_${g}`), ...tempoTokens(tempo));
    const override = (prefix, add) => { names = [...names.filter((n) => !n.startsWith(prefix)), ...add]; };
    if (instruments !== null) override("Inst_", instruments.map((i) => `Inst_${i}`));
    if (density !== null) override("Density_", [`Density_${density}`]);
    if (poly !== null) override("Poly_", [`Poly_${poly}`]);
    if (pitchRange !== null) override("Range_", [`Range_${pitchRange}`]);
    return this.headerIdsFor(attrs.sortHeader(names));
  }

  // sequence_format.py's layouts, as continueIds / accompany / infill build them
  continuationPrompt(header, body) { return [this.bosId, ...header, ...body]; }
  accompanimentPrompt(header, cond) { return [this.bosId, ...(this.taskAccomp !== null ? [this.taskAccomp] : []), ...header, ...cond, this.sepId]; }
  infillPrompt(header, prefix, suffix) {
    return [this.bosId, ...(this.taskInfill !== null ? [this.taskInfill] : []), ...header, ...prefix, this.maskId, ...suffix, this.sepId];
  }
}

// ---------------------------------------------------------------- infill_span.py

/** infill_span.plan_score on a Score here (drums normalized by the caller): tasks.mjs's plan, with its windows
 *  as Scores here. */
export function infillPlan(score, start, bars) {
  const p = planScore(toPort(score), start, bars);
  return {
    port: p, score, edges: p.edges, start, bars, ctxStart: p.ctxStart, ctxEnd: p.ctxEnd, spanTicks: p.spanTicks,
    prefixBars: p.prefixBars, suffixBars: p.suffixBars,
    window: () => fromPort(p.window()), prefix: () => fromPort(p.prefix()), suffix: () => fromPort(p.suffix()),
  };
}

/** infill_span.prompt_segments: (prefix ids, suffix ids), each padded to its bar count. */
export function promptSegments(gen, p) {
  return [gen.padToBars(gen.tokenizer.encode(p.prefix()), p.prefixBars), gen.padToBars(gen.tokenizer.encode(p.suffix()), p.suffixBars)];
}

/** infill_span.splice: the score with bars [start, start+bars) replaced by `answer` (a segment from the span's bar 0). */
export function splice(p, answer) {
  return fromPort(portSplice(p.port, toPort(answer)));
}

/** infill_span.run for one take on a score as parsed from the upload: {plan, score: spliced, notes: generated count,
 *  ids}. Drum-like tracks are promoted first, as `plan` does (on a copy). */
export async function infillRun(gen, score, start, bars, { tempoBpm = 120.0, ...opts } = {}) {
  const sc = copyScore(score);
  normalizeDrums(sc, "upload.mid");
  const p = infillPlan(sc, start, bars);
  const [prefix, suffix] = promptSegments(gen, p);
  const header = gen.headerIdsFor(headerForScore(p.window(), { tempo: tempoBpm }));
  const ids = await gen.infill(prefix, suffix, bars, { header, ...opts });
  const answer = gen.tokenizer.decode(ids);
  return { plan: p, score: splice(p, answer), notes: answer.tracks.reduce((s, t) => s + t.notes.length, 0), ids };
}

// ---------------------------------------------------------------- loading

export function tokenizerUrl(version = DEFAULT_VERSION, hub = HUB) {
  return `${hub.replace(/\/+$/, "")}/${ALIASES[version] || version}/tokenizer.json`;
}

const _tokenizers = new Map();

/** A version's Tokenizer, from its tokenizer.json on the hub (once per page). */
export function loadTokenizer({ version = DEFAULT_VERSION, hub = HUB, fetchImpl = (...a) => globalThis.fetch(...a) } = {}) {
  const url = tokenizerUrl(version, hub);
  if (!_tokenizers.has(url)) {
    const p = (async () => {
      const res = await fetchImpl(url);
      if (!res.ok) throw new Error(`no midigenai tokenizer at ${url} (HTTP ${res.status})`);
      return new Tokenizer(await res.json());
    })();
    _tokenizers.set(url, p);
    p.catch(() => _tokenizers.delete(url));
  }
  return _tokenizers.get(url);
}

/** A Generator for `version` whose model is `decode(spec) -> ids`; each spec carries the version. */
export async function loadGenerator({ version = DEFAULT_VERSION, hub = HUB, decode, maxSeqLen = 8192, fetchImpl } = {}) {
  const v = ALIASES[version] || version;
  const tok = await loadTokenizer({ version: v, hub, ...(fetchImpl ? { fetchImpl } : {}) });
  return new Generator(tok, decode, { maxSeqLen, version: v });
}
