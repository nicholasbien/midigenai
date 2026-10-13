// tasks: what the server does with an upload, in the page. Continuation, accompaniment and infill as
// modal_serve.MidiGen runs them (with accompany_tracks.py, infill_span.py and the data/v4_docs.py window
// helpers), on midi.mjs, tokenizer.mjs and a Generator whose decoder is the model in this page.
//
//   import * as ort from "./ort.webgpu.bundle.min.mjs";
//   import { loadMidigen } from "./tasks.mjs";
//   const mg = await loadMidigen({ ort, engine: "gpu" });
//   const res = await mg.accompany(midiBytes, { bars: 8, instrument: "bass" });
//   play(res.midis[0]);                                   // Uint8Array, a standard MIDI file
//
// Results carry the same fields as the server's (camelCase, as the site's JSON names them), with `midis` as
// bytes. Inputs are MIDI file bytes (Uint8Array or ArrayBuffer).

import { clip, copyTrack, downbeats, mspqOf, newScore, newTrack, noteCount, parseMidi, qpm, resample, scoreEnd, shiftTime, writeMidi } from "./midi.mjs";
import { DRUMS, familyOf, headerForScore, isAutoInstrument, normalizeDrums, resolveFamily, tempoTokens, withInstruments } from "./attributes.mjs";
import { Generator, overlay } from "./generator.mjs";
import { loadTokenizer } from "./tokenizer.mjs";
import { DEFAULT_VERSION, HUB, ALIASES, loadModel, modelBase } from "./midigen-web.mjs";

const asBytes = b => (b instanceof Uint8Array ? b : new Uint8Array(b));
const repr = x => (typeof x === "string" ? `'${x}'` : JSON.stringify(x));

// ---------------------------------------------------------------- windows (data/v4_docs.py)

/** Shift so the first onset is at tick 0. */
export function trimLeading(sc) {
  let min = Infinity;
  for (const t of sc.tracks) for (const n of t.notes) min = Math.min(min, n.time);
  return min !== Infinity && min > 0 ? shiftTime(sc, -min) : sc;
}

/** A score holding only the given tracks (and the time signatures). */
export function subscore(sc, trackIdxs) {
  const out = newScore(sc.tpq);
  out.timeSignatures = sc.timeSignatures.map(x => ({ ...x }));
  for (const i of trackIdxs) out.tracks.push(copyTrack(sc.tracks[i]));
  return out;
}

/** Bar-aligned clip re-based to tick 0, with the meter in force at the start restated at 0. */
export function window(sc, startTick, endTick) {
  const guard = Math.max(1, Math.floor(sc.tpq / 16));
  let w = clip(sc, startTick, endTick - guard, true);
  if (startTick) w = shiftTime(w, -startTick);
  let active = null;
  for (const ts of sc.timeSignatures) if (ts.time <= startTick && (active === null || ts.time >= active.time)) active = ts;
  const inside = w.timeSignatures.filter(ts => ts.time > 0);
  w.timeSignatures = [];
  if (active !== null) w.timeSignatures.push({ time: 0, numerator: active.numerator, denominator: active.denominator });
  w.timeSignatures.push(...inside);
  return w;
}

/** Tick of every bar line from 0 through the end of the last bar. */
export function barEdges(sc) {
  const db = downbeats(sc).map(x => Math.trunc(x));
  if (!db.length) db.push(0);
  const last = db.length > 1 ? db[db.length - 1] - db[db.length - 2] : sc.tpq * 4;
  const end = scoreEnd(sc);
  while (db[db.length - 1] <= end) db.push(db[db.length - 1] + last);
  return db;
}

// ---------------------------------------------------------------- which tracks (accompany_tracks.py)

export class TrackChoiceError extends Error { constructor(m) { super(m); this.name = "TrackChoiceError"; } }

/** {window, bars, available} for the first `bars` bars of an upload, drums normalized, leading silence cut. */
export function prepareWindow(midiBytes, bars) {
  let score = parseMidi(asBytes(midiBytes));
  normalizeDrums(score, "upload.mid");
  score = trimLeading(score);
  const edges = barEdges(score);
  const available = Math.max(0, edges.length - 1);
  if (available < 1) throw new TrackChoiceError("upload has no complete bar to accompany");
  bars = Math.max(1, Math.min(bars, available));
  return { window: window(score, edges[0], edges[bars]), bars, available };
}

export function densest(w) {
  if (!w.tracks.length) throw new TrackChoiceError("upload has no tracks");
  let best = 0;
  w.tracks.forEach((t, i) => { if (t.notes.length > w.tracks[best].notes.length) best = i; });
  return best;
}

export function chooseTrack(w, track) {
  const n = w.tracks.length;
  if (track === null || track === undefined) return densest(w);
  if (!(track >= 0 && track < n)) throw new TrackChoiceError(`track=${track} is out of range; this upload has tracks 0-${n - 1}`);
  if (!w.tracks[track].notes.length) {
    const playable = w.tracks.flatMap((t, i) => (t.notes.length ? [i] : []));
    throw new TrackChoiceError(`track ${track} has no notes in the first bars; tracks with notes: [${playable.join(", ")}]`);
  }
  return track;
}

const toInt = x => {
  if (typeof x === "number" && Number.isInteger(x)) return x;
  if (typeof x === "string" && /^\s*[+-]?\d+\s*$/.test(x)) return parseInt(x, 10);
  throw new TrackChoiceError(`track must be an index, a comma list like 0,2, or 'all'; got ${repr(x)}`);
};

/** `track=` value -> list of indices, "all", or null (auto). Accepts 2, [0, 2], "0,2", "all", "auto". */
export function parseTrackSpec(raw) {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === "number") return [toInt(raw)];
  if (Array.isArray(raw)) return raw.map(toInt);
  const s = String(raw).trim().toLowerCase();
  if (s === "" || s === "auto") return null;
  if (s === "all") return "all";
  const parts = s.split(",").filter(x => x.trim() !== "");
  const out = parts.map(x => { try { return toInt(x); } catch { throw new TrackChoiceError(`track must be an index, a comma list like 0,2, or 'all'; got ${repr(raw)}`); } });
  if (!out.length) throw new TrackChoiceError(`track=${repr(raw)} names no tracks`);
  return out;
}

/** Condition track indices for a parsed spec: null -> [densest], "all" -> every track with notes. */
export function chooseTracks(w, spec) {
  if (spec === null) return [densest(w)];
  if (spec === "all") {
    const live = w.tracks.flatMap((t, i) => (t.notes.length ? [i] : []));
    if (!live.length) throw new TrackChoiceError("upload has no notes in the first bars");
    return live;
  }
  const seen = [];
  for (const i of spec) {
    chooseTrack(w, i);
    if (!seen.includes(i)) seen.push(i);
  }
  return seen;
}

/** {bytes, tracks, names}: what continuation is given for `track=` (the whole file, or just those tracks). */
export function continuationSource(midiBytes, raw) {
  midiBytes = asBytes(midiBytes);
  const score = parseMidi(midiBytes);
  const n = score.tracks.length;
  const live = score.tracks.flatMap((t, i) => (t.notes.length ? [i] : []));
  const names = idx => idx.map(i => score.tracks[i].name || "");
  const spec = parseTrackSpec(raw);
  if (spec === null || spec === "all") return { bytes: midiBytes, tracks: live, names: names(live) };
  const chosen = [];
  for (const i of spec) {
    if (!(i >= 0 && i < n)) throw new TrackChoiceError(`track=${i} is out of range; this upload has tracks 0-${n - 1}`);
    if (!score.tracks[i].notes.length) throw new TrackChoiceError(`track ${i} has no notes; tracks with notes: [${live.join(", ")}]`);
    if (!chosen.includes(i)) chosen.push(i);
  }
  const sorted = [...chosen].sort((a, b) => a - b);
  if (sorted.length === live.length && sorted.every((x, j) => x === live[j])) return { bytes: midiBytes, tracks: live, names: names(live) };
  const sub = newScore(score.tpq);
  sub.timeSignatures = score.timeSignatures.map(x => ({ ...x }));
  sub.tempos = score.tempos.map(x => ({ ...x }));
  sub.keySignatures = score.keySignatures.map(x => ({ ...x }));
  for (const i of chosen) sub.tracks.push(copyTrack(score.tracks[i]));
  return { bytes: writeMidi(sub), tracks: chosen, names: names(chosen) };
}

/** What a track picker needs, with no generation (accompany_tracks.summarize). */
export function summarize(midiBytes, bars = 8) {
  const { window: w, bars: used, available } = prepareWindow(midiBytes, bars);
  const whole = parseMidi(asBytes(midiBytes));
  return {
    fileNoteCounts: whole.tracks.map(t => t.notes.length),
    trackNames: w.tracks.map(t => t.name || ""),
    trackNoteCounts: w.tracks.map(t => t.notes.length),
    trackPrograms: w.tracks.map(t => (t.isDrum ? -1 : t.program)),
    defaultTrack: densest(w),
    bars: used,
    barsAvailable: available,
  };
}

// ---------------------------------------------------------------- infill spans (infill_span.py)

export const CONTEXT_BARS = 16;
export const MAX_SPAN_BARS = 4;

export class InfillError extends Error { constructor(m) { super(m); this.name = "InfillError"; } }

/** [b0, b1) of at most `contextBars` bars around the span, the span as near the middle as the file allows. */
export function contextWindow(nBars, start, bars, contextBars = CONTEXT_BARS) {
  const ctx = Math.min(contextBars, nBars);
  let b0 = start - Math.floor((ctx - bars) / 2);
  b0 = Math.max(0, Math.min(b0, nBars - ctx));
  return [b0, b0 + ctx];
}

export class InfillPlan {
  constructor(score, edges, start, bars, ctxStart, ctxEnd) {
    Object.assign(this, { score, edges, start, bars, ctxStart, ctxEnd });
  }
  get spanTicks() { return [this.edges[this.start], this.edges[this.start + this.bars]]; }
  window() { return window(this.score, this.edges[this.ctxStart], this.edges[this.ctxEnd]); }
  prefix() { return window(this.score, this.edges[this.ctxStart], this.edges[this.start]); }
  suffix() { return window(this.score, this.edges[this.start + this.bars], this.edges[this.ctxEnd]); }
  middle() { return window(this.score, ...this.spanTicks); }
  get prefixBars() { return this.start - this.ctxStart; }
  get suffixBars() { return this.ctxEnd - this.start - this.bars; }
}

/** Check a span request against the upload and the trained shape. `start` is 0-based in the file's bars. */
export function planInfill(midiBytes, start, bars) {
  const score = parseMidi(asBytes(midiBytes));
  normalizeDrums(score, "upload.mid");
  return planScore(score, start, bars);
}

export function planScore(score, start, bars) {
  if (!noteCount(score)) throw new InfillError("upload has no notes");
  const edges = barEdges(score);
  const nBars = edges.length - 1;
  if (!(bars >= 1 && bars <= MAX_SPAN_BARS))
    throw new InfillError(`bars=${bars}: infill can regenerate 1-${MAX_SPAN_BARS} bars at a time (the span lengths it was trained on)`);
  if (start < 1 || start + bars > nBars - 1) {
    const hi = nBars - bars - 1;
    const where = hi >= 1 ? `start=${start}..${hi}` : "none: the upload is too short";
    throw new InfillError(`start=${start}, bars=${bars}: infill needs at least one kept bar before and after the span; ` +
      `this upload has ${nBars} bars (valid ${where})`);
  }
  const [b0, b1] = contextWindow(nBars, start, bars);
  const p = new InfillPlan(score, edges, start, bars, b0, b1);
  if (!noteCount(p.prefix()) && !noteCount(p.suffix())) throw new InfillError("the bars around the span are empty; nothing to infill from");
  return p;
}

const trackKey = t => `${!!t.isDrum}:${t.isDrum ? 0 : Math.trunc(t.program)}`;
const noteOrder = (a, b) => a.time - b.time || a.duration - b.duration || a.pitch - b.pitch || a.velocity - b.velocity;

/** The upload with bars [start, start + bars) replaced by `answer` (a segment from bar 0 of the span). */
export function splice(p, answer) {
  const [s, e] = p.spanTicks;
  const out = { ...p.score, tracks: p.score.tracks.map(copyTrack) };
  for (const k of ["tempos", "timeSignatures", "keySignatures", "markers"]) out[k] = p.score[k].map(x => ({ ...x }));
  for (const tr of out.tracks) {
    const kept = [];
    for (let n of tr.notes) {
      if (s <= n.time && n.time < e) continue;
      n = { ...n };
      if (n.time < s && s < n.time + n.duration) n.duration = s - n.time;
      kept.push(n);
    }
    tr.notes = kept;
  }
  const gen = resample(answer, p.score.tpq);
  for (const gt of gen.tracks) {
    if (!gt.notes.length) continue;
    let dst = out.tracks.find(t => trackKey(t) === trackKey(gt));
    if (!dst) { dst = newTrack("infill", gt.program, gt.isDrum); out.tracks.push(dst); }
    for (let n of gt.notes) {
      n = { ...n, time: n.time + s };
      if (n.time >= e) continue;
      n.duration = Math.max(1, Math.min(n.duration, e - n.time));
      dst.notes.push(n);
    }
    dst.notes.sort(noteOrder);
  }
  return out;
}

// ---------------------------------------------------------------- the server's tasks (modal_serve.py)

export const ACCOMPANY_EMPTY_RETRIES = 2;

/** Call `draw()` -> [result, nNotes] until a result has notes, at most 1 + retries times. */
export async function firstNonempty(draw, retries = ACCOMPANY_EMPTY_RETRIES) {
  let result, n, i;
  for (i = 0; i <= retries; i++) {
    [result, n] = await draw();
    if (n) break;
  }
  return [result, n, Math.min(i, retries) + 1];
}

/** [header names, family asked for] for an accompaniment: the condition's families plus the one asked for. */
export function accompanimentHeader(w, condIndex, instrument = null) {
  const names = headerForScore(w);
  if (isAutoInstrument(instrument)) return [names, null];
  const family = resolveFamily(instrument);
  if (family === null) throw new Error(`unknown instrument ${repr(instrument)}`);
  const idxs = Array.isArray(condIndex) ? condIndex : [condIndex];
  const condFams = [];
  for (const i of idxs) {
    const f = familyOf(w.tracks[i].program, w.tracks[i].isDrum);
    if (!condFams.includes(f)) condFams.push(f);
  }
  return [withInstruments(names, [...condFams, family]), family];
}

/** Ids an accompaniment answer may not use: drums asked for -> no pitched notes; no drums asked for or in the
 *  condition -> no drums (the #78 rules). */
export function accompanimentBanIds(gen, family, condHasDrums) {
  const ban = [gen.sepId, gen.maskId].filter(t => t !== null);
  for (const [k, i] of gen.tokenizer.vocab) {
    if (family === DRUMS) { if (k.startsWith("Pitch_") || (k.startsWith("Program_") && k !== "Program_-1")) ban.push(i); }
    else if (!condHasDrums && (k.startsWith("PitchDrum_") || k === "Program_-1")) ban.push(i);
  }
  return ban.sort((a, b) => a - b);
}

/** The tempo stamped on an output: the caller's, else the upload's first tempo, else null. */
function sourceTempo(score, tempoBpm) {
  if (tempoBpm !== null && tempoBpm !== undefined) return tempoBpm;
  return score.tempos.length ? qpm(score.tempos[0]) : null;
}

/** The tempo the model is told: the caller's, else the upload's first tempo, else 120 (the MIDI default). */
const detectTempo = score => (score.tempos.length ? qpm(score.tempos[0]) : 120.0);

/** MIDI bytes with exactly the given tempo (none when null), never the decoder's own. */
export function scoreToBytes(score, tempoBpm) {
  return writeMidi({ ...score, tempos: tempoBpm ? [{ time: 0, mspq: mspqOf(tempoBpm) }] : [] });
}

export class Midigen {
  /** `gen`: a Generator. */
  constructor(gen) {
    this.gen = gen;
    this.tokenizer = gen.tokenizer;
  }

  // each decode gets its own seed when the caller gave one, so takes and retries differ but repeat
  _seeder(seed) {
    let k = 0;
    return () => (seed === null || seed === undefined ? null : Number(seed) + k++);
  }

  /** Continue the upload (MidiGen.generate_batch). `track` narrows the prompt as the site's track= does. */
  async generate(midiBytes, { maxNewTokens = 512, temperature = 1.2, topK = 50, tempoBpm = null, nSamples = 1, track = null, seed = null, onToken, signal } = {}) {
    const gen = this.gen, tok = this.tokenizer, nextSeed = this._seeder(seed);
    const src = continuationSource(midiBytes, track);
    const score = parseMidi(src.bytes);
    let fullPrompt = tok.encode(score);
    const outTempo = sourceTempo(score, tempoBpm);
    if (tempoBpm === null || tempoBpm === undefined) tempoBpm = detectTempo(score);
    fullPrompt = [...gen.headerIdsFor(headerForScore(score, { tempo: tempoBpm })), ...fullPrompt];

    let [prompt, maxNew] = gen.fitToContext(fullPrompt, maxNewTokens);
    const promptScore = tok.decode(prompt);
    const tpq = Math.max(promptScore.tpq, 1);
    const promptEndSeconds = ((scoreEnd(promptScore) / tpq) * 60.0) / tempoBpm;

    // MidiGen._batched_generate: one sampled stream per take, cut at its first stop token, then postprocessed
    const fed = gen.needsBos(prompt) ? [gen.bosId, ...prompt] : prompt;
    const samples = [];
    for (let i = 0; i < nSamples; i++) {
      const spec = gen.remoteSpec(fed, {
        maxNewTokens: maxNew, temperature, topK, minNewTokens: Math.min(128, Math.floor(maxNew / 4)),
        seed: nextSeed(), banIds: gen.defaultBanIds(),
      });
      const raw = (await gen.decode(spec, { onToken: onToken && ((t, s) => onToken(t, s, i)), signal })).map(Number);
      const cut = raw.findIndex(t => gen.stopIds.has(t));
      samples.push(gen.postprocess(fed, cut < 0 ? raw : raw.slice(0, cut)));
    }
    const midis = samples.map(ids => scoreToBytes(tok.decode([...prompt, ...ids]), outTempo));
    return {
      promptTokens: prompt.length,
      promptTokensTotal: fullPrompt.length,
      promptTruncated: prompt.length < fullPrompt.length,
      promptEndSeconds,
      promptTracks: src.tracks,
      promptTrackNames: src.names,
      generatedTokens: samples.map(s => s.length),
      tempoBpm,
      ids: samples,
      midi: midis[0],
      midis,
    };
  }

  /** Parts to play with the upload over its first `bars` bars (MidiGen.accompany_batch). */
  async accompany(midiBytes, { bars = 8, temperature = 1.0, topK = 50, nSamples = 1, tempoBpm = null, instrument = null, track = null, seed = null, onToken, signal } = {}) {
    const gen = this.gen, tok = this.tokenizer, nextSeed = this._seeder(seed);
    midiBytes = asBytes(midiBytes);
    const prep = prepareWindow(midiBytes, bars);
    const w = prep.window;
    bars = prep.bars;
    const raw = parseMidi(midiBytes);
    const outTempo = sourceTempo(raw, tempoBpm);
    if (tempoBpm === null || tempoBpm === undefined) tempoBpm = detectTempo(raw);

    const condIdx = chooseTracks(w, parseTrackSpec(track));
    const condition = subscore(w, condIdx);
    if (!noteCount(condition)) throw new TrackChoiceError("the chosen track has no notes in the first bars");
    const condIds = tok.encode(condition);

    let [names, family] = accompanimentHeader(w, condIdx.length === 1 ? condIdx[0] : condIdx, instrument);
    if (!names.some(n => n.startsWith("Tempo_"))) names = [...names, ...tempoTokens(w, tempoBpm)];
    const header = gen.headerIdsFor(names);
    const banIds = accompanimentBanIds(gen, family, condition.tracks.some(t => t.isDrum && t.notes.length));

    const midis = [], noteCounts = [], draws = [], ids = [];
    for (let i = 0; i < nSamples; i++) {
      let newIds;
      const draw = async () => {
        newIds = await gen.accompany(condIds, bars, {
          header, banIds, temperature, topK, seed: nextSeed(), signal, onToken: onToken && ((t, s) => onToken(t, s, i)),
        });
        const answer = tok.decode(newIds);
        return [answer, noteCount(answer)];
      };
      const [answer, n, k] = await firstNonempty(draw);
      noteCounts.push(n); draws.push(k); ids.push(newIds);
      midis.push(scoreToBytes(overlay(condition, answer), outTempo));
    }
    let beatsPerBar = 4.0;
    if (w.timeSignatures.length) beatsPerBar = (w.timeSignatures[0].numerator * 4.0) / w.timeSignatures[0].denominator;
    return {
      bars,
      barsAvailable: prep.available,
      conditionTrack: condIdx[0],
      conditionTrackName: w.tracks[condIdx[0]].name || "",
      conditionTracks: condIdx,
      conditionTrackNames: condIdx.map(i => w.tracks[i].name || ""),
      trackNames: w.tracks.map(t => t.name || ""),
      conditionNotes: noteCount(condition),
      generatedNotes: noteCounts,
      draws,
      instrument: family,
      header: names,
      tempoBpm,
      windowSeconds: (bars * beatsPerBar * 60.0) / tempoBpm,
      ids,
      midi: midis[0],
      midis,
    };
  }

  /** Regenerate bars [start, start + bars) of the upload and keep the rest (MidiGen.infill_batch). */
  async infill(midiBytes, { start, bars = 2, temperature = 1.0, topK = 50, nSamples = 1, tempoBpm = null, seed = null, onToken, signal } = {}) {
    const gen = this.gen, tok = this.tokenizer, nextSeed = this._seeder(seed);
    if (start === null || start === undefined) throw new InfillError("start=<bar index> is required (0-based)");
    midiBytes = asBytes(midiBytes);
    const raw = parseMidi(midiBytes);
    const outTempo = sourceTempo(raw, tempoBpm);
    if (tempoBpm === null || tempoBpm === undefined) tempoBpm = detectTempo(raw);

    const p = planInfill(midiBytes, start, bars);
    const prefix = gen.padToBars(tok.encode(p.prefix()), p.prefixBars);
    const suffix = gen.padToBars(tok.encode(p.suffix()), p.suffixBars);
    const header = gen.headerIdsFor(headerForScore(p.window(), { tempo: tempoBpm }));
    const scores = [], counts = [], ids = [];
    for (let i = 0; i < nSamples; i++) {
      const out = await gen.infill(prefix, suffix, bars, {
        header, temperature, topK, seed: nextSeed(), signal, onToken: onToken && ((t, s) => onToken(t, s, i)),
      });
      const answer = tok.decode(out);
      counts.push(noteCount(answer)); ids.push(out);
      scores.push(splice(p, answer));
    }
    const tpq = Math.max(p.score.tpq, 1);
    const [s, e] = p.spanTicks;
    const midis = scores.map(sc => scoreToBytes(sc, outTempo));
    return {
      start: p.start,
      bars: p.bars,
      barsAvailable: p.edges.length - 1,
      context: [p.ctxStart, p.ctxEnd],
      spanSeconds: [((s / tpq) * 60.0) / tempoBpm, ((e / tpq) * 60.0) / tempoBpm],
      generatedNotes: counts,
      tempoBpm,
      ids,
      midi: midis[0],
      midis,
    };
  }

  /** Track names, note counts and the default condition track, for a picker (no generation). */
  tracks(midiBytes, bars = 8) { return summarize(midiBytes, bars); }
}

/** The tokenizer.json of a version on the Hub. */
export async function fetchTokenizer(version = DEFAULT_VERSION, { hub = HUB, fetchImpl = (...a) => globalThis.fetch(...a) } = {}) {
  const url = `${hub.replace(/\/+$/, "")}/${ALIASES[version] || version}/tokenizer.json`;
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`no tokenizer at ${url} (HTTP ${res.status})`);
  return loadTokenizer(await res.json());
}

/** Everything in one call: the model (loadModel's options) and the version's tokenizer, as a Midigen. Pass
 *  `model` to reuse one already loaded. */
export async function loadMidigen({ version = DEFAULT_VERSION, hub = HUB, model = null, maxSeqLen = 8192, fetchImpl, ...modelOpts } = {}) {
  const [m, tok] = await Promise.all([
    model || loadModel({ base: modelBase(version, hub), fetchImpl, ...modelOpts }),
    fetchTokenizer(version, { hub, ...(fetchImpl ? { fetchImpl } : {}) }),
  ]);
  const mg = new Midigen(new Generator(tok, (spec, opts) => m.decode(spec, opts), { maxSeqLen }));
  mg.model = m;
  return mg;
}
