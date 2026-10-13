// midigen-prompt: midigenai's prompt side in a web page, so a page needs no Python to use the model.
//
// midigen-web.mjs samples one generation from a spec (`Generator.remote_spec`); this module builds those specs and
// reads the answers, as `Generator(backend="remote")` does in Python: the tokenizer (MidiTok's REMI, as a
// version's tokenizer.json configures it), the v4 document layouts (sequence_format.py), the attribute header
// (attributes.py), `postprocess` and span infill's plan (infill_span.py). The decoder is any function from a
// spec to sampled ids: midigen-web's model in the page, or a server that samples on a GPU (modal_serve.py
// MidiGen.decode, which runs Generator.decode_spec).
//
//   import { loadGenerator } from "./midigen-prompt.mjs";
//   const g = await loadGenerator({ version: "v5", decode: spec => model.decode(spec) });
//   const body = g.tokenizer.encode(score);           // a Score: see below; the ids Python's tokenizer gives it
//   const ids = await g.continueIds(body, { header, bars: 4, maxNewTokens: 400 });
//   const answer = g.tokenizer.decode(ids);           // a Score at the tokenizer's 16 ticks per quarter
//
// A Score is symusic's, as plain data:
//   {tpq, tracks: [{name, program, is_drum, notes: [{time, duration, pitch, velocity}], controls: [{time, number, value}]}],
//    time_signatures: [{time, numerator, denominator}], tempos: [{time, qpm}]}
// with times in ticks. The ids must be the ones Python makes from the same Score, so every step here follows
// MidiTok 3.0.6 and symusic 0.6 (which order notes, how they round when resampling, what a clip keeps); the
// parity test (tests/test_web_prompt.py) checks it against both on random scores. One meter per score on the
// way in (what every prompt here uses); decoding follows meter changes the model writes.

export const HUB = "https://huggingface.co/nicholasbien/midigenai/resolve/main";
export const DEFAULT_VERSION = "v5";
export const ALIASES = { "v5-rl": "v5" };          // earlier names (midigenai.hub.ALIASES)

// ---------------------------------------------------------------- attributes.py

export const INSTRUMENT_FAMILIES = ["Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass",
  "Reed", "Pipe", "SynthLead", "SynthPad", "SynthFX", "Ethnic", "Percussive", "SoundFX"];
export const DRUMS = "Drums";
export const MAX_INST_TOKENS = 6;
const DENSITY_EDGES = [8, 24, 64];
const POLY_EDGES = [1.5, 3.0];
const RANGE_EDGES = [24, 48];
const TEMPO_EDGES = [70, 90, 110, 130, 150, 175];
export const SOURCES = ["lakh", "lamd", "aria", "gigamidi", "maestro", "pop909", "giantmidi", "user", "fma"];
export const GENRES = ["rock", "pop", "jazz", "classical", "electronic", "hiphop", "rnb", "country", "folk", "latin",
  "blues", "metal", "reggae", "soul", "world", "other"];
export const HEADER_PREFIXES = ["Task_", "Inst_", "Density_", "Poly_", "Range_", "Tempo_", "Source_", "Genre_", "Quality_"];
const FAMILY_ORDER = new Map([...INSTRUMENT_FAMILIES, DRUMS].map((f, i) => [f, i]));

export function isHeaderToken(name) {
  return HEADER_PREFIXES.some((p) => name.startsWith(p));
}

function bucket(x, edges) {
  for (let i = 0; i < edges.length; i++) if (x < edges[i]) return i;
  return edges.length;
}

/** attributes.family_of */
export function familyOf(program, isDrum) {
  if (isDrum) return DRUMS;
  return INSTRUMENT_FAMILIES[Math.floor(Math.max(0, Math.min(127, Math.trunc(program))) / 8)];
}

function instrumentTokens(score) {
  const counts = new Map();              // insertion order: Counter.most_common keeps it among equal counts
  for (const t of score.tracks) {
    if (t.notes.length) {
      const f = familyOf(t.program, t.is_drum);
      counts.set(f, (counts.get(f) || 0) + t.notes.length);
    }
  }
  const fams = [...counts.entries()].map(([f, n], i) => [f, n, i])
    .sort((a, b) => b[1] - a[1] || a[2] - b[2]).slice(0, MAX_INST_TOKENS).map(([f]) => f);
  return fams.sort((a, b) => FAMILY_ORDER.get(a) - FAMILY_ORDER.get(b)).map((f) => `Inst_${f}`);
}

function ticksPerBarOf(score) {
  let num = 4, den = 4;
  const ts = score.time_signatures || [];
  if (ts.length) {
    const first = ts.reduce((a, b) => (b.time < a.time ? b : a));
    num = first.numerator; den = first.denominator;
  }
  return Math.trunc(score.tpq * 4 * num / den);
}

function contentTokens(score) {
  const notes = score.tracks.flatMap((t) => t.notes);
  if (!notes.length) return [];
  let start = Infinity, end = -Infinity;
  for (const n of notes) { start = Math.min(start, n.time); end = Math.max(end, n.time + n.duration); }
  const nBars = Math.max(1.0, (end - start) / ticksPerBarOf(score));
  const density = notes.length / nBars;
  const onsets = new Set(notes.map((n) => n.time)).size;
  const poly = notes.length / Math.max(1, onsets);
  const pitched = score.tracks.filter((t) => !t.is_drum).flatMap((t) => t.notes.map((n) => n.pitch));
  const span = pitched.length ? Math.max(...pitched) - Math.min(...pitched) : 0;
  return [`Density_${bucket(density, DENSITY_EDGES)}`, `Poly_${bucket(poly, POLY_EDGES)}`, `Range_${bucket(span, RANGE_EDGES)}`];
}

/** attributes.tempo_tokens with the caller's tempo (the clock the answer plays at): [] without one. */
export function tempoTokens(tempo) {
  if (tempo === null || tempo === undefined || !(tempo > 0)) return [];
  return [`Tempo_${bucket(tempo, TEMPO_EDGES)}`];
}

/** attributes.header_for_score with an explicit tempo (the score's own tempo events are not read here). */
export function headerForScore(score, { tempo = null, source = null, genres = [], quality = null } = {}) {
  const names = [...instrumentTokens(score), ...contentTokens(score), ...tempoTokens(tempo)];
  if (source && SOURCES.includes(source)) names.push(`Source_${source}`);
  for (const g of genres || []) if (GENRES.includes(g)) names.push(`Genre_${g}`);
  if (quality !== null && quality !== undefined && Math.trunc(quality) >= 0 && Math.trunc(quality) < 4) names.push(`Quality_${Math.trunc(quality)}`);
  return names;
}

/** attributes.sort_header: canonical family order, stable within a family. */
export function sortHeader(names) {
  const rank = (n) => HEADER_PREFIXES.findIndex((p) => n.startsWith(p));
  return names.map((n, i) => [n, i]).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([n]) => n);
}

/** attributes.with_instruments: `names` with its Inst_ tokens replaced by exactly `families` (deduped, capped). */
export function withInstruments(names, families) {
  const unknown = families.filter((f) => !FAMILY_ORDER.has(f));
  if (unknown.length) throw new Error(`not instrument families: ${unknown.join(", ")}`);
  const picked = [...new Set(families)].sort((a, b) => FAMILY_ORDER.get(a) - FAMILY_ORDER.get(b)).slice(0, MAX_INST_TOKENS);
  return sortHeader([...picked.map((f) => `Inst_${f}`), ...names.filter((n) => !n.startsWith("Inst_"))]);
}

// ---------------------------------------------------------------- symusic, as far as the prompts use it

const roundHalfAway = (x) => (x < 0 ? -Math.round(-x) : Math.round(x));

export function emptyScore(tpq) {
  return { tpq, tracks: [], time_signatures: [], tempos: [] };
}

function copyTrack(t, notes = t.notes, controls = t.controls || []) {
  return { name: t.name || "", program: t.program, is_drum: !!t.is_drum, notes: notes.map((n) => ({ ...n })),
    controls: controls.map((c) => ({ ...c })) };
}

export function copyScore(sc) {
  return { tpq: sc.tpq, tracks: sc.tracks.map((t) => copyTrack(t)), time_signatures: (sc.time_signatures || []).map((x) => ({ ...x })),
    tempos: (sc.tempos || []).map((x) => ({ ...x })) };
}

/** Score.end(): the last tick anything happens on (a note's end, a control, a tempo or meter change). */
export function scoreEnd(sc) {
  let end = 0;
  for (const t of sc.tracks) {
    for (const n of t.notes) end = Math.max(end, n.time + n.duration);
    for (const c of t.controls || []) end = Math.max(end, c.time);
  }
  for (const x of sc.time_signatures || []) end = Math.max(end, x.time);
  for (const x of sc.tempos || []) end = Math.max(end, x.time);
  return end;
}

/** Note order in symusic's sort(): time, then duration, pitch, velocity. */
export function noteCmp(a, b) {
  return a.time - b.time || a.duration - b.duration || a.pitch - b.pitch || a.velocity - b.velocity;
}

/** Score.resample(tpq, min_dur): times and durations each rounded on their own (half away from zero). */
export function resample(sc, tpq, minDur = 0) {
  const k = (t) => roundHalfAway((t * tpq) / sc.tpq);
  return {
    tpq,
    tracks: sc.tracks.map((t) => copyTrack(t, t.notes.map((n) => ({ ...n, time: k(n.time), duration: Math.max(minDur, k(n.duration)) })),
      (t.controls || []).map((c) => ({ ...c, time: k(c.time) })))),
    time_signatures: (sc.time_signatures || []).map((x) => ({ ...x, time: k(x.time) })),
    tempos: (sc.tempos || []).map((x) => ({ ...x, time: k(x.time) })),
  };
}

/** Score.clip(start, end, clip_end): notes that start in [start, end) (and, with clip_end, also end by `end`:
 *  symusic drops a note that rings past the end rather than cutting it), controls in [start, end). Meter and
 *  tempo events before `start` move to it; empty tracks stay. */
export function clip(sc, start, end, clipEnd = false) {
  const inNote = (n) => n.time >= start && n.time < end && (!clipEnd || n.time + n.duration <= end);
  const keepTimed = (xs) => {
    const before = xs.filter((x) => x.time < start), inside = xs.filter((x) => x.time >= start && x.time < end);
    return (before.length ? [{ ...before[before.length - 1], time: start }] : []).concat(inside.map((x) => ({ ...x })));
  };
  return {
    tpq: sc.tpq,
    tracks: sc.tracks.map((t) => copyTrack(t, t.notes.filter(inNote), (t.controls || []).filter((c) => c.time >= start && c.time < end))),
    time_signatures: keepTimed(sc.time_signatures || []),
    tempos: keepTimed(sc.tempos || []),
  };
}

export function shiftTime(sc, dt) {
  const s = (xs) => xs.map((x) => ({ ...x, time: x.time + dt }));
  return { tpq: sc.tpq, tracks: sc.tracks.map((t) => ({ ...copyTrack(t), notes: s(t.notes), controls: s(t.controls || []) })),
    time_signatures: s(sc.time_signatures || []), tempos: s(sc.tempos || []) };
}

function oneMeter(sc) {
  const ts = sc.time_signatures || [];
  const kinds = new Set(ts.map((x) => `${x.numerator}/${x.denominator}`));
  if (kinds.size > 1) throw new Error("midigen-prompt: one meter per score (this one changes meter)");
  return ts.length ? ts[0] : null;
}

/** data/v4_docs.bar_edges: every bar line from 0 through the end of the last bar (one meter). */
export function barEdges(sc) {
  const ts = oneMeter(sc), end = scoreEnd(sc);
  const bar = ts ? Math.trunc(sc.tpq * 4 * ts.numerator / ts.denominator) : sc.tpq * 4;
  // get_downbeats() lists the bar lines up to the end; with only the first one, the step is a 4/4 bar
  const step = bar <= end ? bar : sc.tpq * 4;
  const db = [0];
  while (db[db.length - 1] <= end) db.push(db[db.length - 1] + step);
  return db;
}

/** data/v4_docs._window: a bar-aligned clip re-based to tick 0, its last half-position left out, the meter restated. */
export function windowOf(sc, startTick, endTick) {
  const guard = Math.max(1, Math.floor(sc.tpq / 16));
  let w = clip(sc, startTick, endTick - guard, true);
  if (startTick) w = shiftTime(w, -startTick);
  let active = null;
  for (const ts of sc.time_signatures || []) if (ts.time <= startTick && (active === null || ts.time >= active.time)) active = ts;
  const inside = (w.time_signatures || []).filter((ts) => ts.time > 0);
  w.time_signatures = [...(active ? [{ time: 0, numerator: active.numerator, denominator: active.denominator }] : []), ...inside];
  return w;
}

// ---------------------------------------------------------------- tokenizer.py's drum promotion

const DRUM_HINTS_ANYWHERE = ["drum", "perc", "kick", "snare", "cymbal", "hihat", "hi-hat", "clap", "batter", "schlag"];
const BOUNDED_RE = /(?<![0-9a-z])(drm|kit|hat|tom|beat|808|909)(?![0-9a-z])/;
const GM_KICK = [35, 36], GM_SNARE = [38, 40], GM_HAT = [42, 44, 46];

export function nameSaysDrums(name) {
  const n = (name || "").toLowerCase();
  return DRUM_HINTS_ANYWHERE.some((k) => n.includes(k)) || BOUNDED_RE.test(n);
}

export function looksLikeDrums(track, minNotes = 16) {
  const pitches = track.notes.map((n) => n.pitch);
  if (pitches.length < minNotes) return false;
  const distinct = new Set(pitches);
  if (distinct.size > 8) return false;
  const inRange = pitches.filter((p) => p >= 35 && p < 60).length / pitches.length;
  if (inRange < 0.85) return false;
  return [GM_KICK, GM_SNARE, GM_HAT].every((fam) => fam.some((p) => distinct.has(p)));
}

/** tokenizer.normalize_drums: promote drum-like tracks to is_drum, in place; how many it promoted. */
export function normalizeDrums(score, filenameHint = "") {
  const fname = nameSaysDrums(filenameHint);
  let changed = 0;
  for (const t of score.tracks) {
    if (t.is_drum) continue;
    if (nameSaysDrums(t.name) || (fname && score.tracks.length === 1) || looksLikeDrums(t)) {
      t.is_drum = true;
      changed += 1;
    }
  }
  return changed;
}

// ---------------------------------------------------------------- MidiTok's REMI

/** numpy's np_get_closest over a sorted array: the nearest value, the higher one on a tie. */
function closest(arr, v) {
  let lo = 0, hi = arr.length;                         // searchsorted(side="left")
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
  let i = lo;
  if (i === arr.length || Math.abs(v - arr[Math.max(i - 1, 0)]) < Math.abs(v - arr[Math.min(i, arr.length - 1)])) i -= 1;
  return arr[i];
}

function pyRange([a, b]) {
  return Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
}

export class Tokenizer {
  /** `params`: a tokenizer.json (`{config, tokenization: "REMI", ...}`). */
  constructor(params) {
    if (params.tokenization !== "REMI") throw new Error(`midigen-prompt reads REMI tokenizers (v4 line), not ${params.tokenization}`);
    const c = this.config = params.config;
    for (const k of ["use_chords", "use_rests", "use_tempos", "use_sustain_pedals", "use_pitch_bends", "use_pitch_intervals"]) {
      if (c[k]) throw new Error(`midigen-prompt: tokenizer option ${k} is not supported`);
    }
    const extra = c.additional_params || {};
    if (extra.max_bar_embedding != null || extra.use_bar_end_tokens || extra.add_trailing_bars) {
      throw new Error("midigen-prompt: Bar embeddings, Bar_End and trailing bars are not supported");
    }
    this.beatRes = Object.entries(c.beat_res).map(([k, res]) => [...k.split("_").map(Number), res]);
    this.maxPos = Math.max(...this.beatRes.map((r) => r[2]));
    this.timeSignatures = [];
    for (const [den, nums] of Object.entries(c.time_signature_range)) for (const num of nums) this.timeSignatures.push([num, Number(den)]);
    const maxDen = Math.max(...this.timeSignatures.map((t) => t[1]));
    this.tpbPerTs = {};
    for (const den of Object.keys(c.time_signature_range)) this.tpbPerTs[den] = this.maxPos * Math.floor(maxDen / Number(den));
    this.timeDivision = this.tpbPerTs[4];
    // durations as (beat, pos, res), the zero one dropped, the longest last (MusicTokenizer._create_durations_tuples)
    const durs = [];
    for (const [a, b, res] of this.beatRes) for (const beat of pyRange([a, b])) for (let pos = 0; pos < res; pos++) durs.push([beat, pos, res]);
    const top = this.beatRes.reduce((m, r) => (r[0] > m[0] || (r[0] === m[0] && r[1] > m[1]) ? r : m));
    durs.push([Math.max(top[0], top[1]), 0, top[2]]);
    durs.shift();
    this.durations = durs;
    this.tpbTicks = {};                                // tpb -> sorted duration ticks
    this.tpbTicksToToken = {};                         // tpb -> {ticks: "b.p.r"}
    this.tpbTokenToTicks = {};                         // tpb -> {"b.p.r": ticks}
    for (const tpb of Object.values(this.tpbPerTs)) {
      const ticks = durs.map(([b, p, r]) => Math.floor((b * r + p) * tpb / r));
      this.tpbTicks[tpb] = ticks;
      this.tpbTicksToToken[tpb] = {};
      this.tpbTokenToTicks[tpb] = {};
      durs.forEach((d, i) => { this.tpbTicksToToken[tpb][ticks[i]] = d.join("."); this.tpbTokenToTicks[tpb][d.join(".")] = ticks[i]; });
    }
    const nv = c.num_velocities;
    this.velocities = c.use_velocities === false ? [] : Array.from({ length: nv }, (_, i) => (i + 1 === nv ? 127 : Math.trunc((i + 1) * (127 / nv))));
    this.programs = c.programs;
    this.durationPrograms = new Set(c.use_note_duration_programs);
    // the vocabulary: special tokens, then REMI's base vocabulary (REMI._create_base_vocabulary)
    const vocab = [...c.special_tokens, "Bar_None"];
    for (let p = c.pitch_range[0]; p <= c.pitch_range[1]; p++) vocab.push(`Pitch_${p}`);
    if (c.use_velocities !== false) for (const v of this.velocities) vocab.push(`Velocity_${v}`);
    if (c.use_note_duration_programs.length) for (const d of durs) vocab.push(`Duration_${d.join(".")}`);
    const maxBeats = Math.max(...this.timeSignatures.map((t) => t[0]));
    for (let i = 0; i < this.maxPos * maxBeats; i++) vocab.push(`Position_${i}`);
    if (c.use_pitchdrum_tokens) for (let p = c.drums_pitch_range[0]; p <= c.drums_pitch_range[1]; p++) vocab.push(`PitchDrum_${p}`);
    if (c.use_programs) for (const p of c.programs) vocab.push(`Program_${p}`);
    if (c.use_time_signatures) for (const [n, d] of this.timeSignatures) vocab.push(`TimeSig_${n}/${d}`);
    this.vocab = new Map();
    this.tokens = [];
    for (const t of vocab) {
      if (this.vocab.has(t)) continue;
      this.vocab.set(t, this.tokens.length);
      this.tokens.push(t);
    }
  }

  get vocabSize() { return this.tokens.length; }

  id(token) {
    const i = this.vocab.get(token);
    if (i === undefined) throw new Error(`not in the vocabulary: ${token}`);
    return i;
  }

  /** tokenizer.special_id: the id of BOS / MASK / Inst_Piano ..., or null. */
  specialId(name) {
    for (const t of [`${name}_None`, name]) if (this.vocab.has(t)) return this.vocab.get(t);
    return null;
  }

  // ---- encode: MusicTokenizer.preprocess_score + _score_to_tokens + REMI._add_time_events

  _preprocess(score) {
    let ts = (score.time_signatures || []).filter((x) => this.timeSignatures.some(([n, d]) => n === x.numerator && d === x.denominator));
    if (!ts.length || ts[0].time !== 0) ts = [{ time: 0, numerator: 4, denominator: 4 }, ...ts];
    if (new Set(ts.map((x) => `${x.numerator}/${x.denominator}`)).size > 1) throw new Error("midigen-prompt: one meter per score");
    ts = [ts[0]];
    const maxDen = Math.max(...ts.map((x) => x.denominator));
    const tpq = Math.trunc(this.maxPos * maxDen / 4);
    let sc = score.tpq !== tpq ? resample(score, tpq, 1) : copyScore(score);
    sc.time_signatures = ts.map((x) => ({ ...x, time: 0 }));
    // merge tracks of one program (drums are program -1) into the first, notes sorted
    const prog = (t) => (t.is_drum ? -1 : t.program);
    const groups = new Map();
    sc.tracks.forEach((t, i) => { const p = prog(t); if (!groups.has(p)) groups.set(p, []); groups.get(p).push(i); });
    const drop = new Set();
    for (const idx of groups.values()) {
      if (idx.length < 2) continue;
      const first = sc.tracks[idx[0]];
      first.name = [first.name, ...idx.slice(1).map((i) => sc.tracks[i].name)].join(" / ");
      first.notes = idx.flatMap((i) => sc.tracks[i].notes).sort(noteCmp);
      first.controls = idx.flatMap((i) => sc.tracks[i].controls).sort((a, b) => a.time - b.time || a.number - b.number || a.value - b.value);
      for (const i of idx.slice(1)) drop.add(i);
    }
    sc.tracks = sc.tracks.filter((_, i) => !drop.has(i));
    const tpb = Math.trunc(sc.tpq * 4 / ts[0].denominator);
    const durTicks = this.tpbTicks[tpb];
    if (!durTicks) throw new Error(`midigen-prompt: no durations at ${tpb} ticks per beat`);
    const c = this.config;
    sc.tracks = sc.tracks.filter((t) => {
      if (!t.notes.length || (c.use_programs && !this.programs.includes(prog(t)))) return false;
      const [lo, hi] = t.is_drum && c.use_pitchdrum_tokens ? c.drums_pitch_range : c.pitch_range;
      t.notes = t.notes.filter((n) => n.pitch >= lo && n.pitch <= hi);
      if (!t.notes.length) return false;
      if (c.use_velocities !== false) for (const n of t.notes) n.velocity = closest(this.velocities, n.velocity);
      if (this.durationPrograms.has(prog(t))) for (const n of t.notes) n.duration = closest(durTicks, n.duration);
      return true;
    });
    return { sc, tpb };
  }

  /** The token strings for a Score. */
  encodeTokens(score) {
    const { sc, tpb } = this._preprocess(score);
    const c = this.config;
    let events = sc.time_signatures.map((x) => ({ type: "TimeSig", value: `${x.numerator}/${x.denominator}`, time: x.time, program: 0 }));
    for (const t of sc.tracks) {
      const program = t.is_drum ? -1 : t.program;
      const useDur = this.durationPrograms.has(program);
      for (const n of t.notes) {
        events.push({ type: c.use_pitchdrum_tokens && t.is_drum ? "PitchDrum" : "Pitch", value: n.pitch, time: n.time, program, end: n.time + n.duration });
        if (c.use_velocities !== false) events.push({ type: "Velocity", value: n.velocity, time: n.time, program });
        if (useDur) {
          const d = this.tpbTicksToToken[tpb][n.duration];
          if (d === undefined) throw new Error(`midigen-prompt: no Duration token for ${n.duration} ticks`);
          events.push({ type: "Duration", value: d, time: n.time, program });
        }
      }
    }
    events.sort((a, b) => a.time - b.time);                                  // stable, as list.sort
    if (c.use_programs && c.program_changes) {
      const out = [];
      let prev = null;
      for (const e of events) {
        if (e.program !== null && e.program !== prev && e.type !== "TimeSig" && e.type !== "Tempo") {
          prev = e.program;
          out.push({ type: "Program", value: e.program, time: e.time, program: 0 });
        }
        out.push(e);
      }
      events = out;
    }
    return this._addTimeEvents(events, sc.tpq).map((e) => `${e.type}_${e.value}`);
  }

  /** The ids Python's tokenizer(score).ids gives for the same Score. */
  encode(score) {
    return this.encodeTokens(score).map((t) => this.id(t));
  }

  _ticksPerUnits(ts, tpq) {
    const tpbeat = ts[1] === 4 ? tpq : Math.trunc(tpq * (4 / ts[1]));
    return { bar: Math.trunc(tpbeat * ts[0]), beat: tpbeat, pos: Math.floor(tpbeat / this.maxPos) };
  }

  _addTimeEvents(events, tpq) {
    const useTs = !!this.config.use_time_signatures;
    const all = [];
    let currentBar = -1, barAtLastTs = 0, previousTick = -1, tickAtLastTs = 0, tickAtCurrentBar = 0;
    let ts = [4, 4];
    if (useTs) {
      const first = events.find((e) => e.type === "TimeSig");
      if (first) ts = first.value.split("/").map(Number);
    }
    let u = this._ticksPerUnits(ts, tpq);
    const unitsBetween = (a, b, n) => Math.floor((b - a) / n);
    for (const e of events) {
      if (e.time !== previousTick) {
        const newBars = barAtLastTs + unitsBetween(tickAtLastTs, e.time, u.bar) - currentBar;
        for (let i = 0; i < newBars; i++) {
          currentBar += 1;
          tickAtCurrentBar = tickAtLastTs + (currentBar - barAtLastTs) * u.bar;
          all.push({ type: "Bar", value: "None", time: tickAtCurrentBar });
          if (useTs && !(e.type === "TimeSig" && i + 1 === newBars)) all.push({ type: "TimeSig", value: `${ts[0]}/${ts[1]}`, time: tickAtCurrentBar });
        }
        if (e.type !== "TimeSig") all.push({ type: "Position", value: unitsBetween(tickAtCurrentBar, e.time, u.pos), time: e.time });
        previousTick = e.time;
      }
      if (e.type === "TimeSig") {
        barAtLastTs += unitsBetween(tickAtLastTs, e.time, u.bar);
        tickAtLastTs = e.time;
        ts = e.value.split("/").map(Number);
        u = this._ticksPerUnits(ts, tpq);
        previousTick -= 1;
      }
      all.push(e);
    }
    return all;
  }

  // ---- decode: REMI._tokens_to_score (one token stream for every program)

  /** A Score (tpq = the tokenizer's time division) from ids, as tokenizer.decode(ids) makes it. */
  decode(ids) {
    const seq = ids.map((i) => {
      const t = this.tokens[i];
      if (t === undefined) throw new Error(`not a token id: ${i}`);
      return t;
    });
    return this.decodeTokens(seq);
  }

  decodeTokens(seq) {
    const c = this.config, tpq = this.timeDivision;
    const split = (tok) => {
      const parts = tok.split("_");
      if (parts.length !== 2) throw new Error(`malformed token ${tok}`);
      return parts;
    };
    const tracks = new Map();
    const trackFor = (prog) => {
      if (!tracks.has(prog)) tracks.set(prog, { name: prog === -1 ? "Drums" : "", program: prog === -1 ? 0 : prog, is_drum: prog === -1, notes: [], controls: [] });
      return tracks.get(prog);
    };
    const timeSigs = [];
    if (c.use_time_signatures) {
      for (const tok of seq) {
        const [type, val] = split(tok);
        if (type === "TimeSig") { const [n, d] = val.split("/").map(Number); timeSigs.push({ time: 0, numerator: n, denominator: d }); break; }
        if (["Pitch", "PitchDrum", "Velocity", "Duration", "PitchBend", "Pedal"].includes(type)) break;
      }
    }
    if (!timeSigs.length) timeSigs.push({ time: 0, numerator: 4, denominator: 4 });
    let cur = timeSigs[timeSigs.length - 1];
    let ticksPerBar = Math.trunc((cur.denominator === 4 ? tpq : Math.trunc(tpq * (4 / cur.denominator))) * cur.numerator);
    let ticksPerBeat = this.tpbPerTs[cur.denominator];
    let ticksPerPos = Math.floor(ticksPerBeat / this.maxPos);
    let currentTick = 0, tickAtLastTs = 0, tickAtCurrentBar = 0, currentBar = -1, barAtLastTs = 0, currentProgram = 0;
    let useDur = this.durationPrograms.has(currentProgram);
    const durOffset = c.use_velocities !== false ? 2 : 1;
    for (let ti = 0; ti < seq.length; ti++) {
      const tok = seq[ti];
      const [type, val] = split(tok);
      if (tok === "Bar_None") {
        currentBar += 1;
        if (currentBar > 0) currentTick = tickAtCurrentBar + ticksPerBar;
        tickAtCurrentBar = currentTick;
      } else if (type === "Position") {
        if (currentBar === -1) currentBar = 0;
        currentTick = tickAtCurrentBar + Number(val) * ticksPerPos;
      } else if (type === "Pitch" || type === "PitchDrum") {
        const pitch = Number(val);
        if (!(c.pitch_range[0] <= pitch && pitch <= c.pitch_range[1])) continue;
        // a sequence cut mid-note (an IndexError in Python) drops the note
        if ((c.use_velocities !== false && ti + 1 >= seq.length) || (useDur && ti + durOffset >= seq.length)) continue;
        let velType = "Velocity", vel = 100, durType = "Duration", dur;
        if (c.use_velocities !== false) [velType, vel] = split(seq[ti + 1]);
        if (useDur) [durType, dur] = split(seq[ti + durOffset]);
        else dur = Math.trunc(c.default_note_duration * ticksPerBeat);
        if (velType === "Velocity" && durType === "Duration") {
          if (typeof dur === "string") dur = this.tpbTokenToTicks[ticksPerBeat][dur];
          trackFor(currentProgram).notes.push({ time: currentTick, duration: dur, pitch, velocity: Number(vel) });
        }
      } else if (type === "Program") {
        currentProgram = Number(val);
        useDur = this.durationPrograms.has(currentProgram);
      } else if (type === "TimeSig") {
        const [num, den] = val.split("/").map(Number);
        if (num !== cur.numerator || den !== cur.denominator) {
          cur = { time: currentTick, numerator: num, denominator: den };
          timeSigs.push(cur);
          tickAtLastTs = tickAtCurrentBar;
          barAtLastTs = currentBar;
          ticksPerBar = Math.trunc((den === 4 ? tpq : Math.trunc(tpq * (4 / den))) * num);
          ticksPerBeat = this.tpbPerTs[den];
          ticksPerPos = Math.floor(ticksPerBeat / this.maxPos);
        }
      }
    }
    void tickAtLastTs; void barAtLastTs;              // tracked as REMI does; only Rest tokens read them
    return { tpq, tracks: [...tracks.values()], time_signatures: timeSigs, tempos: [{ time: 0, qpm: 120 }] };
  }
}

// ---------------------------------------------------------------- sequence_format.py + Generator(backend="remote")

export const MIN_PROMPT_TOKENS = 256;
export const ACCOMP_TOKENS_PER_BAR = 160;

/** generate.fit_to_context */
export function fitToContext(promptIds, maxNewTokens, maxSeqLen, minPromptTokens = MIN_PROMPT_TOKENS) {
  let prompt = [...promptIds];
  if (maxSeqLen <= 0) return [prompt, maxNewTokens];
  const minPrompt = Math.min(minPromptTokens, prompt.length, Math.floor(maxSeqLen / 2));
  maxNewTokens = Math.max(0, Math.min(maxNewTokens, maxSeqLen - minPrompt));
  const keep = maxSeqLen - maxNewTokens;
  if (prompt.length > keep) prompt = prompt.slice(0, keep);
  return [prompt, maxNewTokens];
}

/** generate.accompaniment_budget */
export function accompanimentBudget(condTokens, bars) {
  const perBar = Math.max(ACCOMP_TOKENS_PER_BAR, 2 * condTokens / Math.max(1, bars));
  return Math.trunc(perBar * bars) + 64;
}

export class Generator {
  /** `decode(spec) -> ids` (or a promise of them) samples one generation: midigen-web's model.decode, or a server. */
  constructor(tokenizer, decode, { maxSeqLen = 8192, version = null } = {}) {
    this.tokenizer = tokenizer;
    this._decode = decode;
    this.maxSeqLen = maxSeqLen;
    this.version = version;
    const tok = tokenizer;
    this.bosId = tok.specialId("BOS");
    this.eosId = tok.specialId("EOS");
    this.v4 = tok.vocab.has("MASK_None") && tok.vocab.has("Bar_None");
    if (!this.v4) throw new Error("midigen-prompt builds v4-line prompts (v4, v4-large, v5, v5-base)");
    const headerFamily = new Map();
    for (const [name, id] of tok.vocab) if (isHeaderToken(name)) headerFamily.set(id, HEADER_PREFIXES.find((p) => name.startsWith(p)));
    this.sp = { bos: this.bosId, eos: this.eosId, sep: tok.specialId("SEP"), mask: tok.specialId("MASK"), bar: tok.vocab.get("Bar_None"),
      taskAccomp: tok.specialId("Task_accomp"), taskInfill: tok.specialId("Task_infill"), headerIds: new Set(headerFamily.keys()), headerFamily };
    this.barId = this.sp.bar;
    this.timesigIds = new Set([...tok.vocab].filter(([k]) => k.startsWith("TimeSig_")).map(([, v]) => v));
    this.stopIds = new Set([this.eosId, this.sp.sep, this.sp.mask, this.sp.bos].filter((t) => t !== null));
  }

  /** Specials.header_ids_for: header names -> ids (a family this tokenizer predates is skipped). */
  headerIdsFor(names) {
    const known = new Set(this.sp.headerFamily.values());
    const out = [];
    for (const n of names) {
      const id = this.tokenizer.specialId(n);
      if (id === null) {
        const fam = HEADER_PREFIXES.find((p) => n.startsWith(p));
        if (fam !== undefined && !known.has(fam)) continue;
        throw new Error(`not a header token: ${n}`);
      }
      out.push(id);
    }
    return out;
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
    return this.headerIdsFor(sortHeader(names));
  }

  countBars(ids) {
    let n = 0;
    for (const t of ids) if (t === this.barId) n += 1;
    return n;
  }

  /** Append empty `Bar TimeSig` pairs so `ids` spans exactly `nBars`. */
  padToBars(ids, nBars) {
    const have = this.countBars(ids);
    if (have > nBars) throw new Error(`prompt spans ${have} bars > ${nBars}`);
    const ts = ids.find((t) => this.timesigIds.has(t)) ?? this.tokenizer.id("TimeSig_4/4");
    const out = [...ids];
    for (let i = have; i < nBars; i++) out.push(this.barId, ts);
    return out;
  }

  closeBar(ids) {
    return this.endsOnBarLine(ids) ? [...ids] : this.padToBars(ids, this.countBars(ids) + 1);
  }

  endsOnBarLine(ids) {
    if (!ids.length) return false;
    let tail = ids.filter((t) => !this.sp.headerIds.has(t) && t !== this.bosId);
    if (tail.length && this.timesigIds.has(tail[tail.length - 1])) tail = tail.slice(0, -1);
    return tail.length > 0 && tail[tail.length - 1] === this.barId;
  }

  _needsBos(prompt) {
    return this.bosId !== null && (!prompt.length || prompt[0] !== this.bosId);
  }

  fitToContext(promptIds, maxNewTokens) {
    const added = this._needsBos(promptIds);
    let ids = added ? [this.bosId, ...promptIds] : [...promptIds];
    [ids, maxNewTokens] = fitToContext(ids, maxNewTokens, this.maxSeqLen);
    return [added ? ids.slice(1) : ids, maxNewTokens];
  }

  defaultBanIds() {
    return [this.sp.sep, this.sp.mask, this.bosId].filter((t) => t !== null).sort((a, b) => a - b);
  }

  /** Generator.remote_spec: one generation, as plain JSON (what midigen-web's decode and a server's take). */
  remoteSpec(promptIds, { maxNewTokens, temperature, topK, minNewTokens, seed, banIds = null, stopAfterBars = null, trimLeadingBars = true }) {
    const num = (xs) => [...xs].map(Number).sort((a, b) => a - b);
    return {
      prompt: promptIds.map(Number),
      vocab_size: this.tokenizer.vocabSize,
      max_new_tokens: Math.trunc(maxNewTokens),
      max_seq_len: Math.trunc(this.maxSeqLen),
      temperature: Number(temperature),
      top_k: topK === null || topK === undefined ? null : Math.trunc(topK),
      min_new_tokens: Math.trunc(minNewTokens),
      eos_id: this.eosId,
      ban_ids: num(banIds || []),
      stop_ids: num(this.stopIds),
      bar_id: this.barId,
      timesig_ids: num(this.timesigIds),
      stop_after_bars: stopAfterBars ? Math.trunc(stopAfterBars) : null,
      trim_leading_bars: !!trimLeadingBars,
      seed: seed === null || seed === undefined ? null : Math.trunc(seed),
      ...(this.version ? { version: this.version } : {}),
    };
  }

  /** Generator.generate_ids: fit, BOS, sample through the decoder, postprocess. Resolves to the kept ids. */
  async generateIds(promptIds, { maxNewTokens = 512, temperature = 1.0, topK = 50, minNewTokens = 0, seed = null,
    stopAfterBars = null, banIds = null, trimLeadingBars = true, onSpec = null } = {}) {
    if (banIds === null) banIds = this.defaultBanIds();
    let [prompt, maxNew] = this.fitToContext(promptIds, maxNewTokens);
    if (this._needsBos(prompt)) prompt = [this.bosId, ...prompt];
    const spec = this.remoteSpec(prompt, { maxNewTokens: maxNew, temperature, topK, minNewTokens, seed, banIds, stopAfterBars, trimLeadingBars });
    if (onSpec) onSpec(spec);
    const raw = (await this._decode(spec)).map(Number);
    return this.postprocess(prompt, raw, { stopAfterBars, trimLeadingBars });
  }

  /** Generator.postprocess: stop at EOS / SEP / MASK / BOS, drop leading empty bars, honor stopAfterBars. */
  postprocess(promptIds, raw, { stopAfterBars = null, trimLeadingBars = true } = {}) {
    const out = [];
    let bars = stopAfterBars && this.endsOnBarLine(promptIds) ? 1 : 0;
    let started = !trimLeadingBars;
    let held = [];
    const heldOpen = !this.endsOnBarLine(promptIds);
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
          if (stopAfterBars && h === this.barId) bars += 1;
          out.push(h);
        }
        held = [];
      }
      if (stopAfterBars && t === this.barId) {
        bars += 1;
        if (bars > stopAfterBars) return out;
      }
      out.push(t);
    }
    return out;
  }

  // ---- the v4 tasks (Generator.continue_ids / accompany / infill)

  continuationPrompt(header, body) { return [this.sp.bos, ...header, ...body]; }
  accompanimentPrompt(header, cond) { return [this.sp.bos, ...(this.sp.taskAccomp !== null ? [this.sp.taskAccomp] : []), ...header, ...cond, this.sp.sep]; }
  infillPrompt(header, prefix, suffix) {
    return [this.sp.bos, ...(this.sp.taskInfill !== null ? [this.sp.taskInfill] : []), ...header, ...prefix, this.sp.mask, ...suffix, this.sp.sep];
  }

  async continueIds(bodyIds, { header = [], bars = null, ...opts } = {}) {
    let prompt = [...bodyIds];
    if (bars) prompt = this.closeBar(prompt);
    prompt = this.continuationPrompt(header, prompt);
    return this.generateIds(prompt, { stopAfterBars: bars, ...opts });
  }

  async accompany(condIds, bars, { header = [], ...opts } = {}) {
    const prompt = this.accompanimentPrompt(header, this.padToBars(condIds, bars));
    return this.generateIds(prompt, { maxNewTokens: accompanimentBudget(condIds.length, bars), trimLeadingBars: false,
      banIds: [this.sp.sep, this.sp.mask], ...opts, stopAfterBars: bars });
  }

  async infill(prefixIds, suffixIds, bars, { header = [], ...opts } = {}) {
    const prompt = this.infillPrompt(header, prefixIds, suffixIds);
    const ctxBars = this.countBars(prefixIds) + this.countBars(suffixIds);
    const perBar = (prefixIds.length + suffixIds.length) / Math.max(1, ctxBars);
    return this.generateIds(prompt, { maxNewTokens: Math.trunc(Math.max(64, 2 * perBar) * bars) + 64,
      banIds: [this.sp.sep, this.sp.mask, this.bosId, this.eosId].filter((t) => t !== null).sort((a, b) => a - b),
      trimLeadingBars: false, ...opts, stopAfterBars: bars });
  }
}

// ---------------------------------------------------------------- infill_span.py

export const CONTEXT_BARS = 16;
export const MAX_SPAN_BARS = 4;

export class InfillError extends Error {
  constructor(msg) { super(msg); this.name = "InfillError"; }
}

export function contextWindow(nBars, start, bars, contextBars = CONTEXT_BARS) {
  const ctx = Math.min(contextBars, nBars);
  let b0 = start - Math.floor((ctx - bars) / 2);
  b0 = Math.max(0, Math.min(b0, nBars - ctx));
  return [b0, b0 + ctx];
}

const noteCount = (sc) => sc.tracks.reduce((s, t) => s + t.notes.length, 0);

/** infill_span.plan_score: a parsed score (drums normalized by the caller), checked against the trained shape. */
export function infillPlan(score, start, bars) {
  if (!noteCount(score)) throw new InfillError("upload has no notes");
  const edges = barEdges(score);
  const nBars = edges.length - 1;
  if (!(bars >= 1 && bars <= MAX_SPAN_BARS)) {
    throw new InfillError(`bars=${bars}: infill can regenerate 1-${MAX_SPAN_BARS} bars at a time (the span lengths it was trained on)`);
  }
  if (start < 1 || start + bars > nBars - 1) {
    const hi = nBars - bars - 1;
    const where = hi >= 1 ? `start=${start}..${hi}` : "none: the upload is too short";
    throw new InfillError(`start=${start}, bars=${bars}: infill needs at least one kept bar before and after the span; ` +
      `this upload has ${nBars} bars (valid ${where})`);
  }
  const [b0, b1] = contextWindow(nBars, start, bars);
  const p = {
    score, edges, start, bars, ctxStart: b0, ctxEnd: b1,
    spanTicks: [edges[start], edges[start + bars]],
    window: () => windowOf(score, edges[b0], edges[b1]),
    prefix: () => windowOf(score, edges[b0], edges[start]),
    suffix: () => windowOf(score, edges[start + bars], edges[b1]),
    prefixBars: start - b0,
    suffixBars: b1 - start - bars,
  };
  if (!noteCount(p.prefix()) && !noteCount(p.suffix())) throw new InfillError("the bars around the span are empty; nothing to infill from");
  return p;
}

/** infill_span.prompt_segments: (prefix ids, suffix ids), each padded to its bar count. */
export function promptSegments(gen, p) {
  return [gen.padToBars(gen.tokenizer.encode(p.prefix()), p.prefixBars), gen.padToBars(gen.tokenizer.encode(p.suffix()), p.suffixBars)];
}

/** infill_span.splice: the score with bars [start, start+bars) replaced by `answer` (a segment from the span's bar 0). */
export function splice(p, answer) {
  const [s, e] = p.spanTicks;
  const out = copyScore(p.score);
  for (const tr of out.tracks) {
    const kept = [];
    for (const n of tr.notes) {
      if (s <= n.time && n.time < e) continue;
      const m = { ...n };
      if (m.time < s && s < m.time + m.duration) m.duration = s - m.time;
      kept.push(m);
    }
    tr.notes = kept;
  }
  const key = (t) => `${!!t.is_drum}:${t.is_drum ? 0 : Math.trunc(t.program)}`;
  const gen = resample(answer, p.score.tpq);
  for (const gt of gen.tracks) {
    if (!gt.notes.length) continue;
    let dst = out.tracks.find((t) => key(t) === key(gt));
    if (!dst) {
      dst = { name: "infill", program: gt.program, is_drum: gt.is_drum, notes: [], controls: [] };
      out.tracks.push(dst);
    }
    for (const n of gt.notes) {
      const m = { ...n, time: n.time + s };
      if (m.time >= e) continue;
      m.duration = Math.max(1, Math.min(m.duration, e - m.time));
      dst.notes.push(m);
    }
    dst.notes.sort(noteCmp);
  }
  return out;
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
  return { plan: p, score: splice(p, answer), notes: noteCount(answer), ids };
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
