// attributes: the v4 attribute header (midigenai/attributes.py) and the drum-track fix every upload gets
// (tokenizer.normalize_drums), in plain JS. Same buckets, same order, same names, so a header built here is
// the token list Python builds for the same score. Scores are midi.mjs scores.

import { qpm } from "./midi.mjs";

export const INSTRUMENT_FAMILIES = [
  "Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings", "Ensemble",
  "Brass", "Reed", "Pipe", "SynthLead", "SynthPad", "SynthFX", "Ethnic",
  "Percussive", "SoundFX",
];
export const DRUMS = "Drums";
export const MAX_INST_TOKENS = 6;
const DENSITY_EDGES = [8, 24, 64];
const POLY_EDGES = [1.5, 3.0];
const RANGE_EDGES = [24, 48];
const TEMPO_EDGES = [70, 90, 110, 130, 150, 175];
export const HEADER_PREFIXES = ["Task_", "Inst_", "Density_", "Poly_", "Range_", "Tempo_", "Source_", "Genre_", "Quality_"];

export const isHeaderToken = name => HEADER_PREFIXES.some(p => name.startsWith(p));
const prefixOf = name => HEADER_PREFIXES.find(p => name.startsWith(p));

function bucket(x, edges) {
  for (let i = 0; i < edges.length; i++) if (x < edges[i]) return i;
  return edges.length;
}

const FAMILY_ORDER = new Map([...INSTRUMENT_FAMILIES, DRUMS].map((f, i) => [f, i]));

export function familyOf(program, isDrum) {
  if (isDrum) return DRUMS;
  return INSTRUMENT_FAMILIES[Math.floor(Math.max(0, Math.min(127, Math.trunc(program))) / 8)];
}

// Counter.most_common(n): by count, ties in first-seen order
function mostCommon(counts, n) {
  return [...counts.entries()].map((e, i) => [e, i]).sort((a, b) => b[0][1] - a[0][1] || a[1] - b[1]).slice(0, n).map(p => p[0]);
}

export function instrumentTokens(score) {
  const counts = new Map();
  for (const t of score.tracks) {
    if (!t.notes.length) continue;
    const f = familyOf(t.program, t.isDrum);
    counts.set(f, (counts.get(f) || 0) + t.notes.length);
  }
  const fams = mostCommon(counts, MAX_INST_TOKENS).map(([f]) => f);
  return fams.sort((a, b) => FAMILY_ORDER.get(a) - FAMILY_ORDER.get(b)).map(f => `Inst_${f}`);
}

/** Ticks in the first bar (4/4 when the file carries no time signature). */
export function ticksPerBar(score) {
  let num = 4, den = 4;
  if (score.timeSignatures.length) {
    const ts = score.timeSignatures.reduce((m, x) => (x.time < m.time ? x : m));
    num = ts.numerator; den = ts.denominator;
  }
  return Math.trunc((score.tpq * 4 * num) / den);
}

export function contentTokens(score) {
  const notes = score.tracks.flatMap(t => t.notes);
  if (!notes.length) return [];
  let start = Infinity, end = -Infinity;
  for (const n of notes) { start = Math.min(start, n.time); end = Math.max(end, n.time + n.duration); }
  const tpb = ticksPerBar(score);
  if (tpb === 0) throw new Error("division by zero: a bar of 0 ticks");
  const nBars = Math.max(1.0, (end - start) / tpb);
  const density = notes.length / nBars;
  const onsets = new Set(notes.map(n => n.time)).size;
  const poly = notes.length / Math.max(1, onsets);
  let lo = Infinity, hi = -Infinity;
  for (const t of score.tracks) if (!t.isDrum) for (const n of t.notes) { lo = Math.min(lo, n.pitch); hi = Math.max(hi, n.pitch); }
  const span = hi >= lo ? hi - lo : 0;
  return [`Density_${bucket(density, DENSITY_EDGES)}`, `Poly_${bucket(poly, POLY_EDGES)}`, `Range_${bucket(span, RANGE_EDGES)}`];
}

export const tempoBucket = bpm => bucket(bpm, TEMPO_EDGES);

/** The tempo in force for the most ticks of `score`; null without a tempo event. */
export function dominantTempo(score) {
  const tempos = [...score.tempos].sort((a, b) => a.time - b.time);
  if (!tempos.length) return null;
  let end = 0;
  for (const t of score.tracks) for (const n of t.notes) end = Math.max(end, n.time + n.duration);
  if (tempos.length === 1 || end <= 0) return qpm(tempos[0]);
  const weight = new Map();
  tempos.forEach((t, i) => {
    const start = Math.max(0, t.time);
    const stop = i + 1 < tempos.length ? tempos[i + 1].time : end;
    const q = qpm(t);
    weight.set(q, (weight.get(q) || 0) + Math.max(0, Math.min(stop, end) - start));
  });
  let best = null;
  for (const [bpm, ticks] of weight) if (!best || ticks > best[1]) best = [bpm, ticks];
  return best[1] > 0 ? best[0] : qpm(tempos[0]);
}

/** [Tempo_<bucket>], or [] without a usable tempo. An explicit `tempo` (the caller's clock) wins over the
 *  score's events. (The placeholder-source rules of the Python side only matter for corpus files.) */
export function tempoTokens(score, tempo = null) {
  if (tempo === null || tempo === undefined) {
    if (!score) return [];
    tempo = dominantTempo(score);
  }
  if (tempo === null || tempo <= 0) return [];
  return [`Tempo_${tempoBucket(tempo)}`];
}

/** The header (token names) for a document whose content is `score`; `tempo` overrides the score's own. */
export function headerForScore(score, { tempo = null } = {}) {
  return [...instrumentTokens(score), ...contentTokens(score), ...tempoTokens(score, tempo)];
}

// ---------------------------------------------------------------- instrument names, as callers say them

export const INSTRUMENT_ALIASES = {
  "acoustic piano": "Piano", "electric piano": "Piano", "keys": "Piano", "keyboard": "Piano", "rhodes": "Piano",
  "chromatic percussion": "ChromPerc", "vibraphone": "ChromPerc",
  "electric guitar": "Guitar", "acoustic guitar": "Guitar",
  "electric bass": "Bass", "bassline": "Bass", "sub": "Bass", "808": "Bass",
  "string section": "Strings", "violin": "Strings", "cello": "Strings",
  "choir": "Ensemble", "horns": "Brass", "trumpet": "Brass",
  "sax": "Reed", "saxophone": "Reed", "clarinet": "Reed", "flute": "Pipe",
  "lead": "SynthLead", "synth": "SynthLead", "pad": "SynthPad", "synth bass": "Bass",
  "fx": "SynthFX", "sound effects": "SoundFX", "percussion": "Percussive",
  "drum": "Drums", "drum kit": "Drums", "beat": "Drums",
};
const AUTO_INSTRUMENT = new Set(["", "auto", "any", "none", "original", "model"]);

const norm = name => [...name.toLowerCase()].filter(ch => /[\p{L}\p{N}]/u.test(ch)).join("");
const FAMILY_BY_NAME = new Map([...INSTRUMENT_FAMILIES, DRUMS].map(f => [norm(f), f]));
for (const [k, v] of Object.entries(INSTRUMENT_ALIASES)) FAMILY_BY_NAME.set(norm(k), v);

/** GM family for an instrument name ("bass", "synth lead", "Rhodes"), or null if it isn't one we know. */
export function resolveFamily(name) {
  if (!name) return null;
  return FAMILY_BY_NAME.get(norm(name)) ?? null;
}

/** True when the caller asked for no particular instrument. */
export const isAutoInstrument = name => AUTO_INSTRUMENT.has((name || "").trim().toLowerCase());

/** Canonical family order, stable within a family. */
export function sortHeader(names) {
  const rank = n => HEADER_PREFIXES.indexOf(prefixOf(n));
  return names.map((n, i) => [n, i]).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(p => p[0]);
}

/** `names` with its Inst_ tokens replaced by exactly `families` (deduplicated, capped at MAX_INST_TOKENS). */
export function withInstruments(names, families) {
  const unknown = families.filter(f => !FAMILY_ORDER.has(f));
  if (unknown.length) throw new Error(`not instrument families: ${unknown.join(", ")}`);
  const picked = [...new Set(families)].sort((a, b) => FAMILY_ORDER.get(a) - FAMILY_ORDER.get(b)).slice(0, MAX_INST_TOKENS);
  return sortHeader([...picked.map(f => `Inst_${f}`), ...names.filter(n => !n.startsWith("Inst_"))]);
}

// ---------------------------------------------------------------- drum tracks (tokenizer.normalize_drums)

const DRUM_HINTS_ANYWHERE = ["drum", "perc", "kick", "snare", "cymbal", "hihat", "hi-hat", "clap", "batter", "schlag"];
const DRUM_HINTS_BOUNDED = ["drm", "kit", "hat", "tom", "beat", "808", "909"];
const BOUNDED_RE = new RegExp(`(?<![0-9a-z])(${DRUM_HINTS_BOUNDED.join("|")})(?![0-9a-z])`);

/** True when a track or file name names a drum part. */
export function nameSaysDrums(name) {
  name = (name || "").toLowerCase();
  return DRUM_HINTS_ANYWHERE.some(k => name.includes(k)) || BOUNDED_RE.test(name);
}

const inDrumRange = p => p >= 35 && p < 60;
const GM_KICK = [35, 36], GM_SNARE = [38, 40], GM_HAT = [42, 44, 46];

/** Content-based drum detection: enough notes, a tiny pitch alphabet inside the GM kit, kick + snare + hat. */
export function looksLikeDrums(track, minNotes = 16) {
  const pitches = track.notes.map(n => n.pitch);
  if (pitches.length < minNotes) return false;
  const distinct = new Set(pitches);
  if (distinct.size > 8) return false;
  if (pitches.filter(inDrumRange).length / pitches.length < 0.85) return false;
  return [GM_KICK, GM_SNARE, GM_HAT].every(fam => fam.some(p => distinct.has(p)));
}

/** Promote mislabeled drum tracks to isDrum, in place. Returns how many changed. */
export function normalizeDrums(score, filenameHint = "") {
  const fname = nameSaysDrums(filenameHint);
  let changed = 0;
  for (const t of score.tracks) {
    if (t.isDrum) continue;
    if (nameSaysDrums(t.name) || (fname && score.tracks.length === 1) || looksLikeDrums(t)) {
      t.isDrum = true;
      changed++;
    }
  }
  return changed;
}
