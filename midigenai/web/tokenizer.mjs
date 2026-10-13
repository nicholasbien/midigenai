// tokenizer: MidiTok's REMI tokenizer, as midigenai's v4/v5 checkpoints use it, in plain JS.
//
// Built from the tokenizer.json saved next to a checkpoint (the MidiTok config; the vocabulary is derived from
// it in MidiTok's order), it encodes a Score (midi.mjs) to ids and decodes ids back to a Score, following
// MidiTok 3.1 (midi_tokenizer.py preprocess_score / _score_to_tokens, tokenizations/remi.py) step for step,
// rounding and tie-breaking included. Only the options midigenai's REMI configs use are implemented; a config
// that turns on anything else (rests, tempos, chords, pedals, pitch bends, BPE, ...) is refused, not guessed at.
// tests/web checks the ids and the decoded scores against MidiTok.

import { newScore, newTrack, resample, scoreEnd, mspqOf } from "./midi.mjs";

// miditok.constants.MIDI_INSTRUMENTS[*].name: the track names a decoded score carries
export const GM_NAMES = ["Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavi", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensembles 1", "String Ensembles 2", "SynthStrings 1", "SynthStrings 2", "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise, Guitar Cutting Noise", "Breath Noise, Flute Key Click", "Seashore, Rain, Thunder, Wind, Stream, Bubbles", "Bird Tweet, Dog, Horse Gallop", "Telephone Ring, Door Creaking, Door, Scratch, Wind Chime", "Helicopter, Car Sounds", "Applause, Laughing, Screaming, Punch, Heart Beat, Footstep", "Gunshot, Machine Gun, Lasergun, Explosion"];

const DEFAULT_TS = [4, 4];
const DEFAULT_TEMPO = 120;

// numpy's round (half to even), for _adjust_time_to_tpb
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// miditok.utils.np_get_closest: the nearest value of a sorted array; a tie goes to the higher one
function closest(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
  let i = lo;
  if (i === arr.length || Math.abs(v - arr[Math.max(i - 1, 0)]) < Math.abs(v - arr[Math.min(i, arr.length - 1)])) i -= 1;
  return arr[Math.max(i, 0)];
}

// numpy searchsorted(side="left")
function searchsorted(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; }
  return lo;
}

const ticksPerBeat = (den, td) => (den === 4 ? td : Math.trunc(td * (4 / den)));
const ticksPerBar = (num, den, td) => Math.trunc(ticksPerBeat(den, td) * num);
const floorDiv = (a, b) => Math.floor(a / b);

export class RemiTokenizer {
  /** `params`: a parsed tokenizer.json ({config, tokenization, ...}). */
  constructor(params) {
    const kind = params.tokenization || "MIDILike";
    if (kind !== "REMI") throw new Error(`only REMI tokenizers (v4 and later) run in the browser; this one is ${kind}`);
    const c = params.config;
    const unsupported = ["use_rests", "use_tempos", "use_chords", "use_sustain_pedals", "use_pitch_bends",
      "use_pitch_intervals", "use_key_signatures", "use_control_changes", "remove_duplicated_notes"].filter(k => c[k]);
    const ap = c.additional_params || {};
    if (ap.max_bar_embedding != null || ap.use_bar_end_tokens || ap.add_trailing_bars) unsupported.push("additional_params");
    if (!c.use_programs || !c.one_token_stream_for_programs || !c.program_changes || !c.use_time_signatures || !c.use_velocities)
      unsupported.push("programs/time signatures/velocities off");
    if (params.vocab_model || params._model) unsupported.push("a trained (BPE) vocabulary");
    if (unsupported.length) throw new Error(`tokenizer options not supported in the browser: ${unsupported.join(", ")}`);
    this.config = c;

    this.pitchRange = c.pitch_range;
    this.drumsPitchRange = c.drums_pitch_range;
    this.usePitchDrum = !!c.use_pitchdrum_tokens;
    this.programs = c.programs;
    this.programSet = new Set(c.programs);
    this.durationPrograms = new Set(c.use_note_duration_programs);
    this.defaultNoteDuration = c.default_note_duration ?? 0.5;
    // beat_res: {"0_4": 8, "4_12": 4} in insertion order
    this.beatRes = Object.entries(c.beat_res).map(([k, v]) => [k.split("_").map(Number), v]);
    this.posPerBeat = Math.max(...this.beatRes.map(([, r]) => r));
    // time signatures, in config order
    const tsr = Object.entries(c.time_signature_range).map(([d, ns]) => [Number(d), ns]);
    this.timeSigs = [];
    for (const [den, nums] of tsr) for (const n of nums) this.timeSigs.push([n, den]);
    this.timeSigKeys = new Set(this.timeSigs.map(([n, d]) => `${n}/${d}`));
    const maxDen = Math.max(...this.timeSigs.map(t => t[1]));
    this.tpbPerTs = Object.fromEntries(tsr.map(([den]) => [den, this.posPerBeat * Math.floor(maxDen / den)]));
    this.timeDivision = this.tpbPerTs[DEFAULT_TS[1]];
    // durations (beat, pos, res), MidiTok's _create_durations_tuples
    const durs = [];
    for (const [[b0, b1], res] of this.beatRes) for (let b = b0; b < b1; b++) for (let p = 0; p < res; p++) durs.push([b, p, res]);
    const lastKey = this.beatRes.reduce((m, e) => (cmpTuple(e[0], m[0]) > 0 ? e : m));
    durs.push([Math.max(...lastKey[0]), 0, lastKey[1]]);
    durs.shift();
    this.durations = durs;
    const durTicks = (d, tpb) => Math.floor(((d[0] * d[2] + d[1]) * tpb) / d[2]);
    this.tpbToTimeArray = {}; this.tpbTokensToTicks = {}; this.tpbTicksToTokens = {};
    for (const tpb of Object.values(this.tpbPerTs)) {
      this.tpbToTimeArray[tpb] = durs.map(d => durTicks(d, tpb));
      const t2t = new Map(), k2t = new Map();
      for (const d of durs) { const key = d.join("."); t2t.set(key, durTicks(d, tpb)); }
      for (const [k, v] of t2t) k2t.set(v, k);
      this.tpbTokensToTicks[tpb] = t2t; this.tpbTicksToTokens[tpb] = k2t;
    }
    // velocities: np.linspace(0, 127, n + 1, dtype=intc)[1:]
    const nv = c.num_velocities;
    this.velocities = Array.from({ length: nv }, (_, i) => Math.trunc(((i + 1) * 127) / nv));

    // vocabulary, in MidiTok's order: special tokens, then REMI's base tokens
    const vocab = [...c.special_tokens];
    vocab.push("Bar_None");
    for (let p = this.pitchRange[0]; p <= this.pitchRange[1]; p++) vocab.push(`Pitch_${p}`);
    for (const v of this.velocities) vocab.push(`Velocity_${v}`);
    for (const d of durs) vocab.push(`Duration_${d.join(".")}`);
    const maxBeats = Math.max(...this.timeSigs.map(t => t[0]));
    for (let i = 0; i < this.posPerBeat * maxBeats; i++) vocab.push(`Position_${i}`);
    if (this.usePitchDrum) for (let p = this.drumsPitchRange[0]; p <= this.drumsPitchRange[1]; p++) vocab.push(`PitchDrum_${p}`);
    for (const p of this.programs) vocab.push(`Program_${p}`);
    for (const [n, d] of this.timeSigs) vocab.push(`TimeSig_${n}/${d}`);
    this.tokens = [];
    this.vocab = new Map();
    for (const t of vocab) if (!this.vocab.has(t)) { this.vocab.set(t, this.tokens.length); this.tokens.push(t); }
  }

  get vocabSize() { return this.tokens.length; }

  id(token) {
    const i = this.vocab.get(token);
    if (i === undefined) throw new Error(`token ${token} is not in the vocabulary`);
    return i;
  }

  // ------------------------------------------------------------ encode

  /** MidiTok's preprocess_score: resample to the tokenizer's grid, snap time signatures to bars, merge tracks
   *  of one program, drop what the vocabulary can't hold, quantize velocities, onsets and durations. */
  preprocess(score) {
    let sigs = score.timeSignatures.filter(ts => this.timeSigKeys.has(`${ts.numerator}/${ts.denominator}`)).map(ts => ({ ...ts }));
    if (!sigs.length || sigs[0].time !== 0) sigs.unshift({ time: 0, numerator: DEFAULT_TS[0], denominator: DEFAULT_TS[1] });
    const maxDen = Math.max(...sigs.map(ts => ts.denominator));
    const newTpq = Math.trunc((this.posPerBeat * maxDen) / 4);
    let s;
    if (score.tpq !== newTpq) {
      const f = newTpq / score.tpq;
      sigs = sigs.map(ts => ({ ...ts, time: Math.trunc(ts.time * f) }));
      s = resample(score, newTpq, 1);
    } else {
      s = resample(score, score.tpq, 0);      // a copy
    }
    s.timeSignatures = sigs;
    if (s.timeSignatures.length) this.preprocessTimeSignatures(s.timeSignatures, s.tpq);
    const tpb = this.scoreTicksPerBeat(s);
    const factors = new Set(s.timeSignatures.map(ts => ts.denominator)).size > 1 ? this.resamplingFactors(s) : null;

    mergeSameProgramTracks(s.tracks);
    for (let t = s.tracks.length - 1; t >= 0; t--) {
      const tr = s.tracks[t];
      const program = tr.isDrum ? -1 : tr.program;
      if (!tr.notes.length || !this.programSet.has(program)) { s.tracks.splice(t, 1); continue; }
      this.preprocessNotes(tr, factors, tpb);
      if (!tr.notes.length) s.tracks.splice(t, 1);
    }
    return s;
  }

  preprocessTimeSignatures(sigs, td) {
    const eq = (a, b) => a.numerator === b.numerator && a.denominator === b.denominator;
    let i = 0, tpbar = ticksPerBar(sigs[0].numerator, sigs[0].denominator, td), prev = 0;
    while (i < sigs.length) {
      if (i >= 1 && eq(sigs[i], sigs[i - 1])) { sigs.splice(i, 1); continue; }
      if (i >= 1 && sigs[i].time < sigs[i - 1].time) sigs[i].time = sigs[i - 1].time;
      const off = sigs[i].time - prev, bar = floorDiv(off, tpbar), rest = off - bar * tpbar;
      if (rest > 0) sigs[i].time = prev + (bar + 1) * tpbar;
      tpbar = ticksPerBar(sigs[i].numerator, sigs[i].denominator, td);
      prev = sigs[i].time;
      if (i >= 1 && sigs[i].time === sigs[i - 1].time) {
        sigs.splice(i - 1, 1);
        if (i >= 2 && eq(sigs[i - 1], sigs[i - 2])) {
          sigs.splice(i - 1, 1);
          tpbar = ticksPerBar(sigs[i - 2].numerator, sigs[i - 2].denominator, td);
          prev = sigs[i - 2].time;
          i -= 1;
        }
        continue;
      }
      i += 1;
    }
    if (sigs.length) { if (sigs[0].time !== 0) sigs.unshift({ time: 0, numerator: DEFAULT_TS[0], denominator: DEFAULT_TS[1] }); }
    else sigs.push({ time: 0, numerator: DEFAULT_TS[0], denominator: DEFAULT_TS[1] });
  }

  // get_score_ticks_per_beat: [[section end tick, ticks per beat], ...]
  scoreTicksPerBeat(s) {
    const ts = s.timeSignatures, out = [];
    for (let i = 0; i < ts.length - 1; i++) out.push([ts[i + 1].time, ticksPerBeat(ts[i].denominator, s.tpq)]);
    out.push([scoreEnd(s) + 1, ticksPerBeat(ts[ts.length - 1].denominator, s.tpq)]);
    for (let i = out.length - 1; i > 0; i--) if (out[i][1] === out[i - 1][1]) { out[i - 1][0] = out[i][0]; out.splice(i, 1); }
    return out;
  }

  resamplingFactors(s) {
    const ts = s.timeSignatures, step = den => Math.floor(s.tpq / (this.posPerBeat * (den / 4)));
    const out = [];
    for (let i = 0; i < ts.length - 1; i++) out.push([ts[i + 1].time, step(ts[i].denominator)]);
    out.push([scoreEnd(s) + 1, step(ts[ts.length - 1].denominator)]);
    for (let i = out.length - 1; i > 0; i--) if (out[i][1] === out[i - 1][1]) { out[i - 1][0] = out[i][0]; out.splice(i, 1); }
    return out;
  }

  preprocessNotes(tr, factors, tpb) {
    const range = tr.isDrum && this.usePitchDrum ? this.drumsPitchRange : this.pitchRange;
    let notes = tr.notes.filter(n => n.pitch >= range[0] && n.pitch <= range[1]);
    if (!notes.length) { tr.notes = []; return; }
    notes = notes.map(n => ({ ...n, velocity: closest(this.velocities, n.velocity) }));
    const program = tr.isDrum ? -1 : tr.program;
    if (factors) {
      const times = notes.map(n => n.time);
      const ends = factors.map(([end]) => searchsorted(times, end));
      ends[ends.length - 1] = times.length;
      let first = 0, gridStart = 0;
      factors.forEach(([end, step], k) => {
        const last = ends[k];
        if (step !== 1) for (let i = first; i < last; i++) notes[i].time = roundHalfEven((notes[i].time - gridStart) / step) * step + gridStart;
        first = last; gridStart = end;
      });
    }
    if (this.durationPrograms.has(program)) {
      let first = 0;
      const lastTime = notes[notes.length - 1].time;
      for (let k = 0; k < tpb.length; k++) {
        const [lastTick, beat] = tpb[k];
        let last;
        if (k + 1 === tpb.length || lastTick > lastTime) last = notes.length;
        else {
          let j = first;
          while (j < notes.length && !(notes[j].time >= lastTick)) j++;
          last = j < notes.length ? j : first;           // np.argmax of an all-False mask is 0
        }
        const arr = this.tpbToTimeArray[beat];
        if (!arr) throw new Error(`no duration grid for ${beat} ticks per beat`);
        for (let i = first; i < last; i++) notes[i].duration = closest(arr, notes[i].duration);
        if (last === notes.length) break;
        first = last;
      }
    }
    tr.notes = notes;
  }

  /** Score -> token strings (MidiTok's encode on a symusic Score, no BPE). */
  encodeTokens(score) {
    const s = this.preprocess(score);
    const events = s.timeSignatures.map(ts => ({ type: "TimeSig", value: `${ts.numerator}/${ts.denominator}`, time: ts.time, program: null }));
    const tpb = this.scoreTicksPerBeat(s);
    for (const tr of s.tracks) {
      const program = tr.isDrum ? -1 : tr.program;
      const useDur = this.durationPrograms.has(program);
      const pitchType = this.usePitchDrum && tr.isDrum ? "PitchDrum" : "Pitch";
      for (const n of tr.notes) {
        events.push({ type: pitchType, value: n.pitch, time: n.time, program, end: n.time + n.duration });
        events.push({ type: "Velocity", value: n.velocity, time: n.time, program });
        if (useDur) {
          let k = 0;
          while (n.time >= tpb[k][0]) k++;
          const tok = this.tpbTicksToTokens[tpb[k][1]].get(n.duration);
          if (tok === undefined) throw new Error(`no Duration token for ${n.duration} ticks`);
          events.push({ type: "Duration", value: tok, time: n.time, program });
        }
      }
    }
    const sorted = events.map((e, i) => [e, i]).sort((a, b) => a[0].time - b[0].time || a[1] - b[1]).map(p => p[0]);
    // program changes
    const withPc = [];
    let prevProgram = null;
    for (const e of sorted) {
      if (e.program !== null && e.program !== prevProgram && e.type !== "TimeSig") {
        prevProgram = e.program;
        withPc.push({ type: "Program", value: e.program, time: e.time, program: null });
      }
      withPc.push(e);
    }
    return this.addTimeEvents(withPc, s.tpq).map(e => `${e.type}_${e.value}`);
  }

  addTimeEvents(events, td) {
    const out = [];
    let currentBar = -1, barAtLastTs = 0, prevTick = -1, tickAtLastTs = 0, tickAtBar = 0;
    let cur = DEFAULT_TS;
    const firstTs = events.find(e => e.type === "TimeSig");
    if (firstTs) cur = firstTs.value.split("/").map(Number);
    let tpbar = ticksPerBar(cur[0], cur[1], td), tppos = Math.floor(ticksPerBeat(cur[1], td) / this.posPerBeat);
    for (const e of events) {
      if (e.time !== prevTick) {
        const n = barAtLastTs + floorDiv(e.time - tickAtLastTs, tpbar) - currentBar;
        for (let i = 0; i < n; i++) {
          currentBar += 1;
          tickAtBar = tickAtLastTs + (currentBar - barAtLastTs) * tpbar;
          out.push({ type: "Bar", value: "None" });
          if (!(e.type === "TimeSig" && i + 1 === n)) out.push({ type: "TimeSig", value: `${cur[0]}/${cur[1]}` });
        }
        if (e.type !== "TimeSig") out.push({ type: "Position", value: floorDiv(e.time - tickAtBar, tppos) });
        prevTick = e.time;
      }
      if (e.type === "TimeSig") {
        barAtLastTs += floorDiv(e.time - tickAtLastTs, tpbar);
        tickAtLastTs = e.time;
        cur = e.value.split("/").map(Number);
        tpbar = ticksPerBar(cur[0], cur[1], td);
        tppos = Math.floor(ticksPerBeat(cur[1], td) / this.posPerBeat);
        prevTick -= 1;
      }
      out.push(e);
    }
    return out;
  }

  /** Score -> ids, as `tokenizer(score).ids`. */
  encode(score) { return this.encodeTokens(score).map(t => this.id(t)); }

  // ------------------------------------------------------------ decode

  /** ids -> Score, as `tokenizer.decode(ids)`: special and header tokens are skipped (a Tempo_ header token
   *  becomes a tempo event of that many BPM, as in MidiTok; callers stamp the real tempo on afterwards). */
  decode(ids) {
    const seq = ids.map(i => {
      const t = this.tokens[i];
      if (t === undefined) throw new Error(`id ${i} is not in the vocabulary`);
      return t;
    });
    const split = t => { const k = t.indexOf("_"); return [t.slice(0, k), t.slice(k + 1)]; };
    const score = newScore(this.timeDivision);
    const tracks = new Map();
    const tempos = [], sigs = [];
    for (const t of seq) {
      const [type, val] = split(t);
      if (type === "TimeSig") { const [n, d] = val.split("/").map(Number); sigs.push({ time: 0, numerator: n, denominator: d }); break; }
      if (["Pitch", "PitchDrum", "Velocity", "Duration", "PitchBend", "ControlChange", "Pedal"].includes(type)) break;
    }
    if (!sigs.length) sigs.push({ time: 0, numerator: DEFAULT_TS[0], denominator: DEFAULT_TS[1] });
    let curTs = sigs[sigs.length - 1];
    const td = score.tpq;
    let tpbar = ticksPerBar(curTs.numerator, curTs.denominator, td);
    let tpb = this.tpbPerTs[curTs.denominator], tppos = Math.floor(tpb / this.posPerBeat);
    let curTick = 0, tickAtBar = 0, currentBar = -1, program = 0;
    let useDur = this.durationPrograms.has(program);
    for (let ti = 0; ti < seq.length; ti++) {
      const tok = seq[ti];
      const [type, val] = split(tok);
      if (tok === "Bar_None") {
        currentBar += 1;
        if (currentBar > 0) curTick = tickAtBar + tpbar;
        tickAtBar = curTick;
      } else if (type === "Position") {
        if (currentBar === -1) currentBar = 0;
        curTick = tickAtBar + Number(val) * tppos;
      } else if (type === "Pitch" || type === "PitchDrum") {
        const pitch = Number(val);
        if (pitch < this.pitchRange[0] || pitch > this.pitchRange[1]) continue;
        if (ti + 1 >= seq.length) continue;
        const [vt, vel] = split(seq[ti + 1]);
        let dt = "Duration", dur;
        if (useDur) {
          if (ti + 2 >= seq.length) continue;
          [dt, dur] = split(seq[ti + 2]);
        } else dur = Math.trunc(this.defaultNoteDuration * tpb);
        if (vt === "Velocity" && dt === "Duration") {
          if (typeof dur === "string") {
            dur = this.tpbTokensToTicks[tpb].get(dur);
            if (dur === undefined) throw new Error(`bad Duration token ${seq[ti + 2]}`);
          }
          if (!tracks.has(program))
            tracks.set(program, newTrack(program === -1 ? "Drums" : GM_NAMES[program], program === -1 ? 0 : program, program === -1));
          tracks.get(program).notes.push({ time: curTick, duration: dur, pitch, velocity: Number(vel) });
        }
      } else if (type === "Program") {
        program = Number(val);
        useDur = this.durationPrograms.has(program);
      } else if (type === "Tempo") {
        tempos.push({ time: curTick, mspq: mspqOf(Number(val)) });
      } else if (type === "TimeSig") {
        const [n, d] = val.split("/").map(Number);
        if (n !== curTs.numerator || d !== curTs.denominator) {
          curTs = { time: curTick, numerator: n, denominator: d };
          sigs.push(curTs);
          tpbar = ticksPerBar(n, d, td);
          tpb = this.tpbPerTs[d];
          tppos = Math.floor(tpb / this.posPerBeat);
        }
      }
    }
    // a time signature change re-anchors bar counting at the current bar: Position is relative to it, and the
    // next Bar adds the new bar length, so nothing else needs tracking here
    score.tracks = [...tracks.values()];
    score.tempos = tempos;
    score.timeSignatures = sigs;
    if (!score.tempos.length || score.tempos[0].time !== 0) score.tempos.unshift({ time: 0, mspq: mspqOf(DEFAULT_TEMPO) });
    if (!score.timeSignatures.length || score.timeSignatures[0].time !== 0) score.timeSignatures.unshift({ time: 0, numerator: 4, denominator: 4 });
    return score;
  }
}

function cmpTuple(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

const noteKey = (a, b) => a.time - b.time || a.duration - b.duration || a.pitch - b.pitch || a.velocity - b.velocity;

// miditok.utils.merge_same_program_tracks: tracks sharing a program (drums: -1) become the first of them,
// notes concatenated and sorted, names joined with " / "
function mergeSameProgramTracks(tracks) {
  const progs = tracks.map(t => (t.isDrum ? -1 : t.program));
  const groups = [...new Set(progs)].sort((a, b) => a - b).map(p => progs.flatMap((q, i) => (q === p ? [i] : []))).filter(g => g.length >= 2);
  if (!groups.length) return;
  const drop = [];
  for (const g of [...groups].reverse()) {
    const ts = g.map(i => tracks[i]);
    const first = ts[0];
    first.name += ts.slice(1).map(t => " / " + t.name).join("");
    first.notes = ts.flatMap(t => t.notes).sort(noteKey);
    first.pedals = ts.flatMap(t => t.pedals).sort((a, b) => a.time - b.time || a.duration - b.duration);
    first.controls = ts.flatMap(t => t.controls).map((c, i) => [c, i]).sort((a, b) => a[0].time - b[0].time || a[1] - b[1]).map(p => p[0]);
    first.pitchBends = ts.flatMap(t => t.pitchBends).sort((a, b) => a.time - b.time || a.value - b.value);
    tracks[g[0]] = first;
    drop.push(...g.slice(1));
  }
  drop.sort((a, b) => b - a);
  for (const i of drop) tracks.splice(i, 1);
}

/** A tokenizer from a tokenizer.json's text or parsed object. */
export function loadTokenizer(json) {
  return new RemiTokenizer(typeof json === "string" ? JSON.parse(json) : json);
}
