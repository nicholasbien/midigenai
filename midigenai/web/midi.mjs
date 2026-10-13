// midi: the part of symusic midigenai uses, in plain JS. Reading and writing a MIDI file, and the score
// operations the tokenizer and the tasks call (resample, clip, shift, end, downbeats).
//
// It follows symusic 0.6.0 (src/io/midi.cpp, src/score.cpp, src/conversion.cpp) closely enough that the
// same file gives the same Score, and a Score writes the same bytes: the tokenizer sees exactly the notes
// Python sees, and a track index means the same track on both sides. tests/web checks it against symusic.
//
// A Score is plain data, in ticks:
//   {tpq, tracks, tempos: [{time, mspq}], timeSignatures: [{time, numerator, denominator}],
//    keySignatures: [{time, key, tonality}], markers: [{time, text}]}
// and a track {name, program, isDrum, notes: [{time, duration, pitch, velocity}], controls: [{time, number,
// value}], pitchBends: [{time, value}], pedals: [{time, duration}], lyrics: [{time, text}]}.

export function newTrack(name = "", program = 0, isDrum = false) {
  return { name, program, isDrum, notes: [], controls: [], pitchBends: [], pedals: [], lyrics: [] };
}

export function newScore(tpq = 480) {
  return { tpq, tracks: [], tempos: [], timeSignatures: [], keySignatures: [], markers: [] };
}

export const qpm = tempo => 60000000 / tempo.mspq;
// symusic's Tempo(time, qpm): mspq = int32(round(6e7 / qpm)); a value past int32 (qpm 0, from a Tempo_0 header
// token decoded as a tempo) comes out as INT32_MIN, as the C++ cast does on x86
export const mspqOf = bpm => {
  const x = Math.round(60000000 / bpm);
  return Number.isFinite(x) && x <= 2147483647 && x >= -2147483648 ? x : -2147483648;
};

const copyEvents = xs => xs.map(x => ({ ...x }));

export function copyTrack(t) {
  return {
    name: t.name, program: t.program, isDrum: t.isDrum,
    notes: copyEvents(t.notes), controls: copyEvents(t.controls), pitchBends: copyEvents(t.pitchBends),
    pedals: copyEvents(t.pedals), lyrics: copyEvents(t.lyrics),
  };
}

export function copyScore(s) {
  return {
    tpq: s.tpq, tracks: s.tracks.map(copyTrack), tempos: copyEvents(s.tempos),
    timeSignatures: copyEvents(s.timeSignatures), keySignatures: copyEvents(s.keySignatures),
    markers: copyEvents(s.markers),
  };
}

const stableSortByTime = xs => xs.map((x, i) => [x, i]).sort((a, b) => a[0].time - b[0].time || a[1] - b[1]).map(p => p[0]);

// ---------------------------------------------------------------- reading

// utf8::replace_invalid: invalid sequences become U+FFFD
const utf8 = bytes => new TextDecoder("utf-8", { fatal: false }).decode(bytes);

function readVarLen(b, pos) {
  let n = 0, c;
  do {
    if (pos.i >= b.length) throw new Error("MIDI: unexpected end of data in a variable-length number");
    c = b[pos.i++];
    n = (n * 128) + (c & 0x7f);
  } while (c & 0x80);
  return n;
}

const MSG_LEN = (() => {      // minimidi's message lengths by status byte (status included)
  const t = new Array(256).fill(0);
  const set = (s, n, chan) => { if (chan) for (let c = 0; c < 16; c++) t[s | c] = n; else t[s] = n; };
  set(0x80, 3, 1); set(0x90, 3, 1); set(0xa0, 3, 1); set(0xb0, 3, 1); set(0xc0, 2, 1); set(0xd0, 2, 1); set(0xe0, 3, 1);
  for (const [s, n] of [[0xf1, 2], [0xf2, 3], [0xf3, 2], [0xf6, 1], [0xf7, 1], [0xf8, 1], [0xfa, 1], [0xfb, 1], [0xfc, 1], [0xfe, 1]]) set(s, n, 0);
  return t;
})();

/** One MTrk's messages as [{time, status, data (Uint8Array, without the status byte)}], as minimidi reads them. */
function trackMessages(b) {
  const out = [], pos = { i: 0 };
  let tick = 0, prevStatus = 0, prevLen = 0;
  while (pos.i < b.length) {
    tick += readVarLen(b, pos);
    const s = b[pos.i];
    if (s === 0xf0) {
      const start = pos.i;
      pos.i += 1;
      const n = readVarLen(b, pos);
      prevLen = n + (pos.i - start);
      prevStatus = s;
      if (start + prevLen > b.length) throw new Error("MIDI: unexpected end of data in a SysEx event");
      pos.i = start + prevLen;
      continue;
    }
    if (s === 0xff) {
      const start = pos.i;
      pos.i += 2;
      const n = readVarLen(b, pos);
      const len = n + (pos.i - start);
      if (start + len > b.length) throw new Error("MIDI: unexpected end of data in a meta event");
      if (b[start + 1] === 0x2f) break;                       // end of track: the rest is ignored
      out.push({ time: tick, status: 0xff, meta: b[start + 1], data: b.subarray(pos.i, start + len) });
      pos.i = start + len;
      continue;
    }
    if (s < 0x80) {                                           // running status
      if (!prevLen) throw new Error(`MIDI: unexpected running status (0x${s.toString(16)})`);
      if (pos.i + prevLen - 1 > b.length) throw new Error("MIDI: unexpected end of data in running status");
      out.push({ time: tick, status: prevStatus, data: b.subarray(pos.i, pos.i + prevLen - 1) });
      pos.i += prevLen - 1;
      continue;
    }
    const len = MSG_LEN[s];
    if (!len) throw new Error(`MIDI: unknown status byte 0x${s.toString(16)}`);
    prevStatus = s; prevLen = len;
    if (pos.i + len > b.length) throw new Error("MIDI: unexpected end of data in an event");
    out.push({ time: tick, status: s, data: b.subarray(pos.i + 1, pos.i + len) });
    pos.i += len;
  }
  return out;
}

/** Parse MIDI bytes the way symusic's Score.from_midi does. */
export function parseMidi(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const str = (i, n) => String.fromCharCode(...b.subarray(i, i + n));
  const u32 = i => ((b[i] << 24) >>> 0) + (b[i + 1] << 16) + (b[i + 2] << 8) + b[i + 3];
  const u16 = i => (b[i] << 8) | b[i + 1];
  if (b.length < 14) throw new Error("not a MIDI file: too short");
  if (str(0, 4) !== "MThd") throw new Error("not a MIDI file: no MThd header");
  if (u32(4) !== 6) throw new Error("not a MIDI file: the header chunk is not 6 bytes");
  const nTracks = u16(10), division = u16(12);
  const score = newScore(division & 0x7fff);
  let i = 14;
  for (let k = 0; k < nTracks && i + 8 <= b.length; ) {
    const id = str(i, 4), len = u32(i + 4);
    const body = b.subarray(i + 8, Math.min(b.length, i + 8 + len));
    i += 8 + len;
    if (id !== "MTrk") continue;
    k++;
    readTrack(trackMessages(body), score);
  }
  for (const key of ["timeSignatures", "keySignatures", "tempos", "markers"]) score[key] = stableSortByTime(score[key]);
  return score;
}

// symusic's TrackManager for one MTrk: notes split by (channel, program at note-on), events of a channel kept
// in a "straggler" until a note (or lyric) creates the real track; tracks come out ordered by (channel, program).
function readTrack(msgs, score) {
  const formal = new Map();                      // key channel*256+program -> track
  const stragglers = Array.from({ length: 16 }, () => newTrack());
  const program = new Array(16).fill(0);
  const open = new Map();                        // channel*128+pitch -> FIFO of open notes
  const lastPedalOn = new Array(16).fill(0); lastPedalOn[0] = -1;      // std::array<unit,16>{-1}: only [0] is -1
  let name = "";
  const get = (ch, create) => {
    const key = ch * 256 + program[ch];
    if (formal.has(key)) return formal.get(key);
    if (!create) return stragglers[ch];
    const t = stragglers[ch];
    t.program = program[ch]; t.isDrum = ch === 9;
    formal.set(key, t);
    stragglers[ch] = newTrack();
    return t;
  };
  for (const m of msgs) {
    const type = m.status & 0xf0, ch = m.status & 0x0f, d = m.data;
    if (m.status === 0xff) {
      if (m.meta === 0x03) name = utf8(d);
      else if (m.meta === 0x58) score.timeSignatures.push({ time: m.time, numerator: d[0], denominator: 2 ** d[1] });
      else if (m.meta === 0x51) score.tempos.push({ time: m.time, mspq: (d[0] << 16) | (d[1] << 8) | d[2] });
      else if (m.meta === 0x59) score.keySignatures.push({ time: m.time, key: (d[0] << 24) >> 24, tonality: d[1] });
      else if (m.meta === 0x05) {
        const text = utf8(d);
        if (text) get(15, true).lyrics.push({ time: m.time, text });      // a meta's "channel" is 0xFF & 0x0F
      } else if (m.meta === 0x06) {
        const text = utf8(d);
        if (text) score.markers.push({ time: m.time, text });
      }
      continue;
    }
    if (m.status >= 0xf0) continue;
    if ((type === 0x80 || type === 0x90 || type === 0xb0 || type === 0xc0) && d.some(x => x > 0x7f)) throw new Error(`MIDI: invalid data byte ${Math.max(...d)} in a 0x${m.status.toString(16)} message`);
    if (type === 0x90 && d[1] !== 0) {
      const notes = get(ch, true).notes;
      const note = { time: m.time, duration: -1, pitch: d[0], velocity: d[1] };
      notes.push(note);
      const key = ch * 128 + d[0];
      if (!open.has(key)) open.set(key, []);
      open.get(key).push(note);
    } else if (type === 0x80 || type === 0x90) {
      const q = open.get(ch * 128 + d[0]);
      if (q && q.length) { const n = q.shift(); n.duration = m.time - n.time; }
    } else if (type === 0xc0) {
      program[ch] = d[0];
    } else if (type === 0xb0) {
      const t = get(ch, false);
      t.controls.push({ time: m.time, number: d[0], value: d[1] });
      if (d[0] === 64) {
        if (d[1] >= 64) {
          if (lastPedalOn[ch] < 0) lastPedalOn[ch] = m.time;
        } else if (lastPedalOn[ch] >= 0) {
          t.pedals.push({ time: lastPedalOn[ch], duration: m.time - lastPedalOn[ch] });
          lastPedalOn[ch] = -1;
        }
      }
    } else if (type === 0xe0) {
      const value = (d[0] | (d[1] << 7)) - 8192;
      if (value < -8192 || value > 8191) throw new Error(`MIDI: invalid pitch bend ${value}`);
      get(ch, false).pitchBends.push({ time: m.time, value });
    }
  }
  for (const key of [...formal.keys()].sort((a, b) => a - b)) {
    const t = formal.get(key);
    if (trackEmpty(t)) continue;
    t.notes = t.notes.filter(n => n.duration >= 0);      // never closed
    t.name = name;
    score.tracks.push(t);
  }
}

export const trackEmpty = t => !t.notes.length && !t.controls.length && !t.pitchBends.length && !t.pedals.length && !t.lyrics.length;

// ---------------------------------------------------------------- writing

// minimidi's write_variable_length: at most 4 bytes (a delta past 2^28 keeps only what fits)
function writeVarLen(out, n) {
  const k = n < 0x80 ? 1 : n < 0x4000 ? 2 : n < 0x200000 ? 3 : 4;
  for (let i = 0; i < k - 1; i++) out.push(((n >>> (7 * (k - i - 1))) | 0x80) & 0xff);
  out.push(n & 0x7f);
}

const textBytes = s => Array.from(new TextEncoder().encode(s));
const VALID_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];

// a stable merge of two time-sorted runs, the first run winning ties (gfx::timmerge)
function merge(a, b) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) out.push(b[j].time < a[i].time ? b[j++] : a[i++]);
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
}

/** MIDI bytes for a Score, laid out as symusic's dump_midi lays them out (format 1, the meta events in the
 *  first track, one channel per track, running status). */
export function writeMidi(score) {
  const metaMsg = (time, type, data) => ({ time, status: 0xff, data: [type, ...lenPrefixed(data)] });
  const lenPrefixed = data => { const v = []; writeVarLen(v, data.length); return [...v, ...data]; };
  let init = [];
  for (const ts of score.timeSignatures)
    init.push(metaMsg(ts.time, 0x58, [ts.numerator, Math.round(Math.log2(ts.denominator)), 24, 8]));
  for (const ks of score.keySignatures) init.push(metaMsg(ks.time, 0x59, [ks.key & 0xff, ks.tonality]));
  for (const t of score.tempos) init.push(metaMsg(t.time, 0x51, [(t.mspq >> 16) & 0xff, (t.mspq >> 8) & 0xff, t.mspq & 0xff]));
  for (const m of score.markers) init.push(metaMsg(m.time, 0x06, textBytes(m.text)));

  const tracks = [];
  let idx = 0;
  for (const track of score.tracks) {
    let msgs = init; init = [];
    const ch = track.isDrum ? 9 : VALID_CHANNELS[idx % 15];
    if (track.name) msgs.push(metaMsg(0, 0x03, textBytes(track.name)));
    msgs.push({ time: 0, status: 0xc0 | ch, data: [track.program & 0x7f] });
    for (const c of track.controls) msgs.push({ time: c.time, status: 0xb0 | ch, data: [c.number, c.value] });
    for (const p of track.pitchBends) {
      const v = p.value + 8192;
      msgs.push({ time: p.time, status: 0xe0 | ch, data: [v & 0x7f, (v >> 7) & 0x7f] });
    }
    for (const l of track.lyrics) msgs.push(metaMsg(l.time, 0x05, textBytes(l.text)));
    msgs = stableSortByTime(msgs);
    let notes = track.notes;
    if (notes.some((n, k) => k > 0 && n.time < notes[k - 1].time)) notes = stableSortByTime(notes);
    const first = [], second = [];
    for (const n of notes) {
      const on = { time: n.time, status: 0x90 | ch, data: [n.pitch, n.velocity] };
      if (n.duration > 0) {
        first.push({ time: n.time + n.duration, status: 0x80 | ch, data: [n.pitch, n.velocity] });
        second.push(on);
      } else {
        first.push(on);
        second.push({ time: n.time, status: 0x80 | ch, data: [n.pitch, n.velocity] });
      }
    }
    const offs = stableSortByTime(first);
    tracks.push(merge(msgs, merge(offs, second)));
    idx += 1;
  }
  if (init.length) tracks.push(init);              // a score with no tracks: its meta events, unsorted, as symusic

  const out = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1, (tracks.length >> 8) & 0xff, tracks.length & 0xff,
    (score.tpq >> 8) & 0x7f, score.tpq & 0xff];
  for (const msgs of tracks) {
    const start = out.length;
    out.push(0x4d, 0x54, 0x72, 0x6b, 0, 0, 0, 0);
    let prevTime = 0, prevStatus = 0;
    for (const m of msgs) {
      writeVarLen(out, (m.time - prevTime) >>> 0);    // uint32, as minimidi (only an unsorted meta-only track wraps)
      prevTime = m.time;
      if (m.status >= 0xf0 || m.status !== prevStatus) out.push(m.status);
      for (const x of m.data) out.push(x);
      prevStatus = m.status;
    }
    out.push(1, 0xff, 0x2f, 0);
    const n = out.length - start - 8;
    out[start + 4] = (n >>> 24) & 0xff; out[start + 5] = (n >>> 16) & 0xff; out[start + 6] = (n >>> 8) & 0xff; out[start + 7] = n & 0xff;
  }
  return Uint8Array.from(out);
}

// ---------------------------------------------------------------- score operations

const maxOf = (xs, f) => xs.reduce((m, x) => Math.max(m, f(x)), -Infinity);

export function trackEnd(t) {
  if (trackEmpty(t)) return 0;
  return Math.max(maxOf(t.notes, n => n.time + n.duration), maxOf(t.controls, c => c.time),
    maxOf(t.pitchBends, p => p.time), maxOf(t.pedals, p => p.time + p.duration), maxOf(t.lyrics, l => l.time));
}

/** Score.end(): the last tick anything happens on (note ends, controls, meta events). 0 for an empty score. */
export function scoreEnd(s) {
  if (!s.tracks.length && !s.timeSignatures.length && !s.keySignatures.length && !s.tempos.length && !s.markers.length) return 0;
  return Math.max(maxOf(s.tracks, trackEnd), maxOf(s.timeSignatures, x => x.time), maxOf(s.keySignatures, x => x.time),
    maxOf(s.tempos, x => x.time), maxOf(s.markers, x => x.time));
}

// std::round: halves away from zero
const roundHalfAway = x => (x < 0 ? -Math.round(-x) : Math.round(x));

/** Score.resample(tpq, min_dur): every time scaled by tpq/old and rounded; durations scaled on their own. */
export function resample(s, tpq, minDur = 0) {
  const rate = tpq / s.tpq;
  const conv = t => roundHalfAway(rate * t);
  const time = xs => xs.map(x => ({ ...x, time: conv(x.time) }));
  const dur = xs => xs.map(x => ({ ...x, time: conv(x.time), duration: Math.max(conv(x.duration), minDur) }));
  return {
    tpq, timeSignatures: time(s.timeSignatures), keySignatures: time(s.keySignatures), tempos: time(s.tempos),
    markers: time(s.markers),
    tracks: s.tracks.map(t => ({
      name: t.name, program: t.program, isDrum: t.isDrum, notes: dur(t.notes), pedals: dur(t.pedals),
      pitchBends: time(t.pitchBends), controls: time(t.controls), lyrics: time(t.lyrics),
    })),
  };
}

// ops::clip_with_sentinel_inplace: the event in force at `start` is restated at `start`
function clipWithSentinel(xs, start, end) {
  if (!xs.length) return [];
  let sentinel = null;
  const kept = [];
  for (const x of xs) {
    if (x.time <= start) { if (!sentinel || sentinel.time < x.time) sentinel = x; }
    else if (x.time < end) kept.push({ ...x });
  }
  return sentinel ? [{ ...sentinel, time: start }, ...kept] : kept;
}

/** Score.clip(start, end, clip_end): events starting in [start, end); with clipEnd, notes must also END by
 *  `end` (symusic drops a note that rings past the end, it does not shorten it). */
export function clip(s, start, end, clipEnd = false) {
  const inside = x => x.time >= start && x.time < end;
  const withDur = xs => xs.filter(x => (clipEnd ? x.time >= start && x.time + x.duration <= end : inside(x))).map(x => ({ ...x }));
  const plain = xs => xs.filter(inside).map(x => ({ ...x }));
  return {
    tpq: s.tpq,
    tracks: s.tracks.map(t => ({
      name: t.name, program: t.program, isDrum: t.isDrum, notes: withDur(t.notes), controls: plain(t.controls),
      pitchBends: plain(t.pitchBends), pedals: withDur(t.pedals), lyrics: plain(t.lyrics),
    })),
    timeSignatures: clipWithSentinel(s.timeSignatures, start, end),
    keySignatures: clipWithSentinel(s.keySignatures, start, end),
    tempos: clipWithSentinel(s.tempos, start, end),
    markers: plain(s.markers),
  };
}

/** Score.shift_time(offset). Lyrics stay where they are, as in symusic. */
export function shiftTime(s, offset) {
  const sh = xs => xs.map(x => ({ ...x, time: x.time + offset }));
  return {
    tpq: s.tpq,
    tracks: s.tracks.map(t => ({
      name: t.name, program: t.program, isDrum: t.isDrum, notes: sh(t.notes), controls: sh(t.controls),
      pitchBends: sh(t.pitchBends), pedals: sh(t.pedals), lyrics: sh(t.lyrics),
    })),
    timeSignatures: sh(s.timeSignatures), keySignatures: sh(s.keySignatures), tempos: sh(s.tempos), markers: sh(s.markers),
  };
}

// ---------------------------------------------------------------- downbeats (symusic's beat timeline, in float32)

const f32 = Math.fround;
const TOL = f32(1e-4);
const close = (a, b) => Math.abs(f32(a - b)) <= TOL;
const geOrClose = (a, b) => a > b || close(a, b);
const beatStep = ts => {
  if (!ts || ts.denominator === 0) return 1;
  if (ts.numerator % 3 === 0 && ts.numerator !== 3) return f32(12 / ts.denominator);
  return f32(4 / ts.denominator);
};
const beatsPerBar = ts => {
  if (ts.numerator === 0) return 1;
  if (ts.numerator % 3 === 0 && ts.numerator !== 3) return Math.max(1, Math.floor(ts.numerator / 3));
  return Math.max(1, ts.numerator);
};

/** Score.get_downbeats(): bar lines in ticks, from 0 up to (not including) the score's end. Computed on
 *  symusic's beat grid in float32 quarters, quirks included, so the bar numbering matches Python's. */
export function downbeats(s) {
  const tpq = f32(s.tpq);
  const toQ = t => f32(f32(t) / tpq);
  const endQ = toQ(scoreEnd(s)), start = 0;
  if (!(start < endQ)) return [];
  const sigs = stableSortByTime(s.timeSignatures).map(ts => ({ time: toQ(ts.time), numerator: ts.numerator, denominator: ts.denominator }));
  const beats = [start];
  let next = 0, cur = null;
  while (next < sigs.length && geOrClose(beats[beats.length - 1], sigs[next].time)) cur = sigs[next++];
  while (beats[beats.length - 1] < endQ) {
    let nb = f32(beats[beats.length - 1] + beatStep(cur));
    if (next < sigs.length && geOrClose(nb, sigs[next].time)) {
      nb = sigs[next].time;
      cur = sigs[next++];
      while (next < sigs.length && close(sigs[next].time, nb)) cur = sigs[next++];
    }
    beats.push(nb);
  }
  beats.pop();
  if (!beats.length) return [];
  if (!sigs.length || sigs[0].time > start) sigs.unshift({ time: start, numerator: 4, denominator: 4 });
  const find = (target, dflt) => { for (let i = 0; i < beats.length; i++) if (close(beats[i], target)) return i; return dflt; };
  const down = [];
  let endIdx = 0;
  for (let i = 0; i + 1 < sigs.length; i++) {
    const startIdx = find(sigs[i].time, 0);
    endIdx = find(sigs[i + 1].time, startIdx);
    for (let b = startIdx; b < endIdx; b += beatsPerBar(sigs[i])) down.push(beats[b]);
  }
  const last = sigs[sigs.length - 1];
  for (let b = find(last.time, endIdx); b < beats.length; b += beatsPerBar(last)) down.push(beats[b]);
  return down.map(q => roundHalfAway(f32(q * tpq)));
}

export const noteCount = s => s.tracks.reduce((n, t) => n + t.notes.length, 0);
