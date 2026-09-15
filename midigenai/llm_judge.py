"""
LLM-as-judge for continuation pairs, and the validation that says whether to
believe it.

The bottleneck on preference data is human listening time, so a judge that
agreed with the labeler would unlock thousands of pairs (the RLAIF recipe:
judge labels pairs -> cheap reward model fits those labels -> RL against it).
Whether a text model can hear anything useful in symbolic music is an
empirical question, and we can answer it: there are already human votes on
these exact pairs, so the judge is scored the same way any reward is —
agreement with the human, against the human's own self-consistency ceiling.

Two things this guards against:
  * **position bias** — every pair is judged twice with the sides swapped;
    a judge that just says "the first one" is caught by its swap agreement.
  * **silent format failure** — the music is rendered to ABC (what LLMs have
    actually seen in training) and the renderer is checked before any call.

    export OPENAI_API_KEY=...
    python -m midigenai.llm_judge validate \\
        --labels evals/labeling_v3_same/labels.jsonl --limit 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPTS: dict[str, str] = {}

PROMPTS["base"] = """You judge short continuations of a musical phrase, written in symbolic notation.

You are given a PROMPT (the phrase a musician played) and two candidate
CONTINUATIONS produced by a music model. Pick the one a working musician
would rather hear as the next few seconds of that phrase.

Weigh, in order:
1. Fit with the prompt — key, mode, harmonic direction, groove, register,
   and density that follow from what came before.
2. Internal musicality — phrasing that goes somewhere, coherent rhythm,
   no aimless wandering.
3. Absence of defects — near-silence, a single pitch hammered, an exact loop,
   notes crammed into noise.

Ignore genre preference: a well-made polka beats a sloppy nocturne. Ignore
length. If they are genuinely equal, say "tie" — do not guess.

Reply with JSON only: {"winner": "1" | "2" | "tie", "reason": "<12 words>"}"""

# What the labeler's own votes revealed (Bradley-Terry fit over 120 pairs,
# 2026-09-14): the strongest weights are *negative* on rhythmic entropy
# (-0.84) and note density (-0.43), mildly positive on repetition. In plain
# terms this labeler picks the calmer, steadier, more groove-like take over
# the busier one. "taste" states that; "strict" adds an abstention rule,
# since a judge that guesses on coin-flips only adds noise to the data.
PROMPTS["taste"] = PROMPTS["base"].replace(
    "Ignore genre preference:",
    """Two tendencies of the listener you stand in for, when the choice is close:
  * restraint beats busyness \u2014 fewer, better-placed notes over a flurry;
  * a steady, repeating groove beats rhythmic scattering. Repetition with
    intent is a strength here, not a weakness.
These break ties; they never outrank fit with the prompt or obvious defects.

Ignore genre preference:""")

PROMPTS["strict"] = PROMPTS["taste"].replace(
    "If they are genuinely equal, say \"tie\" \u2014 do not guess.",
    """Answer "tie" whenever you would be guessing: an abstention costs nothing,
a coin-flip verdict poisons the data. Name a winner only if you could defend
it in one sentence to the musician who played the prompt.""")

PROMPTS["fit_only"] = """You judge short continuations of a musical phrase, written in symbolic notation.

You are given a PROMPT (the phrase a musician played) and two candidate
CONTINUATIONS. Judge one thing: which one sounds like it belongs to the same
piece of music as the prompt?

Same key and harmony, same groove and subdivision, same register and density,
the instruments behaving as they behaved. A continuation that is pleasant on
its own but unrelated to the prompt loses to a plainer one that clearly
belongs.

Reply with JSON only: {"winner": "1" | "2" | "tie", "reason": "<12 words>"}"""

SYSTEM = PROMPTS["base"]


PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# the GM percussion voices worth naming; anything else prints as its key number
GM_DRUMS = {35: "Kick", 36: "Kick", 37: "Rim", 38: "Snare", 39: "Clap", 40: "Snare",
            41: "TomLo", 42: "HatClosed", 43: "TomLo", 44: "HatPedal", 45: "TomMid",
            46: "HatOpen", 47: "TomMid", 48: "TomHi", 49: "Crash", 50: "TomHi",
            51: "Ride", 53: "Bell", 55: "Splash", 57: "Crash", 59: "Ride"}


def to_notes(midi_path: Path, max_bars: int = 12) -> str | None:
    """Bar-by-bar note list: unambiguous where our ABC is not.

    midi2abc renders machine-generated polyphony as walls of tied chord
    brackets ("[d-c-B-G-E-B,-G,,-]") that look nothing like the folk tunes
    that made ABC familiar to language models, and it silently turns a drum
    channel into pitches. This spells out beat position, pitch name and
    duration, and names the drum voices.
    """
    from symusic import Score
    try:
        sc = Score(str(midi_path))
    except Exception:
        return None
    tpq = max(sc.ticks_per_quarter, 1)
    num, den = 4, 4
    if len(sc.time_signatures):
        ts = min(sc.time_signatures, key=lambda t: t.time)
        num, den = ts.numerator, ts.denominator
    beats_per_bar = num * 4 / den
    bpm = sc.tempos[0].qpm if len(sc.tempos) else 120.0

    bars: dict[int, list[str]] = {}
    for track in sc.tracks:
        for n in track.notes:
            beat = n.start / tpq
            bar = int(beat // beats_per_bar)
            if bar >= max_bars:
                continue
            pos = round(beat - bar * beats_per_bar, 2)
            if track.is_drum:
                label = GM_DRUMS.get(int(n.pitch), f"Perc{int(n.pitch)}")
                bars.setdefault(bar, []).append(f"{pos:g}:{label}")
            else:
                name = f"{PITCH_NAMES[int(n.pitch) % 12]}{int(n.pitch) // 12 - 1}"
                bars.setdefault(bar, []).append(
                    f"{pos:g}:{name}:{round(n.duration / tpq, 2):g}")
    if not bars:
        return None
    lines = [f"tempo {bpm:.0f} bpm, meter {num}/{den}, "
             f"format beat:pitch:duration_in_beats (drums are named voices)"]
    for bar in sorted(bars):
        notes = sorted(bars[bar], key=lambda x: float(x.split(":")[0]))
        lines.append(f"bar {bar + 1}: " + "  ".join(notes))
    return "\n".join(lines)


def render(midi_path: Path, fmt: str) -> str | None:
    return to_abc(midi_path) if fmt == "abc" else to_notes(midi_path)


def to_abc(midi_path: Path) -> str | None:
    from symusic import Score
    try:
        abc = Score(str(midi_path)).dumps_abc()
    except Exception:
        return None
    # drop the header noise (temp filename, generated comments) — keep the
    # musical lines plus key/meter, which is what the judge needs
    keep = [ln for ln in abc.splitlines()
            if not ln.startswith(("T:", "X:", "%%")) and ln.strip()]
    text = "\n".join(keep).strip()
    return text or None


def build_prompt(prompt_abc: str, a_abc: str, b_abc: str) -> str:
    return (f"PROMPT:\n{prompt_abc}\n\n"
            f"CONTINUATION 1:\n{a_abc}\n\n"
            f"CONTINUATION 2:\n{b_abc}\n\n"
            "Which continuation is better? JSON only.")


def ask(client, model: str, user: str, temperature: float = 0.0,
        system: str | None = None) -> tuple[str, str]:
    kw = {"model": model,
          "messages": [{"role": "system", "content": system or SYSTEM},
                       {"role": "user", "content": user}]}
    # the reasoning models (gpt-5.x) reject any temperature but their default
    if temperature is not None and not model.startswith("gpt-5."):
        kw["temperature"] = temperature
    r = client.chat.completions.create(**kw)
    txt = (r.choices[0].message.content or "").strip()
    m = re.search(r'\{.*\}', txt, re.S)
    if not m:
        return "tie", "unparseable"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "tie", "unparseable"
    w = str(d.get("winner", "tie")).strip().lower()
    return (w if w in ("1", "2", "tie") else "tie"), str(d.get("reason", ""))[:80]


def judge_pair(client, model, prompt_abc, a_abc, b_abc, system=None) -> dict:
    """Judged twice with the sides swapped; a stable judge gives mirrored answers."""
    w1, r1 = ask(client, model, build_prompt(prompt_abc, a_abc, b_abc), system=system)
    w2, r2 = ask(client, model, build_prompt(prompt_abc, b_abc, a_abc), system=system)
    flip = {"1": "2", "2": "1", "tie": "tie"}
    w2_unswapped = flip[w2]
    consistent = w1 == w2_unswapped
    verdict = w1 if consistent else "tie"
    return {"verdict": verdict, "first_pass": w1, "second_pass": w2_unswapped,
            "consistent": consistent, "reason": r1 or r2}


def load_cases(labels_path: Path, limit: int, seed: int, split: str = "all"):
    """Decided human votes with their MIDI files, as (pair_id, human, paths).

    `split` partitions by a hash of the pair id: tune prompts on "dev" and
    report on "test", or the number you report is just the tuning score.
    """
    import hashlib
    pairs_dir = labels_path.parent / "pairs"
    out = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("choice") not in ("left", "right"):
            continue
        pid = r["pair_id"]
        win = r.get("preferred") or (r["left_is"] if r["choice"] == "left" else r["right_is"])
        lose = "b" if win == "a" else "a"
        p, w, l = (pairs_dir / f"{pid}_prompt.mid", pairs_dir / f"{pid}_{win}.mid",
                   pairs_dir / f"{pid}_{lose}.mid")
        if split != "all":
            h = int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 2
            if (h == 0) != (split == "dev"):
                continue
        if p.exists() and w.exists() and l.exists():
            out.append((pid, win, p, w, l))
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def validate(args) -> None:
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    client = OpenAI()
    system = (Path(args.system_file).read_text() if args.system_file
              else PROMPTS[args.prompt])
    cases = load_cases(args.labels, args.limit, args.seed, split=args.split)
    print(f"[judge] {len(cases)} human-voted pairs, model={args.model}, "
          f"format={args.format}, prompt={args.system_file or args.prompt}, "
          f"split={args.split}")

    rng = random.Random(args.seed)

    def one(case):
        pid, _win, p, w, l = case
        pa, aw, al = (render(p, args.format), render(w, args.format),
                      render(l, args.format))
        if not (pa and aw and al):
            return None
        # randomise which side the human's winner is shown as, so the judge
        # cannot be right by always answering "1"
        winner_first = rng.random() < 0.5
        first, second = (aw, al) if winner_first else (al, aw)
        try:
            res = judge_pair(client, args.model, pa, first, second, system=system)
        except Exception as e:
            return {"pair_id": pid, "error": f"{type(e).__name__}: {e}"[:120]}
        human_side = "1" if winner_first else "2"
        return {"pair_id": pid, "human": human_side, "winner_first": winner_first, **res}

    with ThreadPoolExecutor(args.concurrency) as ex:
        results = [r for r in ex.map(one, cases) if r]

    errs = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    if errs:
        print(f"[judge] {len(errs)} errors, e.g. {errs[0]['error']}")
    if not ok:
        raise SystemExit("no usable judgements")

    decided = [r for r in ok if r["verdict"] != "tie"]
    agree = sum(r["verdict"] == r["human"] for r in decided)
    swap_ok = sum(r["consistent"] for r in ok)
    firsts = sum(r["first_pass"] == "1" for r in ok)
    out = {
        "model": args.model, "format": args.format,
        "prompt": args.system_file or args.prompt, "split": args.split,
        "n": len(ok), "n_decided": len(decided),
        "agreement_with_human": agree / len(decided) if decided else None,
        "swap_consistency": swap_ok / len(ok),
        "position_bias_pick_first": firsts / len(ok),
        "tie_rate": 1 - len(decided) / len(ok),
        "results": ok,
    }
    print(f"[judge] agreement with human (decided only): "
          f"{out['agreement_with_human']:.3f} on {len(decided)} pairs")
    print(f"[judge] swap consistency: {out['swap_consistency']:.3f}   "
          f"picks side 1 first pass: {out['position_bias_pick_first']:.3f}   "
          f"ties: {out['tie_rate']:.3f}")
    print("[judge] reference: 10-metric reward 0.658, model log-prob 0.717, "
          "labeler ceiling 0.88")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[judge] wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--labels", type=Path, required=True)
    v.add_argument("--model", default="gpt-4.1-mini")
    v.add_argument("--prompt", choices=sorted(PROMPTS), default="base",
                   help="which judging rubric to use")
    v.add_argument("--system-file", type=Path, default=None,
                   help="read the rubric from a file instead")
    v.add_argument("--split", choices=["all", "dev", "test"], default="all",
                   help="tune on dev, report on test")
    v.add_argument("--format", choices=["abc", "notes"], default="notes",
                   help="how the music is shown to the judge; validate both and "
                        "let agreement with the human decide")
    v.add_argument("--limit", type=int, default=60)
    v.add_argument("--concurrency", type=int, default=6)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    validate(a)


if __name__ == "__main__":
    main()
