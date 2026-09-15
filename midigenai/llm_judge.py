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

# The rubrics are plain text files, not code: they are what the judge was
# validated with, and reviewing a prompt means reading it, not reconstructing
# it from three chained .replace() calls. Nothing in them is interpolated —
# the rubric is a static system prompt and the music goes in the user message
# (see build_prompt). Editing a file changes the judge; the agreement figures
# in docs/LLM_JUDGE.md only describe the text as committed.
#
#   judge_base.txt      the rubric: fit, then musicality, then defects
#   judge_taste.txt     base + what the labeler's own votes revealed
#                       (restraint over busyness, groove over scattering)
#   judge_strict.txt    taste + "abstain rather than guess"  <- THE DEFAULT,
#                       0.816 against the labeler, at their 0.881 ceiling
#   judge_fit_only.txt  ablation: judge only whether it belongs to the piece
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "strict"

PROMPTS: dict[str, str] = {
    f.stem[len("judge_"):]: f.read_text().rstrip()
    for f in sorted(PROMPT_DIR.glob("judge_*.txt"))
}
if DEFAULT_PROMPT not in PROMPTS:
    raise RuntimeError(f"missing {PROMPT_DIR}/judge_{DEFAULT_PROMPT}.txt")

SYSTEM = PROMPTS[DEFAULT_PROMPT]


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
    ceiling = Path("evals/ceiling/ceiling.json")
    if ceiling.exists():
        c = json.loads(ceiling.read_text())
        print(f"[judge] reference: 10-metric reward 0.658, model log-prob 0.717, "
              f"labeler ceiling {c['self_consistency']:.3f} "
              f"(measured on {c['n']} blind repeats, {ceiling})")
    else:
        print("[judge] reference: 10-metric reward 0.658, model log-prob 0.717; "
              "labeler ceiling unmeasured — run midigenai.relabel_app")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[judge] wrote {args.out}")



def label(args) -> None:
    """Judge unlabelled pairs and write labels in `label_app`'s schema.

    Appends as it goes, and skips pair ids already in the output, so it is
    resumable and can be pointed at a directory that is still filling up
    while generation runs.
    """
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    client = OpenAI()
    system = (Path(args.system_file).read_text() if args.system_file
              else PROMPTS[args.prompt])

    pairs_dir = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["pair_id"]
                for l in out_path.read_text().splitlines() if l.strip()}

    ids = sorted(f.name[:-len("_prompt.mid")]
                 for f in pairs_dir.glob("*_prompt.mid"))
    todo = [i for i in ids
            if i not in done and (pairs_dir / f"{i}_a.mid").exists()
            and (pairs_dir / f"{i}_b.mid").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[label] {len(todo)} pairs to judge ({len(done)} already done), "
          f"model={args.model}, prompt={args.system_file or args.prompt}")

    lock = __import__("threading").Lock()
    counts = {"a": 0, "b": 0, "tie": 0, "error": 0}

    def one(pid: str):
        pa = render(pairs_dir / f"{pid}_prompt.mid", args.format)
        ra = render(pairs_dir / f"{pid}_a.mid", args.format)
        rb = render(pairs_dir / f"{pid}_b.mid", args.format)
        if not (pa and ra and rb):
            return {"pair_id": pid, "error": "unrenderable"}
        # Show a first half the time, so a side-biased judge cannot look
        # right for the wrong reason. Derived from the pair id rather than a
        # shared random.Random: this runs under a ThreadPoolExecutor (Random
        # is not thread-safe) and resumes re-run the function, which would
        # replay an identical seeded sequence every batch.
        import hashlib
        a_first = int(hashlib.sha1(f"{args.seed}:{pid}".encode()).hexdigest(), 16) & 1 == 0
        first, second = (ra, rb) if a_first else (rb, ra)
        try:
            res = judge_pair(client, args.model, pa, first, second, system=system)
        except Exception as e:
            return {"pair_id": pid, "error": f"{type(e).__name__}: {e}"[:120]}
        if res["verdict"] == "tie":
            preferred = "tie"
        else:
            won_first = res["verdict"] == "1"
            preferred = ("a" if won_first else "b") if a_first else ("b" if won_first else "a")
        # label_app's schema, which every downstream loader expects: `choice`
        # is the SIDE that won ("left"/"right") and `preferred` resolves it to
        # canonical a/b. reward_probe filters on choice, so writing "a"/"b"
        # there made it skip every row.
        choice = ("tie" if preferred == "tie"
                  else "left" if preferred == "a" else "right")
        return {"ts": utcnow(), "pair_id": pid, "preferred": preferred,
                "choice": choice, "left_is": "a", "right_is": "b",
                "judge_model": args.model, "judge_prompt": args.system_file or args.prompt,
                "swap_consistent": res["consistent"], "a_shown_first": a_first,
                "reason": res["reason"]}

    with ThreadPoolExecutor(args.concurrency) as ex, out_path.open("a") as fh:
        for rec in ex.map(one, todo):
            with lock:
                if rec.get("error"):
                    counts["error"] += 1
                    continue
                counts[rec["preferred"]] += 1
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n = counts["a"] + counts["b"] + counts["tie"]
                if n % 25 == 0:
                    print(f"[label] {n} judged  a={counts['a']} b={counts['b']} "
                          f"tie={counts['tie']} err={counts['error']}")

    n = counts["a"] + counts["b"] + counts["tie"]
    decided = counts["a"] + counts["b"]
    print(f"[label] wrote {decided} decided + {counts['tie']} ties to {out_path} "
          f"({counts['error']} errors)")
    if n:
        print(f"[label] tie rate {counts['tie'] / n:.2f}, "
              f"side balance a={counts['a']} b={counts['b']} "
              f"(a lopsided split means the judge is reading position, not music)")


def utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--labels", type=Path, required=True)
    v.add_argument("--model", default="gpt-4.1-mini")
    v.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT,
                   help=f"which judging rubric to use (default: {DEFAULT_PROMPT}, "
                        "the one the agreement figures describe)")
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
    lb = sub.add_parser("label", help="judge unlabelled pairs into labels.jsonl")
    lb.add_argument("--pairs", type=Path, required=True,
                    help="directory of <id>_prompt.mid / _a.mid / _b.mid")
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--model", default="gpt-5.6-sol")
    lb.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT)
    lb.add_argument("--system-file", type=Path, default=None)
    lb.add_argument("--format", choices=["abc", "notes"], default="notes")
    lb.add_argument("--limit", type=int, default=0)
    lb.add_argument("--concurrency", type=int, default=12)
    lb.add_argument("--seed", type=int, default=0)

    a = p.parse_args()
    if a.cmd == "label":
        label(a)
    else:
        validate(a)


if __name__ == "__main__":
    main()
