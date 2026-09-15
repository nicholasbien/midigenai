# The LLM judge, and the ceiling that makes it readable

*Measured 2026-09-15. Numbers below are reproducible from the commands in each section.*

The bottleneck on preference data is listening time. A judge that agrees with
the labeler unlocks thousands of pairs for the price of an API call, which is
the RLAIF recipe: judge labels pairs → cheap reward model fits those labels →
RL against it. Whether a text model can hear anything useful in symbolic music
is an empirical question, and we can answer it, because there are already
human votes on these exact pairs.

## The headline

| rater | agreement | n | 95% CI |
|---|---|---|---|
| **labeler vs. their own earlier vote** (the ceiling) | **0.881** | 42 | ±0.098 |
| **judge vs. labeler**, same 42 pairs | **0.816** | 38 | ±0.123 |
| judge vs. labeler, full corpus | 0.782 | 225 | ±0.054 |
| model log-prob | 0.717 | — | — |
| 10-feature linear reward | 0.658 | — | — |
| hidden-state probe (768-d, 92 pairs) | 0.598 | — | — |

The gap between the judge and the ceiling is 0.065 — z = 0.81, p = 0.42.
**Statistically indistinguishable.** The judge is at the limit of what these
labels can resolve, so further prompt engineering is fitting noise.

Config: `gpt-5.6-sol`, `strict` rubric, `notes` rendering, every pair judged
twice with the sides swapped.

## Why the ceiling had to be measured first

Before 2026-09-15 every judge run printed `labeler ceiling 0.88` and that
number was not traceable to any measurement in the repo: the preference
corpus contained **4 repeated pairs**, and the corpus quality-rating task
(a different task) had 7 repeats at 0.429 exact agreement.

Without the ceiling, an agreement of 0.79 is uninterpretable — it is either a
judge at the limit of the signal or a judge with 15 points of headroom, and
those two worlds call for opposite decisions. Measuring it cost one labeling
session.

`label_app --dup-rate` only re-serves pairs voted *within the same live
session*, so it cannot measure the ceiling on the historical pairs the judge
is scored against. `relabel_app` serves those pairs from disk instead:

```bash
python -m midigenai.relabel_app select -n 60 --out evals/ceiling
python -m midigenai.relabel_app serve  --out evals/ceiling --port 7791
python -m midigenai.relabel_app score  --out evals/ceiling
```

Blind (the UI shows "repeat check" and blanks the model names), with the
sides re-randomised independently of the first showing, written to a separate
file so the original labels are never touched. The 60 pairs are stratified
across the four label sets in proportion to their size.

## Scoring rules, and why they are what they are

**Both raters are scored only on what they were willing to decide.** The
judge's ties are dropped from its denominator, so the labeler's skips are
dropped from theirs. Folding one rater's abstentions in as errors while the
other's are excluded would compare two different things.

**"Both bad" is not an abstention.** It is a verdict on the *pair*: there is
no right answer to be scored on, so those pairs leave both denominators and
their ids are written to `evals/ceiling/unusable_pairs.txt`. 4 of 60.

**Swap-inconsistency is recorded as a tie.** A judge that answers differently
when the sides are exchanged has not expressed a preference. This is why
"decided" already implies "survived the swap check".

`score` also prints a pessimistic floor (skips counted as disagreements,
0.661) purely to show how much the choice of denominator moves the number.

## Two independent reasons to believe it

**The two raters find the same pairs hard.** The judge ties on 7% of the
pairs the labeler re-voted, and 22% of the pairs the labeler abstained on —
3× higher. Nothing was tuned to produce that; the judge's uncertainty tracks
the human's on the same music.

**Some disagreement is irreducible.** On the 5 pairs where the labeler
contradicted their own earlier vote, the judge never tied and matched the
*original* vote 4 times. Those score as judge errors against pass-1 labels,
but there is no ground truth to be right about. Roughly 12% of pairs are
unresolvable by anyone, and that caps every rater.

## What was ruled out along the way

| variable | result |
|---|---|
| rendering: `notes` vs `abc` | `notes` wins clearly (0.73 vs 0.60–0.67) — `midi2abc` renders machine polyphony as walls of tied chord brackets and turns drums into pitches |
| rubric: base / fit_only / taste / strict | `strict` (0.737 dev) > taste (0.722) > base (0.692) > fit_only (0.667) |
| model | `gpt-5.6-sol` > `gpt-5.6-luna` > `gpt-4.1-mini` |
| position bias | 0.433 first-pass pick rate — no meaningful side preference |

**Do not tune the tie rate down.** Abstention is what makes the judge useful:
at 0.82 on the 85% it decides, it produces better training data than a judge
forced to 1.00 coverage at 0.74.

## Per-set results

| set | n | decided | agreement | swap-cons | tie |
|---|---|---|---|---|---|
| v1 | 30 | 25 | 0.720 | 0.833 | 0.17 |
| v3_same | 92 | 78 | 0.769 | 0.848 | 0.15 |
| v4 | 48 | 39 | 0.795 | 0.812 | 0.19 |
| v4_final | 105 | 93 | 0.817 | 0.886 | 0.11 |
| **pooled** | **275** | **235** | **0.787** | 0.855 | 0.15 |

Only `v3_same` was used to select the `strict` rubric. Held out from that
tuning (v1 + v4 + v4_final): **0.796 ±0.063** on 157 pairs.

The pair MIDI lives in `evals/labeling*/pairs/`, which is gitignored; for
every set except v1 it is only in the `midigenai-v4` worktree. Point
`--labels` there.

```bash
python -m midigenai.llm_judge validate \
    --labels ~/midigenai-v4/evals/labeling_v4_final/labels.jsonl \
    --model gpt-5.6-sol --prompt strict --format notes \
    --split all --limit 0 --concurrency 12 \
    --out evals/reward/judge_all_v4_final.json
```

## Cost, measured

2,032 input / 196 output tokens per call (mostly reasoning), 5.5s median
latency, two calls per pair.

| | per pair | 5,800 pairs |
|---|---|---|
| tokens | 4.1k in / 0.4k out | 23.6M in / 2.3M out |
| wall-clock @ concurrency 12 | ~0.9s | ~90 min |
| usable labels | — | ~4,250 |

~8% of generated pairs are unusable (both-bad or bad prompt) and the judge
abstains on ~15% of the rest, so generate ~5,800 to land ~4,250 labels.
Prefix caching saves little: the rubric is ~400 of the 2,032 input tokens and
the rest is music that differs per pair.

**Generation, not judging, is the bottleneck** — pair MIDI was written at a
median 18.2s per pair on one local worker. Parallelize it.

## What this licenses, and what it does not

Scaling to machine labels is justified: the judge is at the labeler's ceiling,
its uncertainty tracks theirs, and it costs ~$100 for what would otherwise be
~40 hours of listening.

What it does not license is trusting the resulting reward model past the noise
floor. ~12% of pairs are unresolvable and the judge misses ~18% of the rest,
so a reward model fit on judge labels that reports held-out accuracy much
above ~0.85 is memorizing, not learning. And the judge is only validated on
*this* distribution — pairs from v3/v4 checkpoints on held-out prompts. A
policy pushed far from that by RL is outside the distribution the judge was
checked on, which is what the KL leash in `grpo.py` is for.
