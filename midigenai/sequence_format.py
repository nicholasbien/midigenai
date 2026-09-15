"""
v4 document layouts, shared by the dataset builder, the training loop and
the generator so the three can never disagree about where SEP goes.

    continuation:   BOS [header] body EOS
    accompaniment:  BOS [header] cond SEP target EOS
    span infill:    BOS [header] prefix MASK suffix SEP middle EOS

`cond` and `target` cover the same bars (same number of Bar tokens); the
infill `middle` is the bars removed between `prefix` and `suffix`. Each
segment is tokenized on its own, so its Bar/Position tokens start at bar 0 of
that segment. At inference the prompt is everything up to and including SEP
(or, for continuation, the header + body); the model writes the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attributes import HEADER_PREFIXES, NEVER_DROP, is_header_token
from .tokenizer import special_id


@dataclass(frozen=True)
class Specials:
    bos: int
    eos: int
    sep: int
    mask: int | None
    bar: int | None
    task_accomp: int | None
    task_infill: int | None
    header_ids: frozenset[int]          # every Inst_/Density_/... id
    header_family: dict[int, str]       # id -> prefix, for per-family dropout

    @classmethod
    def from_tokenizer(cls, tok) -> "Specials":
        fam = {}
        for name, tid in tok.vocab.items():
            if is_header_token(name):
                fam[tid] = next(p for p in HEADER_PREFIXES if name.startswith(p))
        return cls(
            bos=special_id(tok, "BOS"), eos=special_id(tok, "EOS"),
            sep=special_id(tok, "SEP"), mask=special_id(tok, "MASK"),
            bar=tok.vocab.get("Bar_None"),
            task_accomp=special_id(tok, "Task_accomp"),
            task_infill=special_id(tok, "Task_infill"),
            header_ids=frozenset(fam), header_family=fam,
        )

    def header_ids_for(self, tok, names: list[str]) -> list[int]:
        out = []
        for n in names:
            tid = special_id(tok, n)
            if tid is None:
                raise KeyError(f"not a header token: {n}")
            out.append(tid)
        return out


# ---------- assembly (ids in, ids out) ---------- #

def continuation_doc(sp: Specials, header: list[int], body: list[int]) -> list[int]:
    return [sp.bos, *header, *body, sp.eos]


def _task(tid: int | None) -> list[int]:
    return [tid] if tid is not None else []


def accompaniment_doc(sp: Specials, header: list[int], cond: list[int],
                      target: list[int]) -> list[int]:
    return [sp.bos, *_task(sp.task_accomp), *header, *cond, sp.sep, *target, sp.eos]


def infill_doc(sp: Specials, header: list[int], prefix: list[int],
               suffix: list[int], middle: list[int]) -> list[int]:
    assert sp.mask is not None, "infill needs a MASK token (v4 tokenizer)"
    return [sp.bos, *_task(sp.task_infill), *header, *prefix, sp.mask, *suffix, sp.sep, *middle, sp.eos]


def continuation_prompt(sp: Specials, header: list[int], body: list[int]) -> list[int]:
    return [sp.bos, *header, *body]


def accompaniment_prompt(sp: Specials, header: list[int], cond: list[int]) -> list[int]:
    return [sp.bos, *_task(sp.task_accomp), *header, *cond, sp.sep]


def infill_prompt(sp: Specials, header: list[int], prefix: list[int],
                  suffix: list[int]) -> list[int]:
    return [sp.bos, *_task(sp.task_infill), *header, *prefix, sp.mask, *suffix, sp.sep]


# ---------- inspection ---------- #

def split_header(sp: Specials, ids) -> tuple[list[int], list[int]]:
    """(header, rest) for a document body that starts right after BOS."""
    i = 0
    while i < len(ids) and int(ids[i]) in sp.header_ids:
        i += 1
    return [int(x) for x in ids[:i]], list(ids[i:])


def count_bars(sp: Specials, ids) -> int:
    if sp.bar is None:
        return 0
    return int(np.count_nonzero(np.asarray(ids) == sp.bar))


def drop_header_families(sp: Specials, header: list[int], rng,
                         p_family: float, p_all: float) -> list[int]:
    """Classifier-free-guidance style dropout: with prob `p_all` drop the
    whole header, otherwise drop each attribute family independently with
    prob `p_family`. The unconditional model stays in-distribution and any
    subset of attributes is a valid prompt at inference."""
    if not header:
        return header
    keep_always = [t for t in header if sp.header_family[t] in NEVER_DROP]
    if rng.random() < p_all:
        return keep_always
    fams = {sp.header_family[t] for t in header}
    keep = {f for f in fams if f in NEVER_DROP or rng.random() >= p_family}
    return [t for t in header if sp.header_family[t] in keep]
