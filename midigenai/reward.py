"""
Load a reward spec from `reward_align` and score generated continuations.

The scorer must produce exactly the features the spec was fitted on, in the
same order and with the same normalization, or the weights mean nothing:

    reward(x) = w · (f(x) / diff_std)

`diff_std` is the standard deviation of the winner-minus-loser differences
seen during fitting, so a reward difference of 1.0 between two samples is
roughly "one typical pair's worth" of preference.

Only *differences* are meaningful — Bradley-Terry has no absolute scale — so
GRPO's group-relative advantage (subtract the group mean) is exactly the
right consumer of these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from midigenai.eval import (ioi_entropy, note_density_hz, pitch_class_entropy,
                            pitch_range, polyphony_rate, repetition_rate,
                            scale_consistency)

# same names reward_align.FEATURES / DRIFT_FEATURES use
_SCORE_FEATURES = {
    "pitch_class_entropy": pitch_class_entropy,
    "scale_consistency": scale_consistency,
    "polyphony_rate": polyphony_rate,
    "note_density_hz": note_density_hz,
    "pitch_range": pitch_range,
    "repetition_rate": repetition_rate,
    "ioi_entropy": ioi_entropy,
}
_DRIFT_FEATURES = ("repetition_drift", "density_drift", "pce_drift")


class Reward:
    def __init__(self, spec: dict):
        self.features: list[str] = list(spec["features"])
        self.weights = np.asarray(spec["weights"], dtype=np.float64)
        self.diff_std = np.asarray(spec["diff_std"], dtype=np.float64)
        self.diff_std[self.diff_std == 0] = 1.0
        self.heldout_accuracy = spec.get("heldout_accuracy")
        self.self_consistency = spec.get("self_consistency")
        unknown = [f for f in self.features
                   if f not in _SCORE_FEATURES and f not in _DRIFT_FEATURES]
        if unknown:
            raise ValueError(f"reward spec wants features this scorer cannot compute: {unknown}")

    @classmethod
    def load(cls, path: str | Path) -> "Reward":
        return cls(json.loads(Path(path).read_text()))

    def feature_vector(self, tokenizer, cont_ids) -> np.ndarray | None:
        """Features of one continuation, from its token ids alone."""
        cont_ids = list(cont_ids)
        if len(cont_ids) < 8:
            return None
        try:
            score = tokenizer.decode(cont_ids)
        except Exception:
            return None
        if sum(len(t.notes) for t in score.tracks) == 0:
            return None
        halves = None
        if any(f in _DRIFT_FEATURES for f in self.features):
            if len(cont_ids) < 32:
                return None
            h = len(cont_ids) // 2
            try:
                halves = (tokenizer.decode(cont_ids[:h]), tokenizer.decode(cont_ids[h:]))
            except Exception:
                return None
        out = []
        for name in self.features:
            try:
                if name in _SCORE_FEATURES:
                    out.append(float(_SCORE_FEATURES[name](score)))
                elif name == "repetition_drift":
                    out.append(float(repetition_rate(halves[1]) - repetition_rate(halves[0])))
                elif name == "density_drift":
                    out.append(float(note_density_hz(halves[1]) - note_density_hz(halves[0])))
                else:
                    out.append(float(pitch_class_entropy(halves[1]) - pitch_class_entropy(halves[0])))
            except Exception:
                return None
        v = np.asarray(out, dtype=np.float64)
        return v if np.isfinite(v).all() else None

    def score(self, tokenizer, cont_ids) -> float | None:
        v = self.feature_vector(tokenizer, cont_ids)
        if v is None:
            return None
        return float(self.weights @ (v / self.diff_std))
