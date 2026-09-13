"""
Fit and apply a deliberately simple corpus-quality predictor.

Labels come from midigenai.rate_corpus_app (evals/quality/ratings.jsonl): a
1-5 rating per ~30 s excerpt plus a dict of symbolic features computed on
that excerpt. With a few hundred labels the right model is ridge regression on
standardized features: closed form, no hyperparameters beyond alpha (chosen by
grouped cross-validation), coefficients you can read. If scikit-learn is
importable a GradientBoostingRegressor is also cross-validated for comparison,
but only the ridge model is saved.

    python -m midigenai.quality_predictor fit
    python -m midigenai.quality_predictor score --manifest ~/midigenai_data/manifest_all_dedup.jsonl \\
        --out ~/midigenai_data/quality_all

`fit` reports grouped 5-fold Spearman/Pearson/MAE, the self-consistency
ceiling from blind repeats, standardized coefficients, and per-source bias
(mean predicted vs mean true rating and within-source correlation) so you can
tell whether the model learned craft or just "which source is this".

`score` computes the same features on each manifest file (first --max-seconds
seconds of long files) with multiprocessing and writes <out>.jsonl rows of
{path, source, quality, q_bucket}; q_bucket is the quartile (0 = worst 25%,
3 = best 25%) of predicted quality over the scored set, intended as a shard
tag the dataset builder can weight via training's --mixture.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from midigenai.rate_corpus_app import (compute_features, extract_excerpt,
                                       load_ratings, self_consistency,
                                       source_of, tick_to_seconds_fn)

DEFAULT_LOG = Path("evals/quality/ratings.jsonl")
DEFAULT_MODEL = Path("evals/quality/predictor.json")
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


# --------------------------------- data ------------------------------------ #

def aggregate_ratings(records: list[dict]) -> list[dict]:
    """One row per excerpt: skips dropped, blind repeats averaged, an explicit
    revision (re-rated from the session list) replaces earlier ratings."""
    by_id: dict[str, dict] = {}
    for r in records:
        if r.get("rating") is None or not r.get("excerpt_id"):
            continue
        row = by_id.setdefault(r["excerpt_id"], {
            "excerpt_id": r["excerpt_id"], "path": r.get("path", ""),
            "source": r.get("source") or source_of(r.get("path", "")),
            "features": r.get("features") or {}, "ratings": []})
        if r.get("revision"):
            # a deliberate re-rate supersedes everything before it
            row["ratings"] = []
        row["ratings"].append(float(r["rating"]))
        if not row["features"] and r.get("features"):
            row["features"] = r["features"]
    rows = list(by_id.values())
    for row in rows:
        row["rating"] = float(np.mean(row["ratings"]))
    return rows


def feature_names_from(dicts: list[dict]) -> list[str]:
    names = set()
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float, bool)) and not isinstance(v, str):
                names.add(k)
    return sorted(names)


def feature_matrix(dicts: list[dict], names: list[str]) -> np.ndarray:
    X = np.full((len(dicts), len(names)), np.nan, dtype=np.float64)
    for i, d in enumerate(dicts):
        for j, k in enumerate(names):
            v = d.get(k)
            if isinstance(v, bool):
                X[i, j] = float(v)
            elif isinstance(v, (int, float)) and math.isfinite(v):
                X[i, j] = float(v)
    return X


# --------------------------------- model ----------------------------------- #

class RidgeModel:
    """Ridge regression on median-imputed, standardized features."""

    def __init__(self, names, medians, means, stds, coef, intercept, alpha, meta=None):
        self.names = list(names)
        self.medians = np.asarray(medians, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.alpha = float(alpha)
        self.meta = meta or {}

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, names: list[str], alpha: float):
        X = np.array(X, dtype=np.float64, copy=True)
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        X = np.where(np.isnan(X), medians, X)
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-12, 1.0, stds)
        Z = (X - means) / stds
        ybar = float(y.mean())
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        coef = np.linalg.solve(A, Z.T @ (y - ybar))
        return cls(names, medians, means, stds, coef, ybar, alpha)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), self.medians, X)
        return (X - self.means) / self.stds

    def predict(self, X: np.ndarray, clip: bool = True) -> np.ndarray:
        pred = self.transform(np.asarray(X, dtype=np.float64)) @ self.coef + self.intercept
        return np.clip(pred, 1.0, 5.0) if clip else pred

    def predict_features(self, feats: dict) -> float:
        return float(self.predict(feature_matrix([feats], self.names))[0])

    def to_dict(self) -> dict:
        return {"type": "ridge", "feature_names": self.names,
                "medians": self.medians.tolist(), "means": self.means.tolist(),
                "stds": self.stds.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "alpha": self.alpha, **self.meta}

    @classmethod
    def from_dict(cls, d: dict):
        meta = {k: v for k, v in d.items() if k not in
                ("type", "feature_names", "medians", "means", "stds", "coef",
                 "intercept", "alpha")}
        return cls(d["feature_names"], d["medians"], d["means"], d["stds"],
                   d["coef"], d["intercept"], d["alpha"], meta)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(Path(path).read_text()))


# ------------------------------ statistics --------------------------------- #

def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    return pearson(_rank(a), _rank(b))


def grouped_folds(groups: list[str], k: int, seed: int) -> np.ndarray:
    """Fold index per row; all rows sharing a group (file path) share a fold."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def cv_predict_ridge(X, y, names, folds, alpha) -> np.ndarray:
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = RidgeModel.fit(X[tr], y[tr], names, alpha)
        pred[te] = model.predict(X[te])
    return pred


def cv_predict_gbr(X, y, folds, seed):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[te]), med, X[te])
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=seed)
        gbr.fit(Xtr, y[tr])
        pred[te] = np.clip(gbr.predict(Xte), 1.0, 5.0)
    return pred


def metrics(y, pred) -> dict:
    return {"spearman": spearman(y, pred), "pearson": pearson(y, pred),
            "mae": float(np.mean(np.abs(y - pred)))}


def per_source_bias(rows, y, pred) -> dict:
    out = {}
    for src in sorted({r["source"] for r in rows}):
        idx = np.array([r["source"] == src for r in rows])
        out[src] = {"n": int(idx.sum()),
                    "mean_true": float(y[idx].mean()),
                    "mean_pred": float(pred[idx].mean()),
                    "spearman_within": spearman(y[idx], pred[idx]) if idx.sum() >= 3
                    else float("nan")}
    return out


def fit_predictor(records: list[dict], k: int = 5, seed: int = 0,
                  alpha: float | None = None, try_gbr: bool = True) -> tuple[RidgeModel, dict]:
    rows = aggregate_ratings(records)
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} rated excerpts; rate more first")
    names = feature_names_from([r["features"] for r in rows])
    X = feature_matrix([r["features"] for r in rows], names)
    y = np.array([r["rating"] for r in rows])
    folds = grouped_folds([r["path"] for r in rows], k, seed)

    # choose alpha by grouped CV Spearman (ranking is what buckets use)
    if alpha is None:
        scores = {a: spearman(y, cv_predict_ridge(X, y, names, folds, a)) for a in ALPHA_GRID}
        alpha = max(scores, key=lambda a: (np.nan_to_num(scores[a], nan=-2), -a))
        alpha_search = {str(a): float(s) for a, s in scores.items()}
    else:
        alpha_search = None
    pred = cv_predict_ridge(X, y, names, folds, alpha)
    report = {
        "n_excerpts": len(rows), "n_ratings": int(sum(len(r["ratings"]) for r in rows)),
        "n_files": len({r["path"] for r in rows}), "k_folds": k, "alpha": alpha,
        "alpha_search": alpha_search,
        "ridge_cv": metrics(y, pred),
        "baseline_mae_predict_mean": float(np.mean(np.abs(y - y.mean()))),
        "self_consistency": self_consistency(records),
        "per_source": per_source_bias(rows, y, pred),
        "rating_hist": {str(k_): int(v) for k_, v in
                        sorted(Counter(int(round(v)) for v in y).items())},
    }
    if try_gbr:
        gpred = cv_predict_gbr(X, y, folds, seed)
        report["gbr_cv"] = metrics(y, gpred) if gpred is not None else None

    model = RidgeModel.fit(X, y, names, alpha)
    model.meta = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "report": report}
    report["coefficients"] = dict(sorted(zip(names, model.coef.tolist()),
                                         key=lambda kv: -abs(kv[1])))
    return model, report


def format_report(report: dict) -> str:
    L = []
    L.append(f"labels: {report['n_ratings']} ratings on {report['n_excerpts']} excerpts "
             f"from {report['n_files']} files; rating hist {report['rating_hist']}")
    sc = report["self_consistency"]
    if sc["pairs"]:
        L.append(f"self-consistency ceiling ({sc['pairs']} repeat pairs): exact "
                 f"{sc['exact_agreement']:.0%}, mean |diff| {sc['mean_abs_diff']:.2f}")
    else:
        L.append("self-consistency: no repeats yet")
    r = report["ridge_cv"]
    L.append(f"ridge (alpha={report['alpha']}), grouped {report['k_folds']}-fold CV: "
             f"spearman {r['spearman']:.3f}  pearson {r['pearson']:.3f}  MAE {r['mae']:.3f} "
             f"(predict-the-mean MAE {report['baseline_mae_predict_mean']:.3f})")
    if report.get("alpha_search"):
        L.append("  alpha search (spearman): " +
                 ", ".join(f"{a}: {s:.3f}" for a, s in report["alpha_search"].items()))
    if report.get("gbr_cv"):
        g = report["gbr_cv"]
        L.append(f"sklearn GBR, same folds: spearman {g['spearman']:.3f}  "
                 f"pearson {g['pearson']:.3f}  MAE {g['mae']:.3f}  (not saved)")
    L.append("standardized coefficients (rating change per 1 SD of feature):")
    for k, v in report["coefficients"].items():
        L.append(f"  {k:<24} {v:+.3f}")
    L.append("per-source bias (CV predictions):")
    L.append(f"  {'source':<10} {'n':>4} {'mean true':>10} {'mean pred':>10} {'spearman within':>16}")
    for src, d in report["per_source"].items():
        sw = d["spearman_within"]
        L.append(f"  {src:<10} {d['n']:>4} {d['mean_true']:>10.2f} {d['mean_pred']:>10.2f} "
                 f"{(f'{sw:.3f}' if not math.isnan(sw) else 'n/a'):>16}")
    L.append("  (a large within-source spearman means it learned craft; if mean pred "
             "tracks mean true across sources but within-source spearman is ~0, "
             "it only learned source.)")
    return "\n".join(L)


# --------------------------------- scoring --------------------------------- #

_MODEL: RidgeModel | None = None


def _init_worker(model_dict: dict):
    global _MODEL
    _MODEL = RidgeModel.from_dict(model_dict)


def seconds_to_tick(score, seconds: float) -> int:
    to_sec = tick_to_seconds_fn(score)
    end = int(score.end())
    if to_sec(end) <= seconds:
        return end
    lo, hi = 0, end
    while lo < hi:                       # to_sec is monotone: bisect
        mid = (lo + hi) // 2
        if to_sec(mid) < seconds:
            lo = mid + 1
        else:
            hi = mid
    return lo


def file_features(path: str, max_seconds: float, file_meta: dict | None = None) -> dict:
    from symusic import Score
    score = Score(path)
    end = int(score.end())
    cut = seconds_to_tick(score, max_seconds) if max_seconds else end
    part = extract_excerpt(score, 0, cut) if cut < end else extract_excerpt(score, 0, end)
    meta = dict(file_meta or {})
    if "duration_seconds" not in meta:
        meta["duration_seconds"] = tick_to_seconds_fn(score)(end)
    if "n_tracks" not in meta:
        meta["n_tracks"] = sum(1 for t in score.tracks if len(t.notes))
    if "n_notes" not in meta:
        meta["n_notes"] = sum(len(t.notes) for t in score.tracks)
    return compute_features(part, meta)


def _score_row(job: tuple[dict, float]) -> dict:
    row, max_seconds = job
    try:
        feats = file_features(row["path"], max_seconds, row)
        if feats["n_notes"] == 0:
            raise ValueError("no notes")
        return {"path": row["path"], "source": source_of(row["path"]),
                "quality": _MODEL.predict_features(feats)}
    except Exception as e:
        return {"path": row["path"], "source": source_of(row["path"]),
                "error": f"{type(e).__name__}: {e}"}


def bucketize(qualities: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = [float(np.quantile(qualities, q)) for q in (0.25, 0.5, 0.75)]
    return np.searchsorted(np.array(edges), qualities, side="right").astype(int), edges


def score_manifest(model: RidgeModel, manifest: Path, out: Path, workers: int = 1,
                   limit: int | None = None, max_seconds: float = 60.0,
                   chunksize: int = 16) -> dict:
    rows = []
    with Path(manifest).expanduser().open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    jobs = [(r, max_seconds) for r in rows]
    results, errors = [], 0
    if workers <= 1:
        _init_worker(model.to_dict())
        it = map(_score_row, jobs)
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                            initargs=(model.to_dict(),))
        it = pool.imap_unordered(_score_row, jobs, chunksize=chunksize)
    for i, res in enumerate(it, 1):
        if "error" in res:
            errors += 1
        else:
            results.append(res)
        if i % 1000 == 0:
            print(f"[score] {i}/{len(jobs)} ({errors} errors)", file=sys.stderr)
    if workers > 1:
        pool.close(); pool.join()
    if not results:
        raise SystemExit("nothing scored")

    q = np.array([r["quality"] for r in results])
    buckets, edges = bucketize(q)
    out = Path(out).expanduser()
    out_path = out if out.suffix == ".jsonl" else out.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r, b in zip(results, buckets):
            f.write(json.dumps({"path": r["path"], "source": r["source"],
                                "quality": round(float(r["quality"]), 4),
                                "q_bucket": int(b)}) + "\n")

    dist: dict[str, dict] = defaultdict(lambda: {"n": 0, "sum": 0.0, "buckets": [0, 0, 0, 0]})
    for r, b in zip(results, buckets):
        d = dist[r["source"]]
        d["n"] += 1; d["sum"] += r["quality"]; d["buckets"][int(b)] += 1
    summary = {"scored": len(results), "errors": errors, "out": str(out_path),
               "bucket_edges": edges,
               "per_source": {s: {"n": d["n"], "mean_quality": d["sum"] / d["n"],
                                  "buckets": d["buckets"]} for s, d in sorted(dist.items())}}
    return summary


def format_score_summary(s: dict) -> str:
    L = [f"scored {s['scored']} files ({s['errors']} errors) -> {s['out']}",
         "bucket edges (quartiles of predicted quality): " +
         ", ".join(f"{e:.2f}" for e in s["bucket_edges"]),
         f"  {'source':<10} {'n':>7} {'mean q':>7}   b0 (worst) .. b3 (best)"]
    for src, d in s["per_source"].items():
        pct = " ".join(f"{100 * b / d['n']:4.0f}%" for b in d["buckets"])
        L.append(f"  {src:<10} {d['n']:>7} {d['mean_quality']:>7.2f}   {pct}")
    return "\n".join(L)


# ----------------------------------- CLI ----------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description="corpus quality predictor")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit ridge on evals/quality/ratings.jsonl")
    f.add_argument("--ratings", default=str(DEFAULT_LOG))
    f.add_argument("--model", default=str(DEFAULT_MODEL))
    f.add_argument("--k", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--alpha", type=float, default=None,
                   help="ridge strength (default: pick from grid by grouped CV)")
    f.add_argument("--no-gbr", action="store_true")
    s = sub.add_parser("score", help="apply the saved model to a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True, help="output path (.jsonl appended if missing)")
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--max-seconds", type=float, default=60.0,
                   help="only the first N seconds of long files are featurized")
    args = p.parse_args(argv)

    if args.cmd == "fit":
        records = load_ratings(Path(args.ratings))
        model, report = fit_predictor(records, k=args.k, seed=args.seed,
                                      alpha=args.alpha, try_gbr=not args.no_gbr)
        model.save(Path(args.model))
        print(format_report(report))
        print(f"saved {args.model}")
    else:
        model = RidgeModel.load(Path(args.model))
        summary = score_manifest(model, Path(args.manifest), Path(args.out),
                                 workers=args.workers, limit=args.limit,
                                 max_seconds=args.max_seconds)
        print(format_score_summary(summary))


if __name__ == "__main__":
    main()
