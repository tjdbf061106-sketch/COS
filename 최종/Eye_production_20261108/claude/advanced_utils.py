"""Second round of eye experiments (07-10).

Builds on the first-round harness in ../experiment_utils.py and adds:
  * out-of-fold score retention, so ensembling and threshold policies can be
    studied afterwards without retraining,
  * post-hoc probability calibration fitted on inner out-of-fold scores,
  * a combined representation that keeps both the concentration-weighted and
    the worst-component view of a mixture,
  * alternative fingerprint families computed from component provenance.

Nothing here retrains or overwrites the published production runs, and the
first-round results in ../results are read-only here.
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for p in (str(HERE), str(PARENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiment_utils as E  # noqa: E402  (first-round harness)
import epa_experiment_utils as u  # noqa: E402

RESULTS = HERE / "results"
MATS = HERE / "_mats"
RESULTS.mkdir(exist_ok=True)
MATS.mkdir(exist_ok=True)

MODELS = E.MODELS
CORE = E.CORE
SEEDS = E.SEEDS
GRID_SIZE = E.GRID_SIZE
WORKERS = E.WORKERS


# --------------------------------------------------------------------------
# matrices


def register_matrix(name, X):
    path = MATS / f"{name}.npy"
    if not path.exists():
        np.save(path, np.ascontiguousarray(X, dtype=np.float64))
    return name


_CACHE = {}


def _matrix(name):
    if name not in _CACHE:
        _CACHE[name] = np.load(MATS / f"{name}.npy", mmap_mode="r")
    return _CACHE[name]


def fold_block(block, width=1024):
    """Fold a 2048-bit block down to `width` by OR-ing the two halves."""
    return block.reshape(len(block), width, block.shape[1] // width).max(axis=2)


def combined_representation(df, X):
    """Keep both views of a mixture side by side instead of choosing one.

    Experiment 05 showed the concentration-weighted average helps label A and
    the worst-component pooling helps label B. Here both are supplied and the
    model is left to use whichever carries signal. Each fingerprint block is
    folded to 1024 bits first so the combined width stays near the baseline.
    """
    Xw, covered, multi = E.worst_actor_features(df, X)
    weighted_fp = fold_block(np.asarray(X[:, :2048]))
    worst_fp = fold_block(np.asarray(Xw[:, :2048]))
    desc = np.asarray(X[:, 2048:])
    combined = np.hstack([weighted_fp, worst_fp, desc])
    return combined, covered, multi


FAMILIES = ("morgan", "maccs", "avalon", "atompair", "torsion", "rdkit")


def fingerprint_families(df, families=FAMILIES):
    """Concentration-weighted fingerprints in several molecular representations.

    Uses the same component provenance and the same pct_active_ingredient
    weighting as the production matrix, so the only thing that changes is the
    fingerprint family. Rows without component provenance cannot be rebuilt and
    are reported as uncovered.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem import rdFingerprintGenerator as fg

    RDLogger.DisableLog("rdApp.*")
    gens = {
        "morgan": fg.GetMorganGenerator(radius=2, fpSize=2048),
        "atompair": fg.GetAtomPairGenerator(fpSize=2048),
        "torsion": fg.GetTopologicalTorsionGenerator(fpSize=2048),
        "rdkit": fg.GetRDKitFPGenerator(fpSize=2048),
    }
    widths = {"maccs": 167, "avalon": 512}
    cache = {}

    def encode(smiles):
        text = str(smiles)
        if text in cache:
            return cache[text]
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            cache[text] = None
            return None
        out = {}
        for name in families:
            if name in gens:
                out[name] = gens[name].GetFingerprintAsNumPy(mol).astype(float)
            elif name == "maccs":
                out[name] = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=float)
            elif name == "avalon":
                out[name] = np.asarray(
                    pyAvalonTools.GetAvalonFP(mol, nBits=512), dtype=float
                )
        cache[text] = out
        return out

    comp = pd.read_csv(E.COMPONENTS, low_memory=False)
    positions = {str(k): i for i, k in enumerate(df.formulation_id.astype(str))}
    sizes = {n: (gens[n].GetFingerprintAsNumPy(Chem.MolFromSmiles("CC")).shape[0]
                 if n in gens else widths[n]) for n in families}
    blocks = {n: np.zeros((len(df), sizes[n])) for n in families}
    covered = np.zeros(len(df), dtype=bool)

    for key, frame in comp.groupby("form_key", sort=False):
        i = positions.get(str(key))
        if i is None:
            continue
        encoded = [encode(s) for s in frame.canonical_smiles]
        if any(e is None for e in encoded):
            continue
        raw = pd.to_numeric(frame.pct_active_ingredient, errors="coerce").to_numpy(float)
        if np.isfinite(raw).all() and raw.sum() > 0:
            weights = raw / raw.sum()
        else:
            weights = np.full(len(frame), 1 / len(frame))
        for name in families:
            stack = np.stack([e[name] for e in encoded])
            blocks[name][i] = (stack * weights[:, None]).sum(axis=0)
        covered[i] = True
    return blocks, covered


# --------------------------------------------------------------------------
# calibration


def _calibrate(kind, inner_scores, inner_y, outer_scores):
    """Map raw scores onto a common probability scale using inner OOF only."""
    mask = np.isfinite(inner_scores)
    s, y = inner_scores[mask].reshape(-1, 1), inner_y[mask]
    if kind == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(s, y)
        return model.predict_proba(outer_scores.reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(s.ravel(), y)
        return model.predict(outer_scores)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# jobs


def _job(job):
    """One (matrix, label, seed, model, fold) unit; keeps out-of-fold scores."""
    X = np.asarray(_matrix(job["matrix"]))
    y = np.asarray(job["y"])
    model = job["model"]
    grid = E.reduced_grid(model, job["grid_size"])
    train, test = np.asarray(job["train"]), np.asarray(job["test"])
    inner = [(np.asarray(f["train"]), np.asarray(f["validation"]))
             for f in job["inner"]]

    def inner_oof(params):
        scores = np.full(len(y), np.nan)
        aucs = []
        for tr, va in inner:
            est = u.fit_weighted(u.build_estimator(model, params), X[tr], y[tr])
            s = u.predict_scores(est, X[va], model)
            scores[va] = s
            aucs.append(float(u.roc_auc_score(y[va], s)))
        return scores, float(np.mean(aucs))

    best, errors = None, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rank, candidate in enumerate(grid):
            try:
                scores, mean_auc = inner_oof(candidate["params"])
            except Exception as exc:
                errors.append(f"{candidate['candidate_id']}: {type(exc).__name__}: {exc}")
                continue
            key = (mean_auc, -rank)
            if best is None or key > best[0]:
                best = (key, candidate, scores)
        if best is None:
            raise RuntimeError(f"No usable candidate for {model}; {errors[:2]}")
        _, candidate, inner_scores = best
        est = u.fit_weighted(u.build_estimator(model, candidate["params"]), X[train], y[train])
        outer = u.predict_scores(est, X[test], model)

        kind = job.get("calibrate")
        if kind:
            outer = _calibrate(kind, inner_scores[train], y[train], outer)
            inner_for_cut = _calibrate(kind, inner_scores[train], y[train],
                                       inner_scores[train])
        else:
            inner_for_cut = inner_scores[train]
        cut = u.threshold(inner_for_cut, y[train])

    return {
        "cell": job["cell"],
        "model": model,
        "fold": job["fold"],
        "candidate_id": candidate["candidate_id"],
        "threshold": float(cut),
        "test": test.tolist(),
        "y_true": y[test].astype(int).tolist(),
        "scores": np.asarray(outer, dtype=float).tolist(),
        "predictions": (np.asarray(outer) >= cut).astype(int).tolist(),
    }


def build_jobs(cell, matrix, y, groups, strata, models, seed,
               grid_size=GRID_SIZE, calibrate=None, n_rows=None):
    tag = json.dumps(cell, sort_keys=True, ensure_ascii=False)
    folds = u.nested_splits(strata, groups, seed=seed)
    return [
        {
            "cell": tag, "matrix": matrix,
            "y": np.asarray(y).astype(int).tolist(),
            "model": model, "fold": f, "seed": seed,
            "grid_size": grid_size, "calibrate": calibrate,
            "train": fold["train"], "test": fold["test"], "inner": fold["inner"],
        }
        for model in models
        for f, fold in enumerate(folds)
    ]


def execute(jobs, workers=WORKERS, progress=None, fn=None):
    cost = {"et": 19.0, "rf": 13.0, "xgb": 6.0, "lgbm": 2.0, "lr": 1.3, "svm": 0.4}
    ordered = sorted(jobs, key=lambda j: -cost.get(j["model"], 1.0))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(fn or _job, ordered), 1):
            out.append(r)
            if progress and (i % progress == 0 or i == len(ordered)):
                print(f"  {i}/{len(ordered)} 완료", flush=True)
    return out


def collect(results, n_rows, keep_oof=False):
    rows, keyed = [], {}
    for r in results:
        keyed.setdefault((r["cell"], r["model"]), []).append(r)
    for (cell, model), parts in keyed.items():
        parts = sorted(parts, key=lambda p: p["fold"])
        idx = np.concatenate([p["test"] for p in parts])
        truth = np.concatenate([p["y_true"] for p in parts])
        scores = np.concatenate([p["scores"] for p in parts])
        preds = np.concatenate([p["predictions"] for p in parts])
        if sorted(idx.tolist()) != list(range(n_rows)):
            raise ValueError(f"Outer folds do not tile the evaluation set: {cell}")
        metrics = u.compute_metrics(truth, scores, preds)
        per_fold = [
            u.compute_metrics(np.asarray(p["y_true"]), np.asarray(p["scores"]),
                              np.asarray(p["predictions"]))
            for p in parts
        ]
        for key in CORE:
            metrics[f"foldstd_{key}"] = float(np.std([f[key] for f in per_fold], ddof=1))
        order = np.argsort(idx)
        row = {**json.loads(cell), "model": model,
               **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
               "confusion_matrix": json.dumps(metrics["confusion_matrix"])}
        if keep_oof:
            row["oof_scores"] = scores[order].tolist()
            row["oof_y"] = truth[order].astype(int).tolist()
            row["oof_fold"] = np.concatenate(
                [[p["fold"]] * len(p["test"]) for p in parts]
            )[order].tolist()
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ordinal path with out-of-fold retention


def _ordinal_job(job):
    X = np.asarray(_matrix(job["matrix"]))
    grades = np.asarray(job["grades"]).astype(int)
    labels = {"A": (grades <= 2).astype(int), "B": (grades <= 3).astype(int)}
    model = job["model"]
    grid = E.reduced_grid(model, job["grid_size"])
    train, test = np.asarray(job["train"]), np.asarray(job["test"])
    inner = [(np.asarray(f["train"]), np.asarray(f["validation"]))
             for f in job["inner"]]

    def fit(params, rows):
        est = E.build_multiclass(model, params)
        with u.threadpool_limits(limits=1):
            est.fit(X[rows], grades[rows] - 1,
                    model__sample_weight=E._grade_weights(grades[rows]))
        return est

    def inner_oof(params):
        scores = {k: np.full(len(grades), np.nan) for k in labels}
        aucs = []
        for tr, va in inner:
            est = fit(params, tr)
            with u.threadpool_limits(limits=1):
                proba = est.predict_proba(X[va])
            cum = E._cumulative_scores(proba, est[-1].classes_ + 1)
            for k in labels:
                scores[k][va] = cum[k]
            aucs.append(np.mean([u.roc_auc_score(labels[k][va], cum[k]) for k in labels]))
        return scores, float(np.mean(aucs))

    best, errors = None, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rank, candidate in enumerate(grid):
            try:
                scores, mean_auc = inner_oof(candidate["params"])
            except Exception as exc:
                errors.append(f"{candidate['candidate_id']}: {type(exc).__name__}: {exc}")
                continue
            key = (mean_auc, -rank)
            if best is None or key > best[0]:
                best = (key, candidate, scores)
        if best is None:
            raise RuntimeError(f"No usable candidate for {model}; {errors[:2]}")
        _, candidate, inner_scores = best
        est = fit(candidate["params"], train)
        with u.threadpool_limits(limits=1):
            proba = est.predict_proba(X[test])
        cum = E._cumulative_scores(proba, est[-1].classes_ + 1)

    out = []
    for key in labels:
        cut = u.threshold(inner_scores[key][train], labels[key][train])
        out.append({
            "cell": json.dumps({**json.loads(job["cell"]), "label": key},
                               sort_keys=True, ensure_ascii=False),
            "model": f"{model}-순서형", "fold": job["fold"],
            "candidate_id": candidate["candidate_id"], "threshold": float(cut),
            "test": test.tolist(),
            "y_true": labels[key][test].astype(int).tolist(),
            "scores": cum[key].tolist(),
            "predictions": (cum[key] >= cut).astype(int).tolist(),
        })
    return out


def build_ordinal_jobs(cell, matrix, grades, groups, models, seed, grid_size=GRID_SIZE):
    tag = json.dumps(cell, sort_keys=True, ensure_ascii=False)
    folds = u.nested_splits(grades, groups, seed=seed)
    return [
        {"cell": tag, "matrix": matrix,
         "grades": np.asarray(grades).astype(int).tolist(),
         "model": model, "fold": f, "seed": seed, "grid_size": grid_size,
         "train": fold["train"], "test": fold["test"], "inner": fold["inner"]}
        for model in models
        for f, fold in enumerate(folds)
    ]


def execute_ordinal(jobs, workers=WORKERS, progress=None):
    cost = {"et": 19.0, "rf": 13.0, "xgb": 6.0, "lgbm": 2.0, "lr": 1.3}
    ordered = sorted(jobs, key=lambda j: -cost.get(j["model"], 1.0))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, pair in enumerate(pool.map(_ordinal_job, ordered), 1):
            out.extend(pair)
            if progress and (i % progress == 0 or i == len(ordered)):
                print(f"  {i}/{len(ordered)} 완료", flush=True)
    return out


# --------------------------------------------------------------------------
# ensembling over retained out-of-fold scores


def rank_norm(column):
    return np.argsort(np.argsort(column)) / max(1, len(column) - 1)


def blend(table, label, seed, members):
    """Rank-average a set of (model) members and score the blend leak-free."""
    sub = table[(table.label == label) & (table.seed == seed)]
    sub = sub[sub.model.isin(members)]
    if len(sub) != len(members):
        return None
    y = np.asarray(sub.iloc[0]["oof_y"])
    folds = np.asarray(sub.iloc[0]["oof_fold"])
    R = np.column_stack([rank_norm(np.asarray(r["oof_scores"])) for _, r in sub.iterrows()])
    score = R.mean(axis=1)
    preds = np.zeros(len(y), dtype=int)
    for f in sorted(set(folds.tolist())):
        held = folds == f
        cut = u.threshold(score[~held], y[~held])
        preds[held] = (score[held] >= cut).astype(int)
    return u.compute_metrics(y, score, preds)
