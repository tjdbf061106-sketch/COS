"""Shared harness for the eye model-strengthening experiments.

Reuses the audited EPA training primitives (nested CV, threshold policy,
metrics) from the production code. This module only adds the grouping fix,
repeated seeds, alternative representations, ensembling and result caching.
Nothing here retrains or overwrites the published production runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EPA_CODE = Path(r"C:/Users/DS/COS/최종/label 조합 변경/EPA")
if str(EPA_CODE) not in sys.path:
    sys.path.insert(0, str(EPA_CODE))

import epa_experiment_utils as u  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MATS = ROOT / "_mats"
RESULTS.mkdir(exist_ok=True)
MATS.mkdir(exist_ok=True)

ENDPOINT = "eye"
DATA = Path(r"C:/Users/DS/COS/최종/production데이터/eye_vectorized.csv")
COMPONENTS = Path(r"C:/Users/DS/COS/시행착오/2. vectorized/최종/component_level_clean.csv")
SCHEMA = Path(r"C:/Users/DS/COS/시행착오/2. vectorized/최종/feature_schema.json")
BASELINE = Path(
    r"C:/Users/DS/COS/최종/label 조합 변경/EPA/runs/"
    r"epa-production-20261106-no-catboost/eye/bb1eca6656e3324e"
)

MODELS = list(u.MODELS)
CORE = ["sensitivity", "specificity", "MCC", "ROC-AUC"]
SEEDS = [42, 202, 777]
GRID_SIZE = 6
WORKERS = max(1, (os.cpu_count() or 8) - 4)


# --------------------------------------------------------------------------
# data and grouping


def load_eye():
    df, X, cols = u.load_csv(DATA)
    return df, X, cols


def labels_of(df):
    return u.epa_labels(df.final_label)


def vector_key(row):
    return hashlib.sha256(
        np.ascontiguousarray(np.round(row, 6), dtype=np.float64).tobytes()
    ).hexdigest()


class _Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.parent))])


def original_groups(df):
    """Grouping exactly as used by the published production run."""
    return u.connected_groups(df, ENDPOINT)


def merged_groups(df, X):
    """Original grouping, additionally merged over identical feature vectors."""
    uf = _Union(len(df))
    columns = [
        [str(v) for v in original_groups(df)],
        [vector_key(X[i]) for i in range(len(X))],
    ]
    for column in columns:
        seen = {}
        for i, value in enumerate(column):
            if value in seen:
                uf.union(i, seen[value])
            else:
                seen[value] = i
    return uf.labels()


def duplicate_report(df, X):
    """Identical-vector groups and whether the published splits straddle them."""
    base = original_groups(df)
    buckets = {}
    for i in range(len(X)):
        buckets.setdefault(vector_key(X[i]), []).append(i)
    dup = {k: v for k, v in buckets.items() if len(v) > 1}
    folds = u.nested_splits(df.final_label.to_numpy(), base, seed=42)
    fold_of = {i: f for f, split in enumerate(folds) for i in split["test"]}
    straddling = [v for v in dup.values() if len({fold_of[i] for i in v}) > 1]
    classes = df.final_label.to_numpy()
    return {
        "n_rows": int(len(df)),
        "n_unique_vectors": len(buckets),
        "n_duplicate_groups": len(dup),
        "n_rows_in_duplicate_groups": int(sum(len(v) for v in dup.values())),
        "n_straddling_groups": len(straddling),
        "n_rows_straddling": int(sum(len(v) for v in straddling)),
        "n_groups_original": int(len(set(base.tolist()))),
        "n_groups_merged": int(len(set(merged_groups(df, X).tolist()))),
        "straddling_detail": [
            {
                "rows": [int(i) for i in v],
                "outer_folds": [int(fold_of[i]) for i in v],
                "epa_classes": [int(classes[i]) for i in v],
                "label_consistent": bool(len(set(classes[v].tolist())) == 1),
            }
            for v in straddling
        ],
    }


# --------------------------------------------------------------------------
# achievable-performance ceiling


def ceiling_estimate(df, X, label_key):
    """Upper bound implied by identical feature vectors carrying different labels.

    Rows sharing an identical feature vector are indistinguishable to any model
    built on these features, so the best such a model can do is predict each
    group's majority label. This is a necessary bound only: real performance is
    limited by far more than exact duplicates.
    """
    y = labels_of(df)[label_key]
    buckets = {}
    for i in range(len(X)):
        buckets.setdefault(vector_key(X[i]), []).append(i)
    best = np.zeros(len(y), dtype=int)
    irreducible, conflicted = 0, 0
    for rows in buckets.values():
        vals = y[rows]
        majority = int(vals.sum() * 2 >= len(vals))
        best[rows] = majority
        irreducible += int((vals != majority).sum())
        if len(set(vals.tolist())) > 1:
            conflicted += len(rows)
    metrics = u.compute_metrics(y, best.astype(float), best)
    return {
        "label": label_key,
        "irreducible_errors": int(irreducible),
        "irreducible_rate": float(irreducible / len(y)),
        "rows_in_conflicted_vector_groups": int(conflicted),
        "ceiling_MCC": metrics["MCC"],
        "ceiling_sensitivity": metrics["sensitivity"],
        "ceiling_specificity": metrics["specificity"],
        "ceiling_balanced_accuracy": metrics["balanced_accuracy"],
    }


# --------------------------------------------------------------------------
# nested cross-validation, globally load balanced


def reduced_grid(model, size=GRID_SIZE):
    full = u.candidate_grid(model)
    if size >= len(full):
        return full
    idx = np.linspace(0, len(full) - 1, size).round().astype(int)
    return [full[i] for i in dict.fromkeys(idx.tolist())]


def register_matrix(name, X):
    """Persist a feature matrix so workers can memory-map it instead of pickling."""
    path = MATS / f"{name}.npy"
    if not path.exists():
        np.save(path, np.ascontiguousarray(X, dtype=np.float64))
    return name


_CACHE = {}


def _matrix(name):
    if name not in _CACHE:
        _CACHE[name] = np.load(MATS / f"{name}.npy", mmap_mode="r")
    return _CACHE[name]


def _fold_job(job):
    """One (matrix, label, seed, model, outer fold) unit of work."""
    X = np.asarray(_matrix(job["matrix"]))
    y = np.asarray(job["y"])
    model = job["model"]
    grid = reduced_grid(model, job["grid_size"])
    train = np.asarray(job["train"])
    test = np.asarray(job["test"])
    inner = [
        (np.asarray(f["train"]), np.asarray(f["validation"])) for f in job["inner"]
    ]

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
        cut = u.threshold(inner_scores[train], y[train])
        est = u.fit_weighted(
            u.build_estimator(model, candidate["params"]), X[train], y[train]
        )
        outer = u.predict_scores(est, X[test], model)
    return {
        "cell": job["cell"],
        "model": model,
        "fold": job["fold"],
        "candidate_id": candidate["candidate_id"],
        "threshold": float(cut),
        "test": test.tolist(),
        "y_true": y[test].astype(int).tolist(),
        "scores": outer.tolist(),
        "predictions": (outer >= cut).astype(int).tolist(),
    }


def build_jobs(cell, matrix, y, groups, strata, models, seed, grid_size=GRID_SIZE):
    """`cell` is a dict of descriptors; it travels as a sorted JSON string."""
    tag = json.dumps(cell, sort_keys=True, ensure_ascii=False)
    folds = u.nested_splits(strata, groups, seed=seed)
    jobs = []
    for model in models:
        for f, fold in enumerate(folds):
            jobs.append(
                {
                    "cell": tag,
                    "matrix": matrix,
                    "y": np.asarray(y).astype(int).tolist(),
                    "model": model,
                    "fold": f,
                    "seed": seed,
                    "grid_size": grid_size,
                    "train": fold["train"],
                    "test": fold["test"],
                    "inner": fold["inner"],
                }
            )
    return jobs


def execute(jobs, workers=WORKERS, progress=None):
    """Run every job with global load balancing; slowest units start first."""
    cost = {"et": 19.0, "rf": 13.0, "xgb": 6.0, "lgbm": 2.0, "lr": 1.3, "svm": 0.4}
    ordered = sorted(jobs, key=lambda j: -cost.get(j["model"], 1.0))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(_fold_job, ordered), 1):
            out.append(result)
            if progress and (i % progress == 0 or i == len(ordered)):
                print(f"  {i}/{len(ordered)} 완료", flush=True)
    return out


def collect(results, n_rows):
    """Pool each cell's outer-fold predictions into one metric row."""
    rows = []
    keyed = {}
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
            u.compute_metrics(
                np.asarray(p["y_true"]),
                np.asarray(p["scores"]),
                np.asarray(p["predictions"]),
            )
            for p in parts
        ]
        for key in CORE:
            metrics[f"foldstd_{key}"] = float(np.std([f[key] for f in per_fold], ddof=1))
        order = np.argsort(idx)
        header = json.loads(cell)
        rows.append(
            {
                **header,
                "model": model,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                "oof_scores": scores[order].tolist(),
                "oof_y": truth[order].astype(int).tolist(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# saved out-of-fold predictions from the published run (no retraining)


def saved_oof(label):
    table, fold_of = {"y": {}}, {}
    for model in MODELS:
        scores = {}
        for fold in range(5):
            payload = json.loads(
                (BASELINE / label / model / f"fold-{fold}" / "outer.json").read_text(
                    encoding="utf-8"
                )
            )["payload"]
            for row, score, truth in zip(
                payload["row_ids"], payload["scores"], payload["y_true"]
            ):
                scores[int(row)] = float(score)
                fold_of[int(row)] = fold
                table["y"][int(row)] = int(truth)
        table[model] = scores
    order = sorted(table["y"])
    y = np.array([table["y"][i] for i in order])
    S = np.column_stack([[table[m][i] for i in order] for m in MODELS])
    folds = np.array([fold_of[i] for i in order])
    return np.array(order), y, S, folds


def _rank(column):
    return np.argsort(np.argsort(column)) / max(1, len(column) - 1)


def ensemble_from_saved(label):
    """Leak-free rank averaging over the published per-model OOF predictions.

    Averaging has no fitted parameters, so pooled ROC-AUC is unbiased. The
    operating point is still chosen per outer fold from the remaining folds.
    """
    _, y, S, folds = saved_oof(label)
    R = np.column_stack([_rank(S[:, j]) for j in range(S.shape[1])])
    subsets = {
        "6개 전체 평균": list(range(len(MODELS))),
        "LR+LightGBM": [MODELS.index(m) for m in ("lr", "lgbm")],
        "LR+RF+LightGBM": [MODELS.index(m) for m in ("lr", "rf", "lgbm")],
        "LR+XGB+LightGBM": [MODELS.index(m) for m in ("lr", "xgb", "lgbm")],
        "트리 3개(RF/ET/XGB)": [MODELS.index(m) for m in ("rf", "et", "xgb")],
        "SVM 제외 5개": [MODELS.index(m) for m in MODELS if m != "svm"],
    }
    rows = []

    def score_rows(name, blend):
        preds = np.zeros(len(y), dtype=int)
        for f in sorted(set(folds.tolist())):
            held = folds == f
            cut = u.threshold(blend[~held], y[~held])
            preds[held] = (blend[held] >= cut).astype(int)
        m = u.compute_metrics(y, blend, preds)
        rows.append({"구성": name, **{k: m[k] for k in CORE}, "PR-AUC": m["PR-AUC"]})

    for model in MODELS:
        score_rows(f"단일 {model}", S[:, MODELS.index(model)])
    for name, cols in subsets.items():
        score_rows(f"앙상블 {name}", R[:, cols].mean(axis=1))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# alternative representations


def worst_actor_features(df, X):
    """Worst-component pooling instead of concentration-weighted averaging.

    The production matrix averages component fingerprints weighted by
    pct_active_ingredient. Irritation is usually driven by the single most
    aggressive component, which an average dilutes. For rows that carry
    component provenance this builds a max-pooled fingerprint block and takes
    descriptors from the heaviest-weighted component.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
    names = json.loads(SCHEMA.read_text(encoding="utf-8"))["descriptor_names"]
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}

    def encode(smiles):
        text = str(smiles)
        if text in cache:
            return cache[text]
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            cache[text] = None
            return None
        largest = max(
            Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True),
            key=lambda m: (m.GetNumHeavyAtoms(), Descriptors.MolWt(m)),
        )
        dmap = Descriptors.CalcMolDescriptors(largest, missingVal=np.nan)
        desc = np.asarray([dmap[n] for n in names], dtype=float)
        desc[~np.isfinite(desc)] = np.nan
        cache[text] = (generator.GetFingerprintAsNumPy(mol).astype(float), desc)
        return cache[text]

    comp = pd.read_csv(COMPONENTS, low_memory=False)
    positions = {str(k): i for i, k in enumerate(df.formulation_id.astype(str))}
    Xw = X.copy()
    covered = np.zeros(len(df), dtype=bool)
    multi = np.zeros(len(df), dtype=bool)
    for key, frame in comp.groupby("form_key", sort=False):
        i = positions.get(str(key))
        if i is None:
            continue
        encoded = [encode(s) for s in frame.canonical_smiles]
        if any(e is None for e in encoded):
            continue
        fps = np.stack([e[0] for e in encoded])
        descs = np.stack([e[1] for e in encoded])
        weights = pd.to_numeric(frame.pct_active_ingredient, errors="coerce").to_numpy(float)
        if not np.isfinite(weights).any():
            weights = np.ones(len(frame))
        weights = np.nan_to_num(weights, nan=0.0)
        heaviest = int(np.argmax(weights)) if weights.sum() > 0 else 0
        Xw[i, :2048] = fps.max(axis=0)
        Xw[i, 2048:] = descs[heaviest]
        covered[i] = True
        multi[i] = len(frame) > 1
    return Xw, covered, multi


def physchem_indices(cols):
    wanted = [
        "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "FractionCSP3", "HeavyAtomCount", "RingCount",
        "NumAromaticRings", "NumAliphaticRings", "MolMR", "LabuteASA",
        "BalabanJ", "BertzCT", "qed", "NumHeteroatoms", "NHOHCount", "NOCount",
        "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons",
    ]
    available = {c[len("desc_"):]: c for c in cols if c.startswith("desc_")}
    return [cols.index(available[w]) for w in wanted if w in available]


def representation_sets(X, cols):
    fp = [i for i, c in enumerate(cols) if c.startswith("fp_")]
    desc = [i for i, c in enumerate(cols) if c.startswith("desc_")]
    phys = physchem_indices(cols)
    folded = X[:, fp].reshape(len(X), 1024, 2).max(axis=2)
    return {
        "기준 fp2048+desc217": X,
        "descriptor만 217": X[:, desc],
        "fingerprint만 2048": X[:, fp],
        f"물성 소수집합 {len(phys)}": X[:, phys],
        "fp1024 접기+desc": np.hstack([folded, X[:, desc]]),
        f"물성 {len(phys)}+fp1024": np.hstack([X[:, phys], folded]),
    }


# --------------------------------------------------------------------------
# caching


def cached(name, fn, force=False):
    path = RESULTS / f"{name}.json"
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return payload
    value = fn()
    payload = value.to_dict("records") if isinstance(value, pd.DataFrame) else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=u._plain),
        encoding="utf-8",
    )
    return value


def show(df, columns=None, sort=None, ascending=False):
    columns = columns or (
        ["model"] + CORE + ["PR-AUC", "abs_gap", "foldstd_MCC", "foldstd_ROC-AUC"]
    )
    out = df[[c for c in columns if c in df.columns]].copy()
    if sort and sort in out.columns:
        out = out.sort_values(sort, ascending=ascending)
    return out.round(3)


# --------------------------------------------------------------------------
# four-class (ordinal) path: one fit yields both A and B scores

MULTICLASS_MODELS = ["lr", "rf", "et", "lgbm", "xgb"]


def build_multiclass(model, params):
    """Mirror of the production pipeline, retargeted at the four EPA grades."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "lr":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(random_state=42, **params)
    elif model in ("rf", "et"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        est = (RandomForestClassifier if model == "rf" else ExtraTreesClassifier)(
            random_state=42, n_jobs=1, **params
        )
    elif model == "lgbm":
        from lightgbm import LGBMClassifier

        est = LGBMClassifier(
            random_state=42, n_jobs=1, deterministic=True, force_col_wise=True,
            verbosity=-1, objective="multiclass", **params,
        )
    elif model == "xgb":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            random_state=42, n_jobs=1, device="cpu", objective="multi:softprob",
            eval_metric="mlogloss", subsample=1, colsample_bytree=1, **params,
        )
    else:
        raise ValueError(model)
    steps = [(
        "imputer",
        SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
    )]
    if model == "lr":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def _grade_weights(grades):
    grades = np.asarray(grades).astype(int)
    counts = np.bincount(grades, minlength=5).astype(float)
    present = counts[1:5] > 0
    return len(grades) / (present.sum() * counts[grades])


def _cumulative_scores(proba, classes):
    """P(grade <= 2) and P(grade <= 3) from the four-class distribution."""
    classes = np.asarray(classes).astype(int)
    a = proba[:, np.isin(classes, [1, 2])].sum(axis=1)
    b = proba[:, np.isin(classes, [1, 2, 3])].sum(axis=1)
    return {"A": a, "B": b}


def _ordinal_job(job):
    """One (seed, model, outer fold) unit of the four-class experiment."""
    X = np.asarray(_matrix(job["matrix"]))
    grades = np.asarray(job["grades"]).astype(int)
    labels = {"A": (grades <= 2).astype(int), "B": (grades <= 3).astype(int)}
    model = job["model"]
    grid = reduced_grid(model, job["grid_size"])
    train, test = np.asarray(job["train"]), np.asarray(job["test"])
    inner = [(np.asarray(f["train"]), np.asarray(f["validation"]))
             for f in job["inner"]]

    def fit(params, rows):
        # XGBoost requires classes 0..k-1; EPA grades run 1..4.
        est = build_multiclass(model, params)
        with u.threadpool_limits(limits=1):
            est.fit(
                X[rows], grades[rows] - 1,
                model__sample_weight=_grade_weights(grades[rows]),
            )
        return est

    def inner_oof(params):
        scores = {k: np.full(len(grades), np.nan) for k in labels}
        aucs = []
        for tr, va in inner:
            est = fit(params, tr)
            with u.threadpool_limits(limits=1):
                proba = est.predict_proba(X[va])
            cum = _cumulative_scores(proba, est[-1].classes_ + 1)
            for k in labels:
                scores[k][va] = cum[k]
            aucs.append(np.mean([
                u.roc_auc_score(labels[k][va], cum[k]) for k in labels
            ]))
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
        cum = _cumulative_scores(proba, est[-1].classes_ + 1)
    out = []
    for key in labels:
        cut = u.threshold(inner_scores[key][train], labels[key][train])
        out.append({
            "cell": json.dumps(
                {"approach": "4-class 순서형", "label": key, "seed": job["seed"]},
                sort_keys=True, ensure_ascii=False,
            ),
            "model": model,
            "fold": job["fold"],
            "candidate_id": candidate["candidate_id"],
            "threshold": float(cut),
            "test": test.tolist(),
            "y_true": labels[key][test].astype(int).tolist(),
            "scores": cum[key].tolist(),
            "predictions": (cum[key] >= cut).astype(int).tolist(),
        })
    return out


def build_ordinal_jobs(matrix, grades, groups, models, seed, grid_size=GRID_SIZE):
    folds = u.nested_splits(grades, groups, seed=seed)
    return [
        {
            "matrix": matrix,
            "grades": np.asarray(grades).astype(int).tolist(),
            "model": model,
            "fold": f,
            "seed": seed,
            "grid_size": grid_size,
            "train": fold["train"],
            "test": fold["test"],
            "inner": fold["inner"],
        }
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
