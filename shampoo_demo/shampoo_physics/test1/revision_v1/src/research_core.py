"""Shared evaluation and grouped splitting; positive class is always stable."""
from pathlib import Path
import hashlib
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["Sensitivity", "Specificity", "MCC", "ROC-AUC"]
TARGETS = {"stability": "stable", "viscosity_100": "log10_viscosity_100", "turbidity": "log10p1_turbidity"}


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classification_metrics(y, score, mask=None):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 1: raise ValueError("Paired one-dimensional labels and scores required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or not ((0 <= score) & (score <= 1)).all():
        raise ValueError("Binary stable labels and finite [0,1] scores required")
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != y.shape: raise ValueError("Mask shape mismatch")
    selected_y, selected_score = y[mask], score[mask]
    pred = selected_score >= 0.5
    tn, fp, fn, tp = confusion_matrix(selected_y, pred, labels=[0, 1]).ravel() if mask.any() else (0, 0, 0, 0)
    both = len(np.unique(selected_y)) == 2
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"n_total": len(y), "n_accepted": int(mask.sum()), "n_stable_accepted": int(tp + fn),
            "n_unstable_accepted": int(tn + fp), "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
            "coverage": float(mask.mean()) if len(mask) else np.nan,
            "stable_coverage": float(mask[y == 1].mean()) if (y == 1).any() else np.nan,
            "unstable_coverage": float(mask[y == 0].mean()) if (y == 0).any() else np.nan,
            "Sensitivity": sens, "Specificity": spec,
            "MCC": float(matthews_corrcoef(selected_y, pred)) if both else np.nan,
            "ROC-AUC": float(roc_auc_score(selected_y, selected_score)) if both else np.nan,
            "selective_error": float((pred != selected_y).mean()) if mask.any() else np.nan,
            "sensitivity_specificity_gap": abs(sens - spec),
            "metrics_status": "both_classes" if both else "one_class_or_empty"}


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Finite paired regression observations required")
    return {"n": len(y), "MAE_log": mean_absolute_error(y, pred) if len(y) else np.nan,
            "RMSE_log": math.sqrt(mean_squared_error(y, pred)) if len(y) else np.nan,
            "R2_log": r2_score(y, pred) if len(y) > 1 and np.var(y) > 0 else np.nan}


def build_model(name, target, seed, protocol):
    is_classification = target == "stability"
    if name == "linear":
        settings = protocol["model_settings"]["linear"]
        estimator = LogisticRegression(C=settings["C"], max_iter=settings["max_iter"], random_state=seed) if is_classification else Ridge(alpha=settings["ridge_alpha"])
        return make_pipeline(StandardScaler(), estimator)
    if name == "extra_trees":
        cls = ExtraTreesClassifier if is_classification else ExtraTreesRegressor
        return cls(random_state=seed, **protocol["model_settings"]["extra_trees"])
    raise ValueError(name)


def linked_groups(frame, feature_columns, require_experimental_ids=True):
    """Union overlapping recipe, manufacturing batch, test and composition identifiers."""
    required = ["record_id", "source_id", "formulation_id", "preparation_batch_id", "measurement_event_id"]
    if require_experimental_ids:
        if not set(required).issubset(frame): raise ValueError("Experimental identifiers missing")
        for col in required:
            if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
                raise ValueError(f"Unresolved identifier: {col}")
    if "record_id" in frame and not frame.record_id.is_unique: raise ValueError("record_id must be unique")
    x = frame[feature_columns].to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or (x.sum(axis=1) > 100 + 1e-8).any():
        raise ValueError("Invalid stock percentages")
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    identifiers = [c for c in ["formulation_id", "preparation_batch_id", "measurement_event_id", "related_group_id"] if c in frame]
    seen = {}
    for i, (_, row) in enumerate(frame.iterrows()):
        keys = [("composition", tuple(np.round(x[i], 10)))]
        for col in identifiers:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                keys.append((col, str(row.get("source_id", "")), str(value).strip()))
        for key in keys:
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    return pd.Series([f"component_{find(i):06d}" for i in range(len(frame))], index=frame.index)


def split_group_sets(groups, test_fraction, seed):
    values = np.array(sorted(set(groups)), dtype=object)
    if len(values) < 2: raise ValueError("At least two disjoint groups required")
    if not 0 < test_fraction < 1: raise ValueError("Fraction must be between zero and one")
    order = np.random.default_rng(seed).permutation(values)
    n_test = min(len(order) - 1, max(1, math.ceil(len(order) * test_fraction)))
    return set(order[n_test:]), set(order[:n_test])


def wilson_interval(successes, n, confidence=0.95):
    if not 0 < confidence < 1 or n < 0 or successes < 0 or successes > n: raise ValueError("Invalid count/confidence")
    if n == 0: return (np.nan, np.nan)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center-half), min(1.0, center+half)


def assess_registration(classification, regressions, criteria, independent_units_verified=False):
    required = ["sensitivity_lower_bound_min", "specificity_lower_bound_min", "max_sensitivity_specificity_gap",
                "coverage_min", "min_independent_stable_units", "min_independent_unstable_units",
                "viscosity_log_mae_max", "turbidity_log_mae_max", "min_viscosity_test_units", "min_turbidity_test_units"]
    missing = [key for key in required if criteria.get(key) is None]
    if missing: return {"decision": "HOLD", "reasons": ["Use-case acceptance targets are unset"], "missing": missing}
    if not independent_units_verified:
        return {"decision": "HOLD", "reasons": ["Independent evaluation units/prospective evidence are unresolved"]}
    if classification["metrics_status"] != "both_classes":
        return {"decision": "HOLD", "reasons": ["Both stability classes are required"]}
    confidence = criteria.get("confidence_level", 0.95)
    sl = wilson_interval(classification["TP"], classification["n_stable_accepted"], confidence)[0]
    tl = wilson_interval(classification["TN"], classification["n_unstable_accepted"], confidence)[0]
    checks = {"sensitivity_lower_bound": sl >= criteria["sensitivity_lower_bound_min"],
              "specificity_lower_bound": tl >= criteria["specificity_lower_bound_min"],
              "class_balance": classification["sensitivity_specificity_gap"] <= criteria["max_sensitivity_specificity_gap"],
              "coverage": classification["coverage"] >= criteria["coverage_min"],
              "stable_units": classification["n_stable_accepted"] >= criteria["min_independent_stable_units"],
              "unstable_units": classification["n_unstable_accepted"] >= criteria["min_independent_unstable_units"]}
    for target, prefix in [("viscosity_100", "viscosity"), ("turbidity", "turbidity")]:
        r = regressions.get(target, {})
        checks[prefix + "_precision"] = bool(np.isfinite(r.get("MAE_log", np.nan)) and r["MAE_log"] <= criteria[prefix + "_log_mae_max"])
        checks[prefix + "_units"] = r.get("n", 0) >= criteria["min_" + prefix + "_test_units"]
    return {"decision": "PASS_LIMITED_SCOPE" if all(checks.values()) else "HOLD", "checks": checks,
            "sensitivity_lower_bound": sl, "specificity_lower_bound": tl,
            "note": "Wilson intervals assume independent units; repeated readings cannot be counted as independent samples."}
