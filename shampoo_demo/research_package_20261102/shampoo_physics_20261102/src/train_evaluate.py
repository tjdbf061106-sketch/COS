"""Fixed physical-property baselines; no skin/eye labels enter this learner."""
from __future__ import annotations

import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_inputs, read_json, sha256, write_csv, write_json

TARGETS = {"stability": "stable", "viscosity_100": "log10_viscosity_100", "turbidity": "log10p1_turbidity"}


def original_units(target, values):
    values = np.asarray(values, dtype=float)
    if target == "stability":
        return values
    return np.power(10.0, values) - (1 if target == "turbidity" else 0)


def metrics(target, truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if target == "stability":
        label = prediction >= 0.5
        has_both = len(np.unique(truth)) == 2
        return {
            "roc_auc": roc_auc_score(truth, prediction) if has_both else np.nan,
            "average_precision": average_precision_score(truth, prediction) if (truth == 1).any() else np.nan,
            "balanced_accuracy": balanced_accuracy_score(truth, label) if has_both else np.nan,
            "mcc": matthews_corrcoef(truth, label) if has_both else np.nan,
            "brier": brier_score_loss(truth, prediction),
            "recall": label[truth == 1].mean() if (truth == 1).any() else np.nan,
            "specificity": (~label[truth == 0]).mean() if (truth == 0).any() else np.nan,
        }
    return {"mae_log10": mean_absolute_error(truth, prediction),
            "rmse_log10": np.sqrt(mean_squared_error(truth, prediction)),
            "r2_log10": r2_score(truth, prediction) if len(truth) > 1 else np.nan,
            "median_absolute_error_original_units": np.median(abs(original_units(target, truth) - original_units(target, prediction)))}


def build_model(name, target, seed, protocol):
    classification = target == "stability"
    if name == "dummy":
        return DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="median")
    if name == "linear":
        model = LogisticRegression(C=protocol["linear"]["classification_C"], max_iter=5000, random_state=seed) if classification else Ridge(alpha=protocol["linear"]["regression_alpha"])
        return make_pipeline(StandardScaler(), model)
    if name != "extra_trees":
        raise ValueError(name)
    cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
    return cls(random_state=seed, **protocol["extra_trees"])


def make_splits(data, protocol):
    rows = []
    for seed in protocol["repeat_seeds"]:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (_, test) in enumerate(splitter.split(data, data.stable, data.ingredient_combination)):
            for i in test:
                rows.append({"scheme": "combination_holdout", "seed": seed, "fold": fold, "sample_id": int(data.iloc[i].sample_id),
                             "group": data.iloc[i].ingredient_combination})
    splitter = LeaveOneGroupOut()
    for fold, (_, test) in enumerate(splitter.split(data, groups=data.polymer_thickener_group)):
        for i in test:
            rows.append({"scheme": "subsystem_holdout", "seed": protocol["repeat_seeds"][0], "fold": fold,
                         "sample_id": int(data.iloc[i].sample_id), "group": data.iloc[i].polymer_thickener_group})
    return pd.DataFrame(rows)


def lock_inputs():
    paths = ["protocol.json", "data/formulations.csv", "data/feature_names.json", "data/ingredient_registry.csv",
             "src/common.py", "src/prepare_data.py", "src/train_evaluate.py"]
    hashes = {p: sha256(ROOT / p) for p in paths}
    lock_path = ROOT / "training_lock.json"
    if lock_path.exists() and read_json(lock_path)["hashes"] != hashes:
        raise ValueError("Locked training inputs/code changed. Use an explicitly versioned new experiment.")
    write_json(lock_path, {"hashes": hashes, "python": platform.python_version(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__, "source_labels_used_for_physics": False})


def run():
    protocol = read_json(ROOT / "protocol.json")
    data, features = load_inputs()
    lock_inputs()
    splits = make_splits(data, protocol)
    write_csv(ROOT / "results/outer_folds.csv", splits)
    predictions, fold_results = [], []
    for (scheme, seed, fold), held_out in splits.groupby(["scheme", "seed", "fold"], sort=False):
        test_mask = data.sample_id.isin(held_out.sample_id)
        group_col = "ingredient_combination" if scheme == "combination_holdout" else "polymer_thickener_group"
        assert not set(data.loc[test_mask, group_col]) & set(data.loc[~test_mask, group_col])
        for target, column in TARGETS.items():
            eligible = data[column].notna()
            if target != "stability":
                eligible &= data.stable == 1
            train = data.loc[~test_mask & eligible]
            test = data.loc[test_mask & eligible]
            if test.empty:
                continue
            assert len(train) > 1
            for name in protocol["models"]:
                model = build_model(name, target, int(seed) + int(fold), protocol)
                model.fit(train[features], train[column])
                if target == "stability":
                    assert list(model.classes_) == [0, 1]
                    pred = model.predict_proba(test[features])[:, 1]
                else:
                    pred = model.predict(test[features])
                    if target == "turbidity":
                        pred = np.maximum(pred, 0)  # NTU cannot be negative; recorded in model card.
                base = {"scheme": scheme, "seed": int(seed), "fold": int(fold), "target": target, "model": name,
                        "n_train": len(train), "n_test": len(test)}
                fold_results.append({**base, **metrics(target, test[column], pred)})
                for sid, y, yhat in zip(test.sample_id, test[column], pred):
                    predictions.append({**base, "sample_id": int(sid), "observed": float(y), "predicted": float(yhat),
                                        "observed_original": float(original_units(target, y)), "predicted_original": float(original_units(target, yhat))})
        print(f"Completed {scheme} seed={seed} fold={fold}", flush=True)
    oof = pd.DataFrame(predictions)
    write_csv(ROOT / "results/oof_predictions.csv", oof)
    write_csv(ROOT / "results/fold_metrics.csv", pd.DataFrame(fold_results))
    repeat_results = []
    for (scheme, seed, target, name), group in oof.groupby(["scheme", "seed", "target", "model"]):
        assert group.sample_id.is_unique
        repeat_results.append({"scheme": scheme, "seed": seed, "target": target, "model": name, "n": len(group),
                               **metrics(target, group.observed, group.predicted)})
    repeats = pd.DataFrame(repeat_results)
    write_csv(ROOT / "results/repeat_metrics.csv", repeats)
    rows = []
    for (scheme, target, name), group in repeats.groupby(["scheme", "target", "model"]):
        for metric in [c for c in repeats if c not in ["scheme", "target", "model", "seed", "n"]]:
            vals = group[metric].dropna()
            if len(vals):
                rows.append({"scheme": scheme, "target": target, "model": name, "metric": metric, "mean": vals.mean(),
                             "repeat_sd": vals.std(ddof=1) if len(vals) > 1 else np.nan, "n_repeats": len(vals), "n_samples": int(group.n.iloc[0])})
    write_csv(ROOT / "results/metrics_summary.csv", pd.DataFrame(rows))
    for target, column in TARGETS.items():
        eligible = data[column].notna() & ((data.stable == 1) if target != "stability" else True)
        model = build_model(protocol["primary_model"], target, protocol["repeat_seeds"][0], protocol)
        model.fit(data.loc[eligible, features], data.loc[eligible, column])
        path = ROOT / "models" / f"{target}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
        write_json(path.with_suffix(".card.json"), {
            "target": target, "model": protocol["primary_model"], "training_rows": int(eligible.sum()),
            "training_sample_ids": data.loc[eligible, "sample_id"].tolist(), "features": features,
            "model_sha256": sha256(path), "protocol_sha256": sha256(ROOT / "protocol.json"),
            "transform": protocol["targets"][target], "probability_calibration": "Not separately calibrated",
            "postprocessing": "turbidity predictions clipped to nonnegative log10(1+NTU)",
            "intended_use": "Exploratory physical-property prediction for this commercial ingredient library and preparation protocol",
            "not_validated": protocol["prediction_limits"],
        })
    write_json(ROOT / "results/run_summary.json", {
        "status": "completed", "n_formulations": len(data), "n_stable": int(data.stable.sum()),
        "oof_prediction_rows": len(oof), "repeat_metric_rows": len(repeats),
        "primary_model_fixed_before_results": protocol["primary_model"],
        "toxicity_prediction_or_transfer_performed": False, "independent_external_validation_performed": False,
    })


if __name__ == "__main__":
    run()
