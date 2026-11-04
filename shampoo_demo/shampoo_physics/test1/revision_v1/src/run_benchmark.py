"""Frozen representation comparison and retrospective new-material onboarding experiment."""
import argparse
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from research_core import ROOT, TARGETS, build_model, classification_metrics, regression_metrics, read_json, write_json, sha, split_group_sets


def load():
    provenance = read_json(ROOT / "source_provenance.json")
    for name, expected in provenance["input_sha256"].items():
        assert sha(ROOT / "inputs" / name) == expected, name
    data = pd.read_csv(ROOT / "inputs/formulations.csv", float_precision="round_trip").set_index("sample_id")
    sets = read_json(ROOT / "inputs/feature_sets.json")
    source = pd.read_csv(ROOT / "inputs/features_source_reported_812.csv", float_precision="round_trip").set_index("sample_id")
    supplier = pd.read_csv(ROOT / "inputs/features_pk45_supplier_sensitivity_812.csv", float_precision="round_trip").set_index("sample_id")
    assert data.index.equals(source.index) and data.index.equals(supplier.index)
    np.testing.assert_allclose(source[sets["stock18"]], data[sets["stock18"]], rtol=0, atol=1e-12)
    matrices = {"stock18": data[sets["stock18"]], "physics_core61": source[sets["physics_core"]],
                "physics_core61_pk45": supplier[sets["physics_core"]]}
    for x in matrices.values(): assert np.isfinite(x).all().all()
    return data, matrices, sets


def prepare_onboarding_splits(data, features, protocol):
    rows = []
    for ingredient in features:
        salt = int.from_bytes(hashlib.sha256(ingredient.encode()).digest()[:4], "little")
        for seed in protocol["repeat_seeds"]:
            present = data[data[ingredient] > 0]
            absent = data[data[ingredient] == 0]
            new_pool, new_test = split_group_sets(present.ingredient_combination, 0.3, seed + salt)
            old_fit, old_test = split_group_sets(absent.ingredient_combination, 0.2, seed + salt + 1)
            order = np.random.default_rng(seed + salt + 2).permutation(sorted(new_pool)).tolist()
            ranks = {group: i + 1 for i, group in enumerate(order)}
            for sid, row in data.iterrows():
                g = row.ingredient_combination
                if row[ingredient] > 0:
                    partition = "new_final_test" if g in new_test else "adaptation_pool"
                else:
                    partition = "old_retention_test" if g in old_test else "old_training"
                rows.append({"held_ingredient": ingredient, "seed": seed, "sample_id": int(sid),
                             "partition": partition, "group": g, "adaptation_group_rank": ranks.get(g, 0)})
            assert not new_pool & new_test and not old_fit & old_test
    split = pd.DataFrame(rows)
    for _, g in split.groupby(["held_ingredient", "seed"]):
        assert len(g) == 812 and g.sample_id.is_unique
        assert g.groupby("group").partition.nunique().eq(1).all()
    return split


def freeze(split, protocol):
    split_path = ROOT / "results/onboarding_splits.csv"
    if split_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(split_path), split)
    else:
        split.to_csv(split_path, index=False, encoding="utf-8-sig")
    paths = [ROOT / "config/research_protocol.json", ROOT / "src/research_core.py", Path(__file__).resolve(), split_path]
    paths += sorted((ROOT / "inputs").iterdir())
    lock = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    lock_path = ROOT / "config/analysis_lock.json"
    if lock_path.exists():
        assert read_json(lock_path)["hashes"] == lock, "Frozen analysis changed: start a new experiment version"
    else:
        write_json(lock_path, {"hashes": lock, "created_before_predictions": True,
                               "note": "Retrospective internal benchmark; data and prior results were already available."})


def fit_predict(data, matrix, train_ids, test_ids, target, name, seed, protocol):
    column = TARGETS[target]
    eligible = data[column].notna()
    if target != "stability": eligible &= data.stable.eq(1)
    train_ids = [sid for sid in train_ids if eligible.loc[sid]]
    test_ids = [sid for sid in test_ids if eligible.loc[sid]]
    assert not set(train_ids) & set(test_ids)
    if len(train_ids) < 2: raise ValueError("Insufficient training rows")
    if target == "stability" and data.loc[train_ids, column].nunique() != 2:
        raise ValueError("Both training classes required")
    model = build_model(name, target, seed, protocol)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(matrix.loc[train_ids], data.loc[train_ids, column])
    if not test_ids: return [], model
    values = model.predict_proba(matrix.loc[test_ids])[:, 1] if target == "stability" else model.predict(matrix.loc[test_ids])
    if target == "turbidity": values = np.maximum(values, 0)
    assert np.isfinite(values).all()
    records = [{"sample_id": int(sid), "observed": float(data.loc[sid, column]), "predicted": float(p),
                "n_train": len(train_ids)} for sid, p in zip(test_ids, values)]
    return records, model


def evaluate(pred, keys):
    rows = []
    for values, g in pred.groupby(keys):
        if not isinstance(values, tuple): values = (values,)
        info = dict(zip(keys, values))
        m = classification_metrics(g.observed, g.predicted) if info["target"] == "stability" else regression_metrics(g.observed, g.predicted)
        rows.append({**info, **m})
    return pd.DataFrame(rows)


def representations(data, matrices, protocol):
    destination = ROOT / "results/representation_predictions.csv.gz"
    if destination.exists(): raise FileExistsError("Do not overwrite an existing experiment")
    folds = pd.read_csv(ROOT / "inputs/outer_folds.csv").query("scheme == 'combination_holdout'")
    results = []
    started = time.monotonic()
    for (seed, fold), test in folds.groupby(["seed", "fold"]):
        test_ids = test.sample_id.tolist()
        train_ids = data.index[~data.index.isin(test_ids)].tolist()
        assert not set(data.loc[train_ids].ingredient_combination) & set(data.loc[test_ids].ingredient_combination)
        for feature, x in matrices.items():
            for model in ["linear", "extra_trees"]:
                for target in TARGETS:
                    records, _ = fit_predict(data, x, train_ids, test_ids, target, model, int(seed + fold), protocol)
                    for record in records:
                        results.append({"seed": seed, "fold": fold, "features": feature, "model": model, "target": target, **record})
        print(f"representation seed={seed} fold={fold} elapsed={time.monotonic()-started:.1f}s", flush=True)
    pred = pd.DataFrame(results)
    pred.to_csv(destination, index=False, compression="gzip")
    evaluate(pred, ["seed", "fold", "features", "model", "target"]).to_csv(ROOT / "results/representation_fold_metrics.csv", index=False, encoding="utf-8-sig")
    evaluate(pred, ["seed", "features", "model", "target"]).to_csv(ROOT / "results/representation_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    selective = []
    for (seed, features, model), group in pred[pred.target == "stability"].groupby(["seed", "features", "model"]):
        assert len(group) == 812 and group.sample_id.is_unique
        for margin in protocol["selective_prediction"]["confidence_margins"]:
            accepted = abs(group.predicted.to_numpy() - 0.5) >= margin
            selective.append({"seed": seed, "features": features, "model": model, "confidence_margin": margin,
                              **classification_metrics(group.observed, group.predicted, accepted)})
    pd.DataFrame(selective).to_csv(ROOT / "results/selective_prediction_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(ROOT / "results/representation_run.json", {"status": "PASS", "fits": 270,
               "prediction_rows": len(pred), "elapsed_seconds": time.monotonic()-started,
               "model_selection_performed": False, "threshold_selection_performed": False,
               "external_validation": False})


def onboarding(data, matrices, splits, protocol):
    destination = ROOT / "results/onboarding_predictions.csv.gz"
    if destination.exists(): raise FileExistsError("Do not overwrite an existing experiment")
    results, allocation, fits = [], [], 0
    started = time.monotonic()
    for (ingredient, seed), split in splits.groupby(["held_ingredient", "seed"]):
        old_train = split[split.partition == "old_training"].sample_id.tolist()
        pool = split[split.partition == "adaptation_pool"]
        tests = split[split.partition.isin(["new_final_test", "old_retention_test"])]
        test_ids = tests.sample_id.tolist()
        cohort = tests.set_index("sample_id").partition.to_dict()
        for budget in [0, 4, 12, "all"]:
            added = pool if budget == "all" else pool[pool.adaptation_group_rank <= budget]
            train_ids = old_train + added.sample_id.tolist()
            assert not set(train_ids) & set(test_ids)
            assert not set(data.loc[train_ids].ingredient_combination) & set(data.loc[test_ids].ingredient_combination)
            if budget == 0: assert data.loc[train_ids, ingredient].eq(0).all()
            else: assert len(added) > 0 and data.loc[added.sample_id, ingredient].gt(0).all()
            allocation.append({"held_ingredient": ingredient, "seed": int(seed), "budget": str(budget),
                               "adaptation_groups": int(added.group.nunique()), "adaptation_recipes": len(added),
                               "adaptation_stable": int(data.loc[added.sample_id, "stable"].sum()),
                               "old_training_recipes": len(old_train), "new_test_recipes": int((tests.partition == "new_final_test").sum()),
                               "old_retention_recipes": int((tests.partition == "old_retention_test").sum())})
            for feature in ["stock18", "physics_core61"]:
                for target in TARGETS:
                    records, _ = fit_predict(data, matrices[feature], train_ids, test_ids, target, "linear", int(seed), protocol)
                    fits += 1
                    for record in records:
                        results.append({"held_ingredient": ingredient, "seed": seed, "budget": str(budget),
                                        "features": feature, "model": "linear", "target": target,
                                        "cohort": cohort[record["sample_id"]], **record})
        print(f"onboarding {ingredient} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
    pred = pd.DataFrame(results)
    pred.to_csv(destination, index=False, compression="gzip")
    pd.DataFrame(allocation).to_csv(ROOT / "results/onboarding_allocation.csv", index=False, encoding="utf-8-sig")
    evaluate(pred, ["held_ingredient", "seed", "budget", "features", "model", "target", "cohort"]).to_csv(
        ROOT / "results/onboarding_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(ROOT / "results/onboarding_run.json", {"status": "PASS", "fits": fits, "prediction_rows": len(pred),
               "elapsed_seconds": time.monotonic()-started, "materials": 18, "independent_external_datasets": 0,
               "real_new_material_experiments": 0, "batch_independence": "unresolved", "model_selection_performed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["freeze", "representations", "onboarding", "all"], default="all")
    args = parser.parse_args()
    data, matrices, sets = load()
    protocol = read_json(ROOT / "config/research_protocol.json")
    split = prepare_onboarding_splits(data, sets["stock18"], protocol)
    freeze(split, protocol)
    print("Input/code/protocol/split hashes frozen and verified", flush=True)
    with threadpool_limits(limits=1):
        if args.phase in ["representations", "all"]: representations(data, matrices, protocol)
        if args.phase in ["onboarding", "all"]: onboarding(data, matrices, split, protocol)
