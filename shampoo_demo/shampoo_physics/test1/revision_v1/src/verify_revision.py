"""Verify locked inputs, split isolation, all saved metrics, and selected fresh refits."""
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_core import ROOT, TARGETS, read_json, write_json, sha, classification_metrics
from run_benchmark import load, fit_predict, evaluate


def check_metrics(actual, expected, keys):
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    for key in keys:
        assert actual[key].astype(str).tolist() == expected[key].astype(str).tolist(), key
    for col in actual.columns:
        if col in keys: continue
        if pd.api.types.is_numeric_dtype(actual[col]):
            np.testing.assert_allclose(actual[col], expected[col], rtol=0, atol=1e-10, equal_nan=True)
        else: assert actual[col].fillna("").tolist() == expected[col].fillna("").tolist(), col


def run(refit=False):
    checks = []
    for path, expected in read_json(ROOT / "config/analysis_lock.json")["hashes"].items():
        assert sha(ROOT / path) == expected, path
    source = read_json(ROOT / "source_provenance.json")
    for path, expected in source["original_project_files"].items():
        assert sha(ROOT.parents[1] / path) == expected, path
    checks.append("Frozen analysis inputs/code/protocol/splits match and shared project snapshot matches sanitized provenance")
    data, matrices, sets = load()
    protocol = read_json(ROOT / "config/research_protocol.json")
    representation = pd.read_csv(ROOT / "results/representation_predictions.csv.gz", float_precision="round_trip")
    onboarding = pd.read_csv(ROOT / "results/onboarding_predictions.csv.gz", float_precision="round_trip", dtype={"budget": str})
    splits = pd.read_csv(ROOT / "results/onboarding_splits.csv")
    folds = pd.read_csv(ROOT / "inputs/outer_folds.csv").query("scheme == 'combination_holdout'")
    for (seed, fold, feature, model, target), pred in representation.groupby(["seed", "fold", "features", "model", "target"]):
        col = TARGETS[target]
        eligible = data[col].notna() & (data.stable.eq(1) if target != "stability" else True)
        test_ids = set(folds[(folds.seed == seed) & (folds.fold == fold)].sample_id)
        train_ids = set(data.index) - test_ids
        assert not set(data.loc[list(test_ids)].ingredient_combination) & set(data.loc[list(train_ids)].ingredient_combination)
        assert pred.sample_id.is_unique and set(pred.sample_id) == test_ids & set(data.index[eligible])
        assert pred.n_train.eq(len(train_ids & set(data.index[eligible]))).all()
        np.testing.assert_allclose(pred.observed, data.loc[pred.sample_id, col], rtol=0, atol=1e-12)
    for (ingredient, seed), split in splits.groupby(["held_ingredient", "seed"]):
        assert len(split) == 812 and split.sample_id.is_unique
        assert split.groupby("group").partition.nunique().eq(1).all()
        old_train = set(split[split.partition == "old_training"].sample_id)
        pool = split[split.partition == "adaptation_pool"]
        for budget in ["0", "4", "12", "all"]:
            added = pool if budget == "all" else pool[pool.adaptation_group_rank <= int(budget)]
            train_ids = old_train | set(added.sample_id)
            assert data.loc[list(old_train), ingredient].eq(0).all()
            if budget != "0": assert len(added) and data.loc[added.sample_id, ingredient].gt(0).all()
            for cohort in ["new_final_test", "old_retention_test"]:
                test_ids = set(split[split.partition == cohort].sample_id)
                assert not train_ids & test_ids
                assert not set(data.loc[list(test_ids)].ingredient_combination) & set(data.loc[list(train_ids)].ingredient_combination)
                selected = onboarding[(onboarding.held_ingredient == ingredient) & (onboarding.seed == seed)
                                      & (onboarding.budget == budget) & (onboarding.cohort == cohort)]
                assert set(selected.features) == {"stock18", "physics_core61"}
                for (features, target), pred in selected.groupby(["features", "target"]):
                    col = TARGETS[target]
                    eligible = data[col].notna() & (data.stable.eq(1) if target != "stability" else True)
                    assert pred.sample_id.is_unique and set(pred.sample_id) == test_ids & set(data.index[eligible])
                    assert pred.n_train.eq(len(train_ids & set(data.index[eligible]))).all()
                    np.testing.assert_allclose(pred.observed, data.loc[pred.sample_id, col], rtol=0, atol=1e-12)
    checks.append("Every representation and onboarding prediction traces to eligible targets and disjoint recipe groups with a locked final test")
    configs = [(representation, ["seed", "fold", "features", "model", "target"], "representation_fold_metrics.csv"),
               (representation, ["seed", "features", "model", "target"], "representation_repeat_metrics.csv"),
               (onboarding, ["held_ingredient", "seed", "budget", "features", "model", "target", "cohort"], "onboarding_metrics.csv")]
    for pred, keys, name in configs:
        actual = evaluate(pred, keys)
        expected = pd.read_csv(ROOT / "results" / name, float_precision="round_trip", dtype={"budget": str})
        check_metrics(actual, expected, keys)
    saved_selective = pd.read_csv(ROOT / "results/selective_prediction_metrics.csv", float_precision="round_trip")
    for row in saved_selective.to_dict("records"):
        pred = representation[(representation.seed == row["seed"]) & (representation.features == row["features"])
                              & (representation.model == row["model"]) & (representation.target == "stability")]
        expected = classification_metrics(pred.observed, pred.predicted, abs(pred.predicted.to_numpy() - .5) >= row["confidence_margin"])
        for name, value in expected.items():
            if isinstance(value, str): assert row[name] == value
            else: np.testing.assert_allclose(row[name], value, rtol=0, atol=1e-12, equal_nan=True)
    checks.append("All saved classification/regression metrics, confusion counts and class-specific coverage recompute from saved predictions")
    differences = {}
    if refit:
        seed = protocol["repeat_seeds"][0]
        test_ids = folds[(folds.seed == seed) & (folds.fold == 0)].sample_id.tolist()
        train_ids = data.index[~data.index.isin(test_ids)].tolist()
        for target in TARGETS:
            records, _ = fit_predict(data, matrices["physics_core61"], train_ids, test_ids, target, "extra_trees", seed, protocol)
            expected = representation[(representation.seed == seed) & (representation.fold == 0) & (representation.features == "physics_core61")
                                      & (representation.model == "extra_trees") & (representation.target == target)].set_index("sample_id")
            actual = pd.DataFrame(records).set_index("sample_id").loc[expected.index]
            np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=0, atol=1e-10)
            differences["representation_"+target] = float(max(abs(actual.predicted-expected.predicted)))
        ingredient = sets["stock18"][0]
        split = splits[(splits.held_ingredient == ingredient) & (splits.seed == seed)]
        train = split[(split.partition == "old_training") | ((split.partition == "adaptation_pool") & (split.adaptation_group_rank <= 4))].sample_id.tolist()
        test = split[split.partition.isin(["new_final_test", "old_retention_test"])].sample_id.tolist()
        for target in TARGETS:
            records, _ = fit_predict(data, matrices["physics_core61"], train, test, target, "linear", seed, protocol)
            expected = onboarding[(onboarding.held_ingredient == ingredient) & (onboarding.seed == seed) & (onboarding.budget == "4")
                                  & (onboarding.features == "physics_core61") & (onboarding.target == target)].set_index("sample_id")
            actual = pd.DataFrame(records).set_index("sample_id").loc[expected.index]
            np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=0, atol=1e-10)
            differences["onboarding_"+target] = float(max(abs(actual.predicted-expected.predicted)))
        checks.append("Six selected target fits reproduce the saved predictions, including a new-material adaptation case")
    result = {"status": "PASS", "checks": checks, "refit_max_absolute_differences": differences,
              "shared_snapshot_matches_provenance": True, "real_new_material_experiment": False, "independent_external_validation": False,
              "representation_predictions": len(representation), "onboarding_predictions": len(onboarding)}
    write_json(ROOT / "results/verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1): run(args.refit)
