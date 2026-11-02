"""Validate source hashes, target provenance, group splits, OOF metrics and model round trips."""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from common import REPO, ROOT, interpolate_viscosity, load_inputs, read_json, sha256, write_json
from train_evaluate import TARGETS, build_model, metrics


def run(refit=False):
    checks = []
    protocol = read_json(ROOT / "protocol.json")
    for path, expected in read_json(ROOT / "training_lock.json")["hashes"].items():
        assert sha256(ROOT / path) == expected, path
    for entry in read_json(ROOT / "sources/manifest.json")["records"]:
        assert sha256(ROOT / entry["file"]) == entry["sha256"]
    for path, expected in read_json(ROOT / "data/skin_eye_source_audit.json")["input_hashes"].items():
        assert sha256(REPO / path) == expected, path
    checks.append("Published source, training inputs/code, and unchanged skin/eye input hashes")
    data, features = load_inputs()
    raw = pd.DataFrame(read_json(ROOT / "sources/LiquidFormulationsDataset_2023.json"))
    assert len(data) == 812 and data.sample_id.is_unique and len(features) == 18
    assert np.array_equal(data.sample_id, raw.ID)
    np.testing.assert_allclose(data[features], raw[features], rtol=0, atol=1e-12)
    assert np.array_equal(data.stable, raw.Stability_Test.astype(int))
    assert (data[features] >= 0).all().all()
    np.testing.assert_allclose(data[features].sum(axis=1) + data.added_water_balance_pct_estimate, 100, atol=1e-10)
    assert not data.loc[data.stable == 0, ["turbidity_ntu", "viscosity_100_mPa_s"]].notna().any().any()
    for source, output in zip(raw.itertuples(index=False), data.itertuples(index=False)):
        if np.isfinite(output.viscosity_100_mPa_s):
            curve = {k: v for part in source.Rheology_Data for k, v in part.items()}
            expected = interpolate_viscosity(curve["shear_rate"], curve["avg_viscosity"])
            np.testing.assert_allclose(output.viscosity_100_mPa_s, expected, atol=1e-8)
    checks.append("All stock quantities preserved; no missing measurements replaced with zero; interpolated viscosity targets trace to source curves")
    folds = pd.read_csv(ROOT / "results/outer_folds.csv")
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv", float_precision="round_trip")
    for (scheme, seed), split in folds.groupby(["scheme", "seed"]):
        assert split.sample_id.is_unique and set(split.sample_id) == set(data.sample_id)
        assert (split.groupby("group").fold.nunique() == 1).all()
        for target, col in TARGETS.items():
            eligible = data[col].notna() & ((data.stable == 1) if target != "stability" else True)
            expected_ids = set(data.loc[eligible, "sample_id"])
            for model in protocol["models"]:
                pred = oof[(oof.scheme == scheme) & (oof.seed == seed) & (oof.target == target) & (oof.model == model)]
                assert pred.sample_id.is_unique and set(pred.sample_id) == expected_ids
                aligned = pred.merge(data[["sample_id", col]], on="sample_id", validate="one_to_one")
                np.testing.assert_allclose(aligned.observed, aligned[col])
                merged = pred.merge(split[["sample_id", "fold"]], on="sample_id", suffixes=("", "_expected"))
                assert (merged.fold == merged.fold_expected).all()
                for fold, fold_pred in pred.groupby("fold"):
                    test_ids = set(split.loc[split.fold == fold, "sample_id"])
                    assert fold_pred.n_test.eq(len(expected_ids & test_ids)).all()
                    assert fold_pred.n_train.eq(len(expected_ids - test_ids)).all()
    checks.append("All grouped holdouts disjoint; one OOF prediction per eligible ID/model/repeat; task-specific training counts checked")
    repeat_metrics = pd.read_csv(ROOT / "results/repeat_metrics.csv", float_precision="round_trip")
    for row in repeat_metrics.itertuples(index=False):
        pred = oof[(oof.scheme == row.scheme) & (oof.seed == row.seed) & (oof.target == row.target) & (oof.model == row.model)]
        recomputed = metrics(row.target, pred.observed, pred.predicted)
        for name, value in recomputed.items():
            np.testing.assert_allclose(getattr(row, name), value, atol=1e-10, equal_nan=True)
    summary = pd.read_csv(ROOT / "results/metrics_summary.csv", float_precision="round_trip")
    for row in summary.itertuples(index=False):
        values = repeat_metrics[(repeat_metrics.scheme == row.scheme) & (repeat_metrics.target == row.target) & (repeat_metrics.model == row.model)][row.metric].dropna()
        np.testing.assert_allclose(row.mean, values.mean(), atol=1e-10)
        np.testing.assert_allclose(row.repeat_sd, values.std(ddof=1) if len(values)>1 else np.nan, atol=1e-10, equal_nan=True)
    checks.append(f"Recomputed {len(repeat_metrics)} pooled evaluation rows and {len(summary)} metric summaries")
    refits = {}
    for target, col in TARGETS.items():
        card = read_json(ROOT / "models" / f"{target}.card.json")
        model_path = ROOT / "models" / f"{target}.joblib"
        assert sha256(model_path) == card["model_sha256"] and card["features"] == features
        eligible = data[col].notna() & ((data.stable == 1) if target != "stability" else True)
        assert card["training_sample_ids"] == data.loc[eligible, "sample_id"].tolist()
        model = joblib.load(model_path)
        assert list(model.feature_names_in_) == features
        assert np.isfinite(model.predict(data[features])).all()
        if refit:
            seed = protocol["repeat_seeds"][0]
            test_ids = folds[(folds.scheme == "combination_holdout") & (folds.seed == seed) & (folds.fold == 0)].sample_id
            test_mask = data.sample_id.isin(test_ids)
            train = data[eligible & ~test_mask]
            pred = oof[(oof.scheme == "combination_holdout") & (oof.seed == seed) & (oof.fold == 0) & (oof.target == target) & (oof.model == "extra_trees")]
            test = data.set_index("sample_id").loc[pred.sample_id]
            fitted = build_model("extra_trees", target, seed, protocol).fit(train[features], train[col])
            values = fitted.predict_proba(test[features])[:, 1] if target == "stability" else fitted.predict(test[features])
            if target == "turbidity":
                values = np.maximum(values, 0)
            np.testing.assert_allclose(values, pred.predicted, rtol=0, atol=1e-10)
            refits[target] = float(np.max(abs(values-pred.predicted)))
    checks.append("Model hashes, feature order, eligible training IDs and finite predictions")
    links = pd.read_csv(ROOT / "data/endpoint_links.csv")
    assert len(links) == len(data) and set(links.sample_id) == set(data.sample_id)
    assert not links.skin_measured_label_available.any() and not links.eye_measured_label_available.any()
    assert not links.complete_molecular_composition.any()
    assert links.toxicity_transfer_status.eq("not_performed").all()
    checks.append("Physical outcomes kept separate from unmeasured skin/eye outcomes")
    candidate_path = ROOT / "results/candidate_predictions.csv"
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path, float_precision="round_trip")
        assert candidates.candidate_id.is_unique
        indexed = data.set_index("sample_id")
        for candidate in candidates.to_dict("records"):
            a, b = indexed.loc[candidate["parent_a"]], indexed.loc[candidate["parent_b"]]
            assert a.stable == b.stable == 1 and a.ingredient_combination == b.ingredient_combination
            expected = candidate["fraction_a"]*a[features].to_numpy(float)+(1-candidate["fraction_a"])*b[features].to_numpy(float)
            np.testing.assert_allclose([candidate[f] for f in features], expected, atol=1e-10)
        np.testing.assert_allclose(candidates[features].sum(axis=1)+candidates.added_water_balance_pct_estimate, 100, atol=1e-10)
        assert not candidates.measured_result_available.any()
        for target in TARGETS:
            model = joblib.load(ROOT / "models" / f"{target}.joblib")
            pred = model.predict_proba(candidates[features])[:, 1] if target == "stability" else model.predict(candidates[features])
            col = "stability_score_uncalibrated" if target == "stability" else f"predicted_log_{target}"
            if target == "turbidity":
                pred = np.maximum(pred, 0)
            np.testing.assert_allclose(candidates[col], pred, atol=1e-10)
        settings = read_json(ROOT / "results/candidate_summary.json")
        point_match = ((candidates.predicted_viscosity_100_mPa_s >= settings["min_viscosity_at_100_s_inv_mPa_s"])
                       & (candidates.predicted_turbidity_ntu <= settings["max_turbidity_ntu"])
                       & (candidates.stability_score_uncalibrated >= settings["min_uncalibrated_stability_score"]))
        assert np.array_equal(point_match, candidates.point_target_match)
        assert int(point_match.sum()) == settings["point_target_matches"]
        band_match = ((candidates.viscosity_100_error_band_lower >= settings["min_viscosity_at_100_s_inv_mPa_s"])
                      & (candidates.turbidity_error_band_upper <= settings["max_turbidity_ntu"])
                      & (candidates.stability_score_uncalibrated >= settings["min_uncalibrated_stability_score"]))
        assert np.array_equal(band_match, candidates.conservative_error_band_target_match)
        assert int(band_match.sum()) == settings["conservative_error_band_target_matches"]
        observed = pd.read_csv(ROOT / "results/observed_starting_formulations.csv")
        selected = data[(data.stable == 1) & (data.viscosity_100_mPa_s >= settings["min_viscosity_at_100_s_inv_mPa_s"])
                        & (data.turbidity_ntu <= settings["max_turbidity_ntu"])]
        assert set(observed.sample_id) == set(selected.sample_id) and len(observed) == settings["observed_starting_formulations"]
        checks.append("Candidate parent recipes, interpolation, mass balance and saved predictions")
    result = {"status": "PASS", "checks": checks, "fold0_refit_max_absolute_differences": refits,
              "external_validation_performed": False, "toxicity_transfer_performed": False}
    write_json(ROOT / "results/verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Re-fit outer fold 0 for all three primary models")
    run(parser.parse_args().refit)
