"""Generate experimental candidates by interpolating existing stable recipes of the same ingredient set."""
from __future__ import annotations

import argparse
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from common import ROOT, load_inputs, write_csv, write_json


def run(min_viscosity=100.0, max_turbidity=100.0, min_stability_score=0.8):
    if min_viscosity <= 0 or max_turbidity < 0 or not 0 <= min_stability_score <= 1:
        raise ValueError("Invalid physical target or stability-score threshold")
    data, features = load_inputs()
    stable = data[data.stable == 1]
    rows = []
    seen = {tuple(np.round(x, 8)) for x in data[features].to_numpy()}
    for group, frame in stable.groupby("ingredient_combination"):
        for ia, ib in combinations(frame.index, 2):
            a, b = data.loc[ia], data.loc[ib]
            for fraction_a in [0.25, 0.5, 0.75]:
                composition = fraction_a * a[features].to_numpy(float) + (1 - fraction_a) * b[features].to_numpy(float)
                signature = tuple(np.round(composition, 8))
                if signature in seen:
                    continue
                seen.add(signature)
                rows.append({"candidate_id": f"C{len(rows)+1:04}", "parent_a": int(a.sample_id), "parent_b": int(b.sample_id),
                             "fraction_a": fraction_a, "ingredient_combination": group,
                             **dict(zip(features, composition)), "added_water_balance_pct_estimate": 100-composition.sum()})
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No pairs of stable recipes with the same ingredient combination")
    for target in ["stability", "viscosity_100", "turbidity"]:
        model = joblib.load(ROOT / "models" / f"{target}.joblib")
        if target == "stability":
            candidates["stability_score_uncalibrated"] = model.predict_proba(candidates[features])[:, 1]
        else:
            pred = model.predict(candidates[features])
            candidates[f"predicted_log_{target}"] = np.maximum(pred, 0) if target == "turbidity" else pred
    candidates["predicted_viscosity_100_mPa_s"] = 10 ** candidates.predicted_log_viscosity_100
    candidates["predicted_turbidity_ntu"] = 10 ** candidates.predicted_log_turbidity - 1
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv", float_precision="round_trip")
    first_seed = int(oof.seed.min())
    residual_bounds = {}
    for target, offset in [("viscosity_100", 0), ("turbidity", 1)]:
        holdout = oof[(oof.scheme == "combination_holdout") & (oof.seed == first_seed) & (oof.model == "extra_trees") & (oof.target == target)]
        q = float(np.quantile(abs(holdout.observed - holdout.predicted), 0.9))
        residual_bounds[target] = q
        pred = candidates[f"predicted_log_{target}"]
        candidates[f"{target}_error_band_lower"] = np.maximum(10 ** (pred - q) - offset, 0)
        candidates[f"{target}_error_band_upper"] = np.maximum(10 ** (pred + q) - offset, 0)
    candidates["point_target_match"] = ((candidates.predicted_viscosity_100_mPa_s >= min_viscosity)
                                         & (candidates.predicted_turbidity_ntu <= max_turbidity)
                                         & (candidates.stability_score_uncalibrated >= min_stability_score))
    candidates["conservative_error_band_target_match"] = ((candidates.viscosity_100_error_band_lower >= min_viscosity)
                                                           & (candidates.turbidity_error_band_upper <= max_turbidity)
                                                           & (candidates.stability_score_uncalibrated >= min_stability_score))
    candidates["ranking_loss"] = (np.maximum(np.log10(min_viscosity)-candidates.predicted_log_viscosity_100, 0)
                                   + np.maximum(candidates.predicted_log_turbidity-np.log10(1+max_turbidity), 0)
                                   + 1-candidates.stability_score_uncalibrated)
    candidates["measured_result_available"] = False
    candidates["skin_status"] = "not_validated_for_shampoo"
    candidates["eye_status"] = "not_validated_for_shampoo"
    candidates["intended_use"] = "candidate_for_experiment_only"
    candidates = candidates.sort_values(["point_target_match", "ranking_loss", "candidate_id"], ascending=[False, True, True])
    write_csv(ROOT / "results/candidate_predictions.csv", candidates)
    observed = stable[(stable.viscosity_100_mPa_s >= min_viscosity) & (stable.turbidity_ntu <= max_turbidity)].copy()
    observed["result_type"] = "published_observation_not_new_prediction"
    observed["skin_status"] = "no_paired_shampoo_label"
    observed["eye_status"] = "no_paired_shampoo_label"
    write_csv(ROOT / "results/observed_starting_formulations.csv", observed)
    write_json(ROOT / "results/candidate_summary.json", {
        "target_is_illustrative_not_user_specified": True,
        "min_viscosity_at_100_s_inv_mPa_s": min_viscosity, "max_turbidity_ntu": max_turbidity,
        "min_uncalibrated_stability_score": min_stability_score,
        "generation": "Convex interpolation at 0.25/0.5/0.75 between two observed stable recipes with exactly the same commercial ingredient set; exact observed recipes removed",
        "candidate_count": len(candidates), "point_target_matches": int(candidates.point_target_match.sum()),
        "conservative_error_band_target_matches": int(candidates.conservative_error_band_target_match.sum()),
        "observed_starting_formulations": len(observed), "residual_log10_halfwidths": residual_bounds,
        "uncertainty": "Descriptive 90th percentile of absolute OOF log residuals from the first grouped-CV repeat. NOT calibrated prediction intervals, NOT joint 90% coverage, NOT a confidence bound on safety.",
        "limitations": "All candidates are unmeasured; interpolation is not proof of stability. Commercial grades, mixing, pH and temperature must match source conditions. No irritation prediction used in ranking.",
    })
    print({"candidates": len(candidates), "point_matches": int(candidates.point_target_match.sum()),
           "error_band_matches": int(candidates.conservative_error_band_target_match.sum()), "measured_starts": len(observed)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-viscosity", type=float, default=100.0, help="mPa.s at 100 s^-1; illustrative default")
    parser.add_argument("--max-turbidity", type=float, default=100.0, help="NTU; illustrative default")
    parser.add_argument("--min-stability-score", type=float, default=0.8, help="Uncalibrated model score; not a guaranteed stability probability")
    args = parser.parse_args()
    run(args.min_viscosity, args.max_turbidity, args.min_stability_score)
