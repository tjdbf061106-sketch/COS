"""Verify the relocated research snapshots without rewriting the published files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent
BASE_NAME = "shampoo_physics_20261102"
REVIEW_NAME = "shampoo_inputs_review_20261102"
METRICS = ["Sensitivity", "Specificity", "MCC", "ROC-AUC"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions_in_both_orientations(p_stable):
    """Keep the original decision at an exact 0.5 tie: stable, not unstable."""
    stable = np.asarray(p_stable) >= 0.5
    return stable, ~stable


def summarize_metrics():
    base = ROOT / BASE_NAME
    original = pd.read_csv(base / "results/oof_predictions.csv", float_precision="round_trip")
    oof = original[original.target == "stability"]
    archived = pd.read_csv(ROOT / "classification_reporting/stability_oof_snapshot.csv",
                           float_precision="round_trip")
    pd.testing.assert_frame_equal(oof[archived.columns].reset_index(drop=True), archived)
    rows = []
    for (scheme, seed, model), group in oof.groupby(["scheme", "seed", "model"]):
        assert len(group) == 812 and group.sample_id.is_unique
        assert group.observed.isin([0, 1]).all() and group.predicted.between(0, 1).all()
        y_stable = group.observed.to_numpy(int)
        p_stable = group.predicted.to_numpy()
        assert (y_stable == 1).sum() == 294
        decisions = predictions_in_both_orientations(p_stable)
        pair = []
        for positive, y, score, pred in [
            ("stable", y_stable, p_stable, decisions[0]),
            # -p keeps the exact reversed ranking. 1-p may merge near-ties in float64.
            ("unstable", 1 - y_stable, -p_stable, decisions[1]),
        ]:
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row = {"scheme": scheme, "seed": int(seed), "model": model,
                   "positive_class": positive, "n": len(group),
                   "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
                   "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                   "MCC": matthews_corrcoef(y, pred), "ROC-AUC": roc_auc_score(y, score),
                   "threshold_ties": int((p_stable == 0.5).sum()),
                   "roc_auc_score_source": "p_stable" if positive == "stable" else "-p_stable"}
            row["sensitivity_specificity_gap"] = abs(row["Sensitivity"] - row["Specificity"])
            row["weaker_class_recall"] = min(row["Sensitivity"], row["Specificity"])
            rows.append(row)
            pair.append(row)
        np.testing.assert_allclose(
            [pair[0][k] for k in METRICS],
            [pair[1][k] for k in ["Specificity", "Sensitivity", "MCC", "ROC-AUC"]],
            rtol=0, atol=1e-12)
    repeat = pd.DataFrame(rows)
    prior = pd.read_csv(base / "results/repeat_metrics.csv", float_precision="round_trip")
    prior = prior[prior.target == "stability"]
    aligned = repeat[repeat.positive_class == "stable"].merge(
        prior, on=["scheme", "seed", "model"], validate="one_to_one")
    assert len(aligned) == 12
    for new, old in zip(METRICS, ["recall", "specificity", "mcc", "roc_auc"]):
        np.testing.assert_allclose(aligned[new], aligned[old], rtol=0, atol=1e-12)
    summary = []
    for (scheme, model, positive), group in repeat.groupby(["scheme", "model", "positive_class"]):
        row = {"scheme": scheme, "model": model, "positive_class": positive,
               "n_independent_samples": 812, "n_repeats": len(group)}
        for metric in METRICS + ["sensitivity_specificity_gap", "weaker_class_recall"]:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_sd"] = group[metric].std(ddof=1) if len(group) > 1 else np.nan
        summary.append(row)
    return repeat, pd.DataFrame(summary)


def run_process(args, cwd):
    result = subprocess.run([sys.executable, "-B", *args], cwd=cwd,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stdout + "\n" + result.stderr)
    return result


def run(repo, refit):
    checks = []
    for inventory in ["frozen_inventory.json", "manifest.json"]:
        for path, expected in read_json(ROOT / inventory)["files"].items():
            assert digest(ROOT / path) == expected, path
    checks.append("All frozen artifacts and package manifest hashes match")
    for directory in [BASE_NAME, REVIEW_NAME]:
        for path, expected in read_json(ROOT / directory / "manifest.json")["files"].items():
            assert digest(ROOT / directory / path) == expected, path
    demo = repo / "shampoo_demo"
    assert read_json(demo / "data/liquid_formulations/LiquidFormulationsDataset_2023.json") == read_json(
        ROOT / BASE_NAME / "sources/LiquidFormulationsDataset_2023.json")
    for existing, frozen in [
        ("data/ingredient_reference/surfactants/BASF Surfactants Information.csv", "surfactants_information.csv"),
        ("data/ingredient_reference/structures/BASF Formulation Ingredient Structures.pptx", "ingredient_structures.pptx"),
    ]:
        assert digest(demo / existing) == digest(ROOT / BASE_NAME / "sources" / frozen)
    checks.append("Existing shampoo_demo JSON content and ingredient source bytes match the snapshots")
    contract = read_json(ROOT / "prediction_contract.json")
    assert contract["feature_order"] == read_json(ROOT / BASE_NAME / "data/feature_names.json")
    assert contract["model_positive_class"] == "stable"
    assert contract["demo_positive_class"] == "unstable"
    assert contract["evaluation_threshold_p_stable"] == 0.5
    stable, unstable = predictions_in_both_orientations([0, 0.49, 0.5, 0.51, 1])
    assert stable.tolist() == [False, False, True, True, True]
    assert unstable.tolist() == [True, True, False, False, False]
    repeat, summary = summarize_metrics()
    for frame, name in [(repeat, "metrics_by_positive_class.csv"), (summary, "metrics_summary_by_positive_class.csv")]:
        saved = pd.read_csv(ROOT / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(frame, saved, check_exact=False, atol=1e-12, rtol=0)
    checks.append("All four metrics and confusion counts reproduced in both positive-class conventions; 0.5 ties preserved")
    with tempfile.TemporaryDirectory(prefix="cos_shampoo_verify_") as folder:
        temp = Path(folder)
        for directory in [BASE_NAME, REVIEW_NAME, "classification_reporting"]:
            shutil.copytree(ROOT / directory, temp / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Review checks all original baseline bytes, so run it before the baseline verifier writes its log.
        run_process(["src/verify_review.py"], temp / REVIEW_NAME)
        review_result = read_json(temp / REVIEW_NAME / "verification.json")
        for directory, expected_count in [(BASE_NAME, 4), (REVIEW_NAME, 8)]:
            result = run_process(["-m", "unittest", "discover", "-s", "tests", "-v"], temp / directory)
            assert f"Ran {expected_count} tests" in result.stderr, result.stderr
        code = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0,str(Path('src').resolve())); "
            "import common; common.REPO=Path(sys.argv[1]); "
            "import verify; verify.run(refit=sys.argv[2]=='1')"
        )
        run_process(["-c", code, str(repo), str(int(refit))], temp / BASE_NAME)
        baseline_result = read_json(temp / BASE_NAME / "results/verification.json")
        code = (
            "import sys; from pathlib import Path; import report_metrics; "
            "report_metrics.BASE=Path(sys.argv[1]); report_metrics.run()"
        )
        run_process(["-c", code, str(temp / BASE_NAME)], temp / "classification_reporting")
        metrics_result = read_json(temp / "classification_reporting/verification.json")
    checks.append("Original review, baseline and classification verifiers pass from temporary copies at the new layout")
    checks.append("Original 12 contract tests pass; shared snapshots are not overwritten")
    if refit:
        assert len(baseline_result["fold0_refit_max_absolute_differences"]) == 3
        checks.append("Outer fold 0 refit reproduced for all three Extra Trees targets")
    for path, expected in read_json(ROOT / "frozen_inventory.json")["files"].items():
        assert digest(ROOT / path) == expected, path
    return {"status": "PASS", "checks": checks, "frozen_files": 97, "unit_tests": 12,
            "metric_rows_both_orientations": len(repeat), "metric_summary_rows": len(summary),
            "baseline_verification": baseline_result, "review_verification": review_result,
            "archived_metrics_verification": metrics_result,
            "python": sys.version.split()[0],
            "versions": {name: importlib.metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "rdkit", "openpyxl"]},
            "external_validation_performed": False, "enriched_feature_model_trained": False,
            "model_reselected": False, "skin_eye_scoring_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT.parents[1], help="COS repository root")
    parser.add_argument("--refit", action="store_true", help="Also refit outer fold 0 for the three primary targets")
    parser.add_argument("--output", type=Path, help="Optional JSON result outside the frozen package")
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(ROOT):
        parser.error("--output must be outside this package to preserve its manifest")
    result = run(args.repo.resolve(), args.refit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
