"""Replay two completed LR folds and verify v2 CLI/bundle/cache compatibility.

Writes only to a new directory. The replay bundles are integration fixtures,
not final production models. Does not repeat the 360-job performance benchmark.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import production_pipeline_v2 as V


IDENTITY = "f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b"


def cli(arguments, log):
    with Path(log).open("x", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, "-u", str(V.ROOT / "run_production_v2.py"), *arguments],
            cwd=V.ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def verify(output, snapshot):
    output, snapshot = Path(output).resolve(), Path(snapshot).resolve()
    output.mkdir(parents=True, exist_ok=False)
    P, E, u = V.legacy()
    source = E.RESULTS / "concentration_production" / IDENTITY[:16]
    config = u.load_record(source / "config.json", IDENTITY)
    V.require(u.hash_json(config) == IDENTITY, "Original scientific identity changed")
    frame, groups = V.load_training_data(E.DATA)
    np.testing.assert_array_equal(groups, config["groups"])
    sources = [E.DATA, E.DATA.with_name("skin_vectorized.csv")]
    before = {str(p): V.file_hash(p) for p in sources}
    original = json.loads(snapshot.read_text(encoding="utf-8"))
    V.require(all(original[str(p)] == before[str(p)] for p in sources), "Input changed")
    completed = list(source.rglob("outer.json"))
    V.require(len(completed) == 360, "Expected all 360 completed outer checkpoints")
    for path in completed:
        u.load_record(path, IDENTITY)
    V.require(
        len(list(source.glob("*/*/*/*/*/candidates/*.json"))) == 7200,
        "Expected all 7200 candidate checkpoints",
    )
    print("Verified original identity, 360 outer and 7200 candidate files", flush=True)

    cli(["evaluate", "--prepare-only"], output / "evaluate_prepare.log")
    # Exercise all 360 cache reads through the actual canonical dispatcher.
    # Threads let a fit guard apply to every task; the original run_fold is unchanged.
    with (
        V.training_lock(),
        patch.object(P, "ProcessPoolExecutor", ThreadPoolExecutor),
        patch.object(
            u, "fit_weighted", side_effect=AssertionError("Cache reran a fit")
        ) as fitting,
        (output / "evaluate_cached.log").open("x", encoding="utf-8") as log,
        redirect_stdout(log),
    ):
        V.evaluate(workers=6)
        V.require(fitting.call_count == 0, "Unexpected fit during cache verification")
    print("Canonical evaluation reused all checkpoints; fit calls = 0", flush=True)

    results = []
    for arm in ["baseline", "concentration"]:
        checkpoint = source / arm / "A/lr/42/fold-0/outer.json"
        outer = u.load_record(checkpoint, IDENTITY)
        fold = config["splits"]["42"][0]
        V.require(
            outer["train_rows"] == fold["train"] and outer["test"] == fold["test"],
            "Replay split differs",
        )
        training = frame.iloc[fold["train"]]
        heldout = frame.iloc[fold["test"]]
        bundle = V.fit_model(
            training,
            u.epa_labels(frame.final_label)["A"][fold["train"]],
            arm=arm,
            model="lr",
            params=outer["selected_candidate"]["params"],
            label="A",
            threshold=outer["threshold"],
            provenance={
                "scope": "integration_validation_only_not_final_training",
                "source_evaluation_identity": IDENTITY,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": V.file_hash(checkpoint),
                "input_sha256": before[str(E.DATA)],
                "selected_candidate": outer["selected_candidate"],
                "training_rows": fold["train"],
                "heldout_rows": fold["test"],
            },
        )
        V.require(
            bundle.features.median_pct_ == outer["outer_concentration_median"],
            "Training median differs from original fold",
        )
        predicted = bundle.predict(heldout)
        np.testing.assert_allclose(
            predicted.score, outer["scores"], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(predicted.prediction, outer["predictions"])
        model_dir = output / f"{arm}_lr_fold0"
        restored = V.save_model(bundle, model_dir, heldout)
        pd.testing.assert_frame_equal(
            predicted, restored.predict(heldout), check_exact=True
        )

        # Exclude labels, role and groups, and reverse feature column order.
        columns = ["formulation_id", *bundle.features.base_columns_]
        if arm == "concentration":
            columns += ["known_pct", "pct_sum_used", "undeclared_pct"]
        raw = heldout[columns[::-1]]
        raw_path = output / f"{arm}_raw_inference.csv"
        raw.to_csv(raw_path, index=False, mode="x", encoding="utf-8-sig")
        predictions_path = output / f"{arm}_predictions.csv"
        cli(
            [
                "predict",
                "--model-dir",
                str(model_dir),
                "--input",
                str(raw_path),
                "--output",
                str(predictions_path),
            ],
            output / f"{arm}_predict.log",
        )
        from_cli = pd.read_csv(predictions_path)
        np.testing.assert_allclose(
            from_cli.score, predicted.score, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(from_cli.prediction, predicted.prediction)
        np.testing.assert_allclose(
            from_cli.threshold, bundle.threshold, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            restored.predict(raw.iloc[[0]]).score,
            predicted.score.iloc[[0]],
            rtol=1e-12,
            atol=1e-12,
        )
        # A fresh interpreter must not import the training harness to load/predict.
        code = (
            "import sys; import production_pipeline_v2 as v; "
            "b=v.load_model(sys.argv[1]); b.predict(v.read_raw_csv(sys.argv[2])); "
            "assert not {'run_concentration_production','experiment_utils',"
            "'epa_experiment_utils'} & set(sys.modules); print('independent inference verified')"
        )
        with (output / f"{arm}_independent_inference.log").open(
            "x", encoding="utf-8"
        ) as log:
            subprocess.run(
                [sys.executable, "-c", code, str(model_dir), str(raw_path)],
                cwd=V.ROOT,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results.append(
            {
                "arm": arm,
                "model": "lr",
                "label": "A",
                "seed": 42,
                "outer_fold": 0,
                "training_rows": len(training),
                "heldout_rows": len(heldout),
                "feature_count": len(bundle.features.feature_names_),
                "training_median_pct": bundle.features.median_pct_,
                "threshold": bundle.threshold,
                "score_max_absolute_difference": float(
                    np.max(np.abs(predicted.score - outer["scores"]))
                ),
                "predictions_equal_original_checkpoint": True,
                "reloading_predictions_exactly_equal": True,
                "fresh_process_cli_predictions_equal": True,
                "inference_without_training_imports": True,
                "input_column_reordering_verified": True,
                "single_row_batch_independence_verified": True,
                "model_sha256": V.file_hash(model_dir / "model.joblib"),
                "manifest_sha256": V.file_hash(model_dir / "model_manifest.json"),
            }
        )
        print(
            f"{arm}: replay/reload/fresh CLI predictions matched on {len(heldout)} rows",
            flush=True,
        )

    print(f"Checking preservation of {len(original)} existing files", flush=True)

    def changed(item):
        path, expected = item
        return (
            path if not Path(path).exists() or V.file_hash(path) != expected else None
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        differences = [p for p in pool.map(changed, original.items()) if p]
    V.require(not differences, f"Existing files changed: {differences}")
    report = {
        "at": datetime.now().astimezone().isoformat(),
        "status": "integration_verified",
        "scope": "v2 pipeline compatibility; not a new performance benchmark",
        "original_evaluation_identity": IDENTITY,
        "original_outer_checkpoints": 360,
        "original_candidate_checkpoints": 7200,
        "cached_evaluation_fit_calls": 0,
        "cache_verification_executor": "threads with fit guard; unchanged original run_fold",
        "new_full_cv_rerun": False,
        "original_full_cv_verification": str(
            E.RESULTS / "continuation_20261108T111518/delivery_verification.json"
        ),
        "replay_models": results,
        "preservation_snapshot_sha256": V.file_hash(snapshot),
        "existing_files_preserved": len(original),
        "changed_or_missing_existing_files": differences,
        "input_sha256": before,
        "v2_code_sha256": {
            name: V.file_hash(V.ROOT / name)
            for name in [
                "production_pipeline_v2.py",
                "run_production_v2.py",
                "test_production_pipeline_v2.py",
                "verify_production_pipeline_v2.py",
                "PRODUCTION_V2.txt",
            ]
        },
        "output_files_sha256": {
            str(p.relative_to(output)): V.file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "production_model_deployed_or_replaced": False,
        "skin_comparison": "not_run_missing_original_files_confirmed_by_user",
    }
    V.write_json_new(output / "verification.json", report)
    print(f"Verified: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preservation-snapshot", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output, args.preservation_snapshot)
