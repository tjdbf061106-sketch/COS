"""Behavior and serialization tests; these are not a new full CV benchmark."""

import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import production_pipeline_v2 as V
import run_production_v2 as CLI


def sample():
    rng = np.random.default_rng(82)
    n = 96
    names = [f"fp_{i}" for i in range(2048)] + [f"desc_{i}" for i in range(217)]
    matrix = np.zeros((n, 2265))
    matrix[:, :4] = rng.normal(size=(n, 4))
    labels = np.tile([1, 2, 3, 4], n // 4)
    matrix[:, 0] += 2 * (labels <= 2)
    d = pd.DataFrame(matrix, columns=names)
    d["final_label"] = labels
    d["formulation_id"] = [f"row-{i}" for i in range(n)]
    d["known_pct"] = rng.uniform(0, 100, n)
    d.loc[0, "known_pct"] = np.nan
    d["pct_sum_used"] = np.nan
    d["undeclared_pct"] = np.nan
    d["evaluation_role"] = ["existing_evaluation_pool"] * 80 + [
        "augmentation_train_only"
    ] * 16
    d["comp_group_id"] = list(range(80)) + list(range(16))
    d["split_group_id"] = d.comp_group_id
    return d


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.d = sample()

    def test_training_median_frozen_and_original_vectors_unchanged(self):
        train = self.d.iloc[1:40].copy()
        held = self.d.iloc[[0, 90]].copy()
        t = V.FeatureTransform("concentration").fit(train)
        median = float(train.known_pct.median())
        before = t.transform(held)
        self.assertEqual(t.median_pct_, median)
        self.assertEqual(before[0, -3], median / 100)
        held.loc[held.index[1], "known_pct"] = 99
        np.testing.assert_array_equal(t.transform(held)[0], before[0])
        np.testing.assert_array_equal(
            before[:, :2265], held[t.base_columns_].to_numpy()
        )
        self.assertEqual(t.transform(held).shape[1], 2269)

    def test_column_reordering_and_batch_independence(self):
        for arm in ["baseline", "concentration"]:
            t = V.FeatureTransform(arm).fit(self.d.iloc[1:50])
            shuffled = self.d[self.d.columns[::-1]]
            np.testing.assert_array_equal(t.transform(self.d), t.transform(shuffled))
            np.testing.assert_array_equal(
                t.transform(self.d.iloc[[0]])[0], t.transform(self.d)[0]
            )

    def test_bad_schema_and_precomputed_export_rejected(self):
        t = V.FeatureTransform("concentration").fit(self.d)
        cases = [
            self.d.drop(columns="fp_0"),
            self.d.rename(columns={"fp_0": "fp_unknown"}),
            self.d.assign(conc_pct=0.5),
            self.d.drop(columns="known_pct"),
            self.d.assign(known_pct=-1),
            self.d.assign(undeclared_pct=np.inf),
            self.d.assign(fp_0=np.inf),
        ]
        for frame in cases:
            with (
                self.subTest(columns=len(frame.columns)),
                self.assertRaises((ValueError, KeyError)),
            ):
                t.transform(frame)
        with self.assertRaises(ValueError):
            V.FeatureTransform("baseline").transform(self.d)

    def test_baseline_needs_no_concentration_labels_or_groups(self):
        t = V.FeatureTransform("baseline").fit(self.d)
        raw = self.d[t.base_columns_]
        self.assertEqual(t.transform(raw).shape, (96, 2265))

    def test_duplicate_csv_headers_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.csv"
            path.write_text("known_pct,known_pct\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                V.read_raw_csv(path)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P, cls.E, cls.u = V.legacy()
        cls.frame, cls.groups = V.load_training_data(cls.E.DATA)

    def test_all_saved_evaluation_splits_and_feature_views_match(self):
        source = (
            self.E.RESULTS / "concentration_production/f91f3c88787bec6b/config.json"
        )
        record = json.loads(source.read_text(encoding="utf-8"))
        cfg = self.u.load_record(source, record["identity"])
        self.assertEqual(self.groups.tolist(), cfg["groups"])
        df, matrix, _ = self.E.load_eye()
        for seed in self.E.SEEDS:
            folds = self.P.production_splits(df, self.groups, seed)
            self.assertEqual(folds, cfg["splits"][str(seed)])
            for fold in folds:
                partitions = [(fold["train"], fold["test"])] + [
                    (p["train"], p["validation"]) for p in fold["inner"]
                ]
                for a, b in partitions:
                    for arm in ["baseline", "concentration"]:
                        expected_a, expected_b, median = self.P.feature_pair(
                            df, matrix, a, b, arm
                        )
                        transform = V.FeatureTransform(arm).fit(df.iloc[a])
                        self.assertEqual(transform.median_pct_, median)
                        np.testing.assert_array_equal(
                            transform.transform(df.iloc[a]), expected_a
                        )
                        np.testing.assert_array_equal(
                            transform.transform(df.iloc[b]), expected_b
                        )

    def test_final_calibration_excludes_augmentation_and_linked_groups(self):
        df, groups = self.frame, self.groups
        aug = set(df.index[df.evaluation_role.eq("augmentation_train_only")])
        for seed in [42, 202, 777]:
            parts, evaluation = V.calibration_splits(df, groups, seed)
            self.assertEqual(len(evaluation), 606)
            self.assertEqual(
                sorted(i for p in parts for i in p["validation"]), evaluation
            )
            for p in parts:
                train, valid = set(p["train"]), set(p["validation"])
                self.assertFalse(valid & aug)
                blocked = set(groups[list(valid)])
                self.assertFalse(set(groups[list(train)]) & blocked)
                self.assertEqual(
                    train & aug, {i for i in aug if groups[i] not in blocked}
                )

    def test_invalid_training_roles_and_ids_fail(self):
        cases = [
            self.frame.assign(evaluation_role="existing_evaluation_pool"),
            self.frame.assign(formulation_id="duplicate"),
        ]
        for frame in cases:
            with (
                patch.object(V, "read_raw_csv", return_value=frame),
                self.assertRaises(ValueError),
            ):
                V.load_training_data(self.E.DATA)

    def test_evaluation_dispatch_preserves_full_grid_and_legacy_entry(self):
        seen = []
        previous = sys.argv
        with patch.object(
            self.P, "main", side_effect=lambda: seen.append(sys.argv[:])
        ) as main:
            V.evaluate(workers=6, prepare_only=True)
            main.assert_called_once()
        self.assertEqual(
            seen,
            [
                [
                    str(self.P.__file__),
                    "--workers",
                    "6",
                    "--models",
                    *V.MODELS,
                    "--prepare-only",
                ]
            ],
        )
        self.assertIs(sys.argv, previous)

    def test_active_legacy_runner_is_rejected_using_cmdline(self):
        for invocation in [
            "run_concentration_production.py",
            "run_concentration_production",
        ]:
            active = SimpleNamespace(
                pid=999999,
                info={"name": "python.exe", "cmdline": ["python", invocation]},
            )
            with (
                patch("psutil.process_iter", return_value=[active]),
                self.assertRaises(RuntimeError),
            ):
                with V.training_lock():
                    self.fail("Overlapping evaluation started")


class FinalTrainingTests(unittest.TestCase):
    def test_full_grid_threshold_and_interrupted_resume_without_refitting_candidates(
        self,
    ):
        # Small synthetic population, actual 20 LR candidates and actual grouped
        # folds. Only the strict real-CSV loader is replaced for this unit test.
        frame = sample()
        groups = frame.comp_group_id.to_numpy()
        u = V.legacy()[2]
        y = u.epa_labels(frame.final_label)["A"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(V, "load_training_data", return_value=(frame, groups)),
        ):
            output = Path(tmp) / "candidate"
            kwargs = dict(
                label="A", model="lr", arm="concentration", seed=42, output=output
            )
            with patch.object(
                V,
                "fit_model",
                side_effect=RuntimeError("simulated interruption before final fit"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    V.train_final(**kwargs)
            candidates = sorted((output / "candidates").glob("*.json"))
            self.assertEqual(len(candidates), 20)
            prior_hashes = {p: V.file_hash(p) for p in candidates}
            config = V.load_record(output / "training_config.json")
            calibration = V.load_record(output / "calibration.json")
            self.assertEqual(calibration["calibration_rows"], list(range(80)))
            self.assertEqual(
                calibration["threshold"],
                u.threshold(calibration["calibration_scores"], y[:80]),
            )
            for part, median in zip(
                config["calibration_splits"], calibration["inner_training_medians"]
            ):
                self.assertFalse(set(part["validation"]) & set(range(80, 96)))
                self.assertFalse(
                    set(groups[part["train"]]) & set(groups[part["validation"]])
                )
                self.assertEqual(
                    median, V.C.fit_concentration_median(frame.iloc[part["train"]])
                )
            with patch.object(u, "fit_weighted", wraps=u.fit_weighted) as fitting:
                self.assertEqual(V.train_final(**kwargs), output)
                self.assertEqual(fitting.call_count, 1)  # Only the final fit.
            self.assertEqual(prior_hashes, {p: V.file_hash(p) for p in candidates})
            bundle = V.load_model(output)
            self.assertEqual(
                bundle.features.median_pct_, V.C.fit_concentration_median(frame)
            )
            self.assertEqual(bundle.threshold, calibration["threshold"])
            self.assertEqual(bundle.provenance["training_rows"], 96)
            self.assertEqual(bundle.provenance["calibration_rows"], 80)
            all_hashes = {p: V.file_hash(p) for p in output.rglob("*") if p.is_file()}
            with patch.object(
                u, "fit_weighted", side_effect=AssertionError("Completed run refitted")
            ):
                self.assertEqual(V.train_final(**kwargs), output)
            self.assertEqual(
                all_hashes,
                {p: V.file_hash(p) for p in output.rglob("*") if p.is_file()},
            )
            with self.assertRaises(ValueError):
                V.train_final(**{**kwargs, "seed": 202})


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = sample()
        cls.u = V.legacy()[2]
        cls.bundle = V.fit_model(
            cls.d,
            cls.u.epa_labels(cls.d.final_label)["A"],
            arm="concentration",
            model="lr",
            params=cls.u.candidate_grid("lr")[0]["params"],
            label="A",
            threshold=0.3,
            provenance={"scope": "unit_test"},
        )

    def test_serialization_raw_inference_order_batch_and_threshold(self):
        cols = self.bundle.features.base_columns_ + [
            "known_pct",
            "pct_sum_used",
            "undeclared_pct",
        ]
        raw = self.d[cols]
        before = self.bundle.predict(raw)
        with tempfile.TemporaryDirectory() as tmp:
            restored = V.save_model(self.bundle, tmp, raw)
            pd.testing.assert_frame_equal(
                before, restored.predict(raw[raw.columns[::-1]]), check_exact=True
            )
            np.testing.assert_allclose(
                restored.predict(raw.iloc[[0]]).score,
                before.score.iloc[[0]],
                rtol=1e-12,
            )
            np.testing.assert_array_equal(
                before.prediction, (before.score >= 0.3).astype(int)
            )
            self.assertFalse(
                np.array_equal(before.prediction, (before.score >= 0.5).astype(int))
            )
            with patch.object(
                V,
                "legacy",
                side_effect=AssertionError("Inference imported training code"),
            ):
                V.load_model(tmp).predict(raw)
            with self.assertRaises(FileExistsError):
                V.save_model(self.bundle, tmp, raw)

    def test_corruption_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            V.save_model(self.bundle, tmp, self.d)
            path = Path(tmp) / "model.joblib"
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with (
                patch.object(
                    V.joblib, "load", side_effect=AssertionError("Must not deserialize")
                ),
                self.assertRaises(ValueError),
            ):
                V.load_model(tmp)

    def test_manifest_environment_and_runtime_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            V.save_model(self.bundle, tmp, self.d)
            with (
                patch.object(V, "runtime_environment", return_value={}),
                self.assertRaises(ValueError),
            ):
                V.load_model(tmp)
            with (
                patch.object(V, "runtime_hashes", return_value={}),
                self.assertRaises(ValueError),
            ):
                V.load_model(tmp)
            record = json.loads(
                (Path(tmp) / "model_manifest.json").read_text(encoding="utf-8")
            )
            record["payload"]["description"]["threshold"] = 0.9
            (Path(tmp) / "model_manifest.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                V.load_model(tmp)

    def test_prediction_files_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            V.save_model(self.bundle, root / "bundle", self.d)
            self.d.to_csv(root / "input.csv", index=False)
            CLI.predict_csv(
                root / "bundle", root / "input.csv", root / "predictions.csv"
            )
            manifest = json.loads(
                (root / "predictions.csv.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["rows"], len(self.d))
            self.assertFalse(manifest["preprocessing_refitted"])
            before = (root / "predictions.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                CLI.predict_csv(
                    root / "bundle", root / "input.csv", root / "predictions.csv"
                )
            self.assertEqual(before, (root / "predictions.csv").read_bytes())

    def test_all_six_estimators_and_both_feature_dimensions(self):
        params = {
            "lr": {"C": 0.01, "solver": "lbfgs", "max_iter": 5000, "tol": 1e-4},
            "rf": {
                "n_estimators": 5,
                "max_depth": 2,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },
            "et": {
                "n_estimators": 5,
                "max_depth": 2,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },
            "svm": {"kernel": "rbf", "probability": False, "C": 1, "gamma": "scale"},
            "xgb": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
            "lgbm": {"n_estimators": 5, "num_leaves": 7, "learning_rate": 0.1},
        }
        for model in V.MODELS:
            for arm in ["baseline", "concentration"]:
                with (
                    self.subTest(model=model, arm=arm),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    bundle = V.fit_model(
                        self.d,
                        self.u.epa_labels(self.d.final_label)["B"],
                        arm=arm,
                        model=model,
                        params=params[model],
                        label="B",
                        threshold=0.1,
                        provenance={
                            "scope": "small_estimator_smoke_test_not_full_grid"
                        },
                    )
                    restored = V.save_model(bundle, tmp, self.d)
                    self.assertEqual(
                        restored.features.transform(self.d).shape[1],
                        2265 if arm == "baseline" else 2269,
                    )
                    pd.testing.assert_frame_equal(
                        bundle.predict(self.d),
                        restored.predict(self.d),
                        check_exact=True,
                    )

    def test_atomic_records_never_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "record.json"
            V.save_record(p, "identity", {"value": 1})
            before = p.read_bytes()
            V.save_record(p, "identity", {"value": 1})
            with self.assertRaises(ValueError):
                V.save_record(p, "identity", {"value": 2})
            with self.assertRaises(FileExistsError):
                V.write_json_new(p, {})
            self.assertEqual(before, p.read_bytes())


if __name__ == "__main__":
    unittest.main()
