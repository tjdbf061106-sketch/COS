"""Behavioral checks for concentration transforms and train-only validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import concentration_features as C
import run_concentration_production as P


def frame(known, fallback=None, undeclared=None):
    n = len(known)
    return pd.DataFrame(
        {
            "known_pct": known,
            "pct_sum_used": fallback if fallback is not None else [np.nan] * n,
            "undeclared_pct": undeclared if undeclared is not None else [np.nan] * n,
        }
    )


class ConcentrationTests(unittest.TestCase):
    def test_formulas_fallback_and_neat_boundary(self):
        df = frame(
            [0, 0.001, 89.999, 90, 100, np.nan, np.nan],
            [np.nan] * 5 + [20, np.nan],
            [100, 0, 10, 20, 0, np.nan, 30],
        )
        block, names, pct = C.concentration_block(df, 7)
        np.testing.assert_array_equal(pct, [0, 0.001, 89.999, 90, 100, 20, 7])
        np.testing.assert_allclose(block[:, 0], np.log10(pct + 0.001))
        np.testing.assert_allclose(block[:, 1], pct / 100)
        np.testing.assert_array_equal(block[:, 2], [0, 0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(block[:, 3], [1, 0, 0.1, 0.2, 0, 0, 0.3])
        self.assertEqual(len(names), 4)

    def test_heldout_values_do_not_influence_median(self):
        df = frame([1.0, 3.0, np.nan, 99.0])
        matrix = np.arange(8.0).reshape(4, 2)
        before = matrix.copy()
        train, heldout, median = P.feature_pair(
            df, matrix, [0, 1], [2, 3], "concentration"
        )
        self.assertEqual(median, 2.0)
        self.assertEqual(heldout[0, -3], 0.02)
        df.loc[3, "known_pct"] = 100.0
        _, changed, median2 = P.feature_pair(
            df, matrix, [0, 1], [2, 3], "concentration"
        )
        self.assertEqual(median2, median)
        np.testing.assert_array_equal(changed[0], heldout[0])
        np.testing.assert_array_equal(matrix, before)
        np.testing.assert_array_equal(train[:, :2], matrix[:2])

    def test_invalid_inputs_and_all_missing_training_fail(self):
        for invalid in [-1, 101, np.inf]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                C.concentration_block(frame([invalid]), 1)
        with self.assertRaises(ValueError):
            C.fit_concentration_median(frame([np.nan]))
        with self.assertRaises(ValueError):
            C.concentration_block(frame([1]), np.nan)


class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.matrix, cls.columns = P.E.load_eye()
        cls.groups = P.E.merged_groups(cls.df, cls.matrix)

    def test_real_data_fallback_and_feature_dimensions(self):
        median = C.fit_concentration_median(self.df)
        block, _, pct = C.concentration_block(self.df, median)
        self.assertEqual(block.shape, (673, 4))
        self.assertTrue(np.isfinite(block).all())
        missing = self.df.known_pct.isna() & self.df.pct_sum_used.isna()
        self.assertEqual(int(missing.sum()), 9)
        np.testing.assert_array_equal(pct[missing], np.full(9, median))

    def test_export_preserves_real_data_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            csv = P.export_features(
                self.df, self.matrix, self.columns, output, "export-test", "source-test"
            )
            before = csv.read_bytes()
            P.export_features(
                self.df, self.matrix, self.columns, output, "export-test", "source-test"
            )
            self.assertEqual(csv.read_bytes(), before)
            exported = pd.read_csv(csv, low_memory=False)
            self.assertEqual(len(exported.columns), len(self.df.columns) + 4)
            csv.write_bytes(before + b"\n")
            with self.assertRaises(ValueError):
                P.export_features(
                    self.df,
                    self.matrix,
                    self.columns,
                    output,
                    "export-test",
                    "source-test",
                )
            self.assertEqual(csv.read_bytes(), before + b"\n")

    def test_real_splits_exclude_augmentation_and_all_heldout_groups(self):
        evaluation = set(
            np.flatnonzero(self.df.evaluation_role.eq("existing_evaluation_pool"))
        )
        augmentation = set(range(len(self.df))) - evaluation
        used_aug = set()
        for seed in P.E.SEEDS:
            folds = P.production_splits(self.df, self.groups, seed)
            observed = []
            for fold in folds:
                test, train = set(fold["test"]), set(fold["train"])
                observed.extend(fold["test"])
                self.assertFalse(test & augmentation)
                self.assertFalse(set(fold["calibration"]) & augmentation)
                self.assertFalse(
                    set(self.groups[list(train)]) & set(self.groups[list(test)])
                )
                used_aug.update(train & augmentation)
                for inner in fold["inner"]:
                    a, b = set(inner["train"]), set(inner["validation"])
                    self.assertFalse(b & augmentation)
                    self.assertFalse(a & (test | b))
                    self.assertFalse(
                        set(self.groups[list(a)]) & set(self.groups[list(test | b)])
                    )
                    self.assertLessEqual(a, train)
            self.assertEqual(len(observed), 606)
            self.assertEqual(set(observed), evaluation)
        self.assertTrue(used_aug)

    def test_small_real_estimator_job_and_checkpoint_integrity(self):
        # Real sklearn fitting, small synthetic population and one candidate;
        # this is a smoke test, not the full production benchmark.
        rng = np.random.default_rng(11)
        n = 90
        df = frame(rng.uniform(0.001, 100, n))
        df["final_label"] = np.tile([1, 2, 3, 4, 3], n // 5)
        df["evaluation_role"] = ["existing_evaluation_pool"] * 80 + [
            "augmentation_train_only"
        ] * 10
        matrix = rng.normal(size=(n, 4))
        matrix[:, 0] += df.final_label.to_numpy() <= 2
        fold = P.production_splits(df, np.arange(n), 42)[0]
        candidate = P.u.candidate_grid("lr")[3]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(P, "worker_data", return_value=(df, matrix, [])),
            patch.object(P.u, "candidate_grid", return_value=[candidate]),
        ):
            job = dict(
                folder=tmp,
                identity="test-identity",
                source_hash="test",
                split=fold,
                arm="concentration",
                label="A",
                model="lr",
                seed=42,
                fold=0,
            )
            result = P.run_fold(job)
            self.assertEqual(result["candidates_evaluated"], 1)
            self.assertEqual(result["test"], fold["test"])
            self.assertTrue(np.isfinite(result["scores"]).all())
            self.assertEqual(result, P.run_fold(job))
            with self.assertRaises(ValueError):
                P.u.load_record(Path(tmp) / "outer.json", "different-identity")


if __name__ == "__main__":
    unittest.main()
