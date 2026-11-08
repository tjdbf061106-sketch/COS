"""Regression tests for the six-decimal grouping used by experiment 03."""

import unittest
from unittest.mock import patch

import numpy as np
import experiment_utils as X


class RoundedGroupingTests(unittest.TestCase):
    def test_roundoff_noise_is_merged_without_mutating_features(self):
        row = np.array([0.123456789, 42.5])
        original = row.copy()
        self.assertEqual(X.vector_key(row), X.vector_key(row + 1e-13))
        np.testing.assert_array_equal(row, original)

    def test_distinct_six_decimal_values_are_not_merged(self):
        self.assertNotEqual(
            X.vector_key(np.array([0.1])), X.vector_key(np.array([0.100002]))
        )

    def test_merges_are_transitive_and_preserve_original_groups(self):
        matrix = np.array([[1.0], [2.0], [2.0 + 1e-13], [3.0]])
        with patch.object(X, "original_groups", return_value=np.array([0, 0, 2, 3])):
            groups = X.merged_groups([None] * 4, matrix)
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_real_data_has_expected_cross_group_duplicates(self):
        import pandas as pd

        df, matrix, _ = X.load_eye()
        keys = [X.vector_key(row) for row in matrix]
        frame = pd.DataFrame({"key": keys, "group": X.original_groups(df)})
        split = frame.groupby("key")["group"].nunique()
        self.assertEqual(int(frame.key.isin(split[split > 1].index).sum()), 77)
        merged = X.merged_groups(df, matrix)
        self.assertTrue(
            pd.DataFrame({"key": keys, "group": merged})
            .groupby("key")["group"]
            .nunique()
            .eq(1)
            .all()
        )
        for seed in X.SEEDS:
            folds = X.u.nested_splits(df.final_label.to_numpy(), merged, seed=seed)
            for fold in folds:
                self.assertFalse(set(merged[fold["train"]]) & set(merged[fold["test"]]))
                for inner in fold["inner"]:
                    self.assertFalse(
                        set(merged[inner["train"]]) & set(merged[inner["validation"]])
                    )


if __name__ == "__main__":
    unittest.main()
