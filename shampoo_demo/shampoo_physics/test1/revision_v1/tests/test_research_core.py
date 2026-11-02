import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core import classification_metrics, linked_groups, split_group_sets, wilson_interval, assess_registration


class ResearchContracts(unittest.TestCase):
    def test_four_metrics_and_tie_positive_stable(self):
        m = classification_metrics([1, 1, 0, 0], [.5, .2, .8, .1])
        self.assertEqual((m["TP"], m["FN"], m["TN"], m["FP"]), (1, 1, 1, 1))
        self.assertEqual(m["Sensitivity"], .5)
        self.assertEqual(m["Specificity"], .5)
        self.assertEqual(m["MCC"], 0)

    def test_one_class_or_zero_coverage_is_not_success(self):
        m = classification_metrics([1, 0], [.9, .1], [False, False])
        self.assertEqual(m["coverage"], 0)
        self.assertTrue(np.isnan(m["Sensitivity"]))
        m = classification_metrics([1, 0], [.9, .1], [True, False])
        self.assertTrue(np.isnan(m["Specificity"]))
        self.assertTrue(np.isnan(m["MCC"]))
        self.assertTrue(np.isnan(m["ROC-AUC"]))

    def test_linked_identifiers_are_transitive_and_duplicate_recipes_grouped(self):
        f = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "source_id": ["s"]*4,
                          "formulation_id": ["r1", "r2", "r2", "r4"],
                          "preparation_batch_id": ["b1", "b1", "b2", "b4"],
                          "measurement_event_id": ["t1", "t2", "t3", "t4"], "x": [1., 2., 3., 3.]})
        g = linked_groups(f, ["x"])
        self.assertEqual(g.nunique(), 1)

    def test_missing_batch_cannot_claim_independence(self):
        with self.assertRaises(ValueError): linked_groups(pd.DataFrame({"x": [1, 2]}), ["x"])

    def test_split_group_disjoint_and_reproducible(self):
        a, b = split_group_sets(["a", "a", "b", "c", "d"], .3, 7)
        self.assertFalse(a & b)
        self.assertEqual(a | b, {"a", "b", "c", "d"})
        self.assertEqual((a, b), split_group_sets(["d", "c", "b", "a"], .3, 7))

    def test_small_perfect_sample_has_uncertainty(self):
        lower, upper = wilson_interval(17, 17)
        self.assertAlmostEqual(lower, .81568186499, places=6)
        self.assertAlmostEqual(upper, 1)

    def test_unset_criteria_and_unverified_units_hold(self):
        m = classification_metrics([1, 0], [.9, .1])
        self.assertEqual(assess_registration(m, {}, {}, True)["decision"], "HOLD")


if __name__ == "__main__": unittest.main()
