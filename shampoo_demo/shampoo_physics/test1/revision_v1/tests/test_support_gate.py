import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core import ROOT, read_json
from support_gate import resolve_verified_formula, joint_domain, guarded_prediction, verify_conditions
from intake_split import create_intake_split


class GateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "inputs/formulations.csv")
        cls.features = read_json(ROOT / "inputs/feature_names.json")
        cls.registry = pd.read_csv(ROOT / "inputs/ingredient_registry.csv")

    def test_unknown_supplier_name_never_falls_back_to_known_smiles(self):
        material = self.registry[self.registry.ingredient_id == "Plantacare 2000"].iloc[0]
        rows = pd.DataFrame([{"ingredient": "NEW_SUPPLIER_UNVERIFIED", "smiles": material.representative_smiles, "wt_pct": 5}])
        x, detail, issues = resolve_verified_formula(rows, self.registry, self.features, [])
        self.assertIsNone(x)
        self.assertTrue(issues)
        self.assertEqual(detail.status.iloc[0], "unverified_identity")

    def test_unique_inci_and_product_name_without_evidence_are_not_approval(self):
        for name in ["Decyl-Glucoside", "Plantacare 2000"]:
            x, _, issues = resolve_verified_formula(pd.DataFrame([{"ingredient": name, "wt_pct": 5}]), self.registry, self.features, [])
            self.assertIsNone(x)
            self.assertTrue(issues)

    def test_source_evidence_cannot_authorize_external_batch(self):
        records = [{"evidence_id": "paper", "ingredient_id": "Plantacare 2000", "supplier": "BASF", "grade": "Plantacare 2000",
                    "status": "documented_source_assumption", "evidence_reference": "Figshare v1"}]
        rows = pd.DataFrame([{"ingredient": "Plantacare 2000", "wt_pct": 5, "supplier": "BASF", "grade": "Plantacare 2000", "evidence_id": "paper"}])
        x, _, issues = resolve_verified_formula(rows, self.registry, self.features, records, "study_assumption")
        self.assertIsNotNone(x)
        self.assertFalse(issues)
        x, _, issues = resolve_verified_formula(rows, self.registry, self.features, records, "external")
        self.assertIsNone(x)
        self.assertTrue(issues)

    def test_external_verified_record_must_match_grade_and_lot(self):
        record = {"evidence_id": "test_fixture_only", "ingredient_id": "Plantacare 2000", "supplier": "BASF", "grade": "Plantacare 2000",
                  "lot_id": "fixture_lot", "status": "verified_external_grade", "evidence_reference": "synthetic unit-test fixture, not real evidence"}
        rows = pd.DataFrame([{"ingredient": "Plantacare 2000", "wt_pct": 5, "supplier": "BASF", "grade": "Plantacare 2000",
                              "lot_id": "different_lot", "evidence_id": "test_fixture_only"}])
        x, _, issues = resolve_verified_formula(rows, self.registry, self.features, [record])
        self.assertIsNone(x)
        self.assertTrue(issues)

    def test_singleton_changed_concentration_is_not_interpolation(self):
        x = self.data.loc[self.data.sample_id == 38, self.features].copy()
        x.loc[:, "Texapon SB 3 KC"] = self.data["Texapon SB 3 KC"].max()
        result = joint_domain(x, self.data, self.features)
        self.assertFalse(result["allow_research_prediction"])
        self.assertEqual(result["status"], "insufficient_joint_support")

    def test_joint_ranges_do_not_imply_convex_hull(self):
        training = pd.DataFrame({"sample_id": [1, 2, 3], "a": [1., 2., 1.], "b": [1., 1., 2.]})
        result = joint_domain(pd.DataFrame({"a": [2.], "b": [2.]}), training, ["a", "b"])
        self.assertEqual(result["status"], "outside_observed_convex_hull")
        self.assertFalse(result["allow_research_prediction"])

    def test_convex_combination_recovers_support_and_observed_is_measured(self):
        training = pd.DataFrame({"sample_id": [1, 2], "a": [2., 4.], "b": [5., 3.]})
        result = joint_domain(pd.DataFrame({"a": [3.], "b": [4.]}), training, ["a", "b"])
        self.assertTrue(result["allow_research_prediction"])
        self.assertAlmostEqual(sum(r["weight"] for r in result["convex_support"]), 1)
        result = joint_domain(training[["a", "b"]].iloc[[0]], training, ["a", "b"])
        self.assertEqual(result["status"], "observed_recipe")
        self.assertFalse(result["allow_research_prediction"])

    def test_blocked_inputs_never_call_models(self):
        model = Mock()
        result = guarded_prediction(None, ["unknown grade"], {}, {}, {"stability": model}, [])
        self.assertEqual(result["status"], "HOLD")
        model.predict_proba.assert_not_called()

    def test_conditions_and_unstable_score_suppress_regression(self):
        self.assertIn("pH", verify_conditions({"pH": 7}))
        classifier = Mock(); classifier.predict_proba.return_value = np.array([[.8, .2]])
        regressor = Mock()
        result = guarded_prediction(pd.DataFrame({"a": [1.]}), [], {"allow_research_prediction": True},
            {"pH": 5.8, "temperature_C": 25, "observation_hours": 36, "shear_rate_s_inv": 100},
            {"stability": classifier, "viscosity_100": regressor, "turbidity": regressor}, ["a"], "study_assumption")
        self.assertIsNone(result["predictions"]["viscosity_100_mPa_s"])
        regressor.predict.assert_not_called()

    def test_intake_split_groups_repeats_and_preserves_new_training(self):
        frame = pd.DataFrame({"record_id": [f"r{i}" for i in range(8)], "source_id": ["fixture"]*8,
          "formulation_id": [f"f{i//2}" for i in range(8)], "preparation_batch_id": [f"b{i//2}" for i in range(8)],
          "measurement_event_id": [f"m{i}" for i in range(8)], "contains_new_material": [True]*8,
          "new_stock": [float(i//2+1) for i in range(8)]})
        split = create_intake_split(frame, ["new_stock"], 7, .3)
        self.assertTrue(split.groupby("formulation_id").partition.nunique().eq(1).all())
        self.assertEqual(set(split.partition), {"A_development", "B_final_evaluation"})

    def test_high_stability_score_cannot_bypass_regression_domain(self):
        classifier = Mock(); classifier.predict_proba.return_value = np.array([[.1, .9]])
        regressor = Mock()
        result = guarded_prediction(pd.DataFrame({"a": [1.]}), [], {"allow_research_prediction": True},
            {"pH": 5.8, "temperature_C": 25, "observation_hours": 36, "shear_rate_s_inv": 100},
            {"stability": classifier, "viscosity_100": regressor, "turbidity": regressor}, ["a"], "study_assumption",
            {"viscosity_100": {"allow_research_prediction": False}, "turbidity": {"allow_research_prediction": False}})
        self.assertIsNone(result["predictions"]["viscosity_100_mPa_s"])
        self.assertIsNone(result["predictions"]["turbidity_ntu"])
        regressor.predict.assert_not_called()


if __name__ == "__main__": unittest.main()
