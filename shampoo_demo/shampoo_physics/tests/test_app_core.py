import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app_core import (
    applicability,
    build_resolver,
    load_models,
    predict_properties,
    project_inputs,
    resolve_formula,
    resolve_ingredient,
)


class AppCoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training, cls.features, cls.registry = project_inputs()
        cls.resolver = build_resolver(cls.registry)

    def test_commercial_name_and_unique_inci_resolve(self):
        product = resolve_ingredient("Plantacare 2000", "", self.resolver, self.registry)
        self.assertEqual(product.ingredient_id, "Plantacare 2000")
        self.assertEqual(product.basis, "commercial_ingredient_name")
        inci = resolve_ingredient("Decyl-Glucoside", "", self.resolver, self.registry)
        self.assertEqual(inci.ingredient_id, "Plantacare 2000")
        self.assertEqual(inci.basis, "inci")

    def test_ambiguous_inci_and_unknown_smiles_do_not_resolve(self):
        ambiguous = resolve_ingredient("Polyquaternium-7", "", self.resolver, self.registry)
        self.assertEqual(ambiguous.status, "ambiguous")
        unknown = resolve_ingredient("", "CCO", self.resolver, self.registry)
        self.assertEqual(unknown.status, "unsupported")
        self.assertEqual(unknown.canonical_smiles, "CCO")

    def test_duplicate_rows_aggregate_and_unknown_blocks(self):
        rows = pd.DataFrame([
            {"ingredient": "Plantacare 2000", "smiles": "", "wt_pct": 2.0},
            {"ingredient": "Decyl-Glucoside", "smiles": "", "wt_pct": 3.0},
            {"ingredient": "new material", "smiles": "CCO", "wt_pct": 1.0},
        ])
        formula, details = resolve_formula(rows, self.registry, self.features)
        self.assertEqual(formula.at[0, "Plantacare 2000"], 5.0)
        self.assertIn("unsupported", set(details.status))

    def test_observed_formula_predicts_finite_outputs(self):
        formula = self.training[self.features].iloc[[0]].copy()
        domain = applicability(formula, self.training, self.features)
        self.assertEqual(domain["status"], "interpolation")
        prediction = predict_properties(formula, load_models(), self.features)
        self.assertTrue(all(np.isfinite(value) for value in prediction.values()))
        self.assertGreaterEqual(prediction["stability_score_uncalibrated"], 0)
        self.assertLessEqual(prediction["stability_score_uncalibrated"], 1)


if __name__ == "__main__":
    unittest.main()
