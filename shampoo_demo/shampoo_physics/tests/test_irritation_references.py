import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_irritation_references import load_references


class IrritationReferenceContracts(unittest.TestCase):
    def test_pinned_source_shapes_and_labels(self):
        expected = {"skin": (3079, 3078, 3066), "eye": (2971, 2970, 2961)}
        for endpoint, counts in expected.items():
            raw, valid, references = load_references(ROOT / f"sources/skin_eye_chemical_{endpoint}.csv")
            self.assertEqual((len(raw), len(valid), len(references)), counts)
            self.assertTrue(raw["class"].isin([0, 1]).all())

    def test_no_exact_shampoo_representative_match_is_claimed(self):
        audit = pd.read_csv(ROOT / "data/skin_eye_chemical_reference_audit.csv")
        self.assertEqual(set(audit.endpoint), {"skin", "eye"})
        self.assertEqual(len(audit), 36)
        self.assertFalse(audit.exact_structure_match.any())
        self.assertTrue(audit.use.eq("reference_only_not_shampoo_prediction").all())


if __name__ == "__main__":
    unittest.main()
