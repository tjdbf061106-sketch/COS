import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core import ROOT, read_json
from acceptance_review import validate_policy, review
from intake_split import create_intake_split


class PolicyContracts(unittest.TestCase):
    def test_invalid_acceptance_target_rejected(self):
        criteria = read_json(ROOT / "config/acceptance_criteria.json")
        criteria["sensitivity_lower_bound_min"] = -1
        with self.assertRaises(ValueError): validate_policy(criteria)

    def test_empty_real_evidence_and_unset_criteria_hold(self):
        result = review({}, read_json(ROOT / "config/acceptance_criteria.json"))
        self.assertEqual(result["decision"], "HOLD")
        self.assertFalse(result["registration_performed"])

    def test_false_new_material_indicator_rejected(self):
        frame = pd.DataFrame({"contains_new_material": [True, True], "novel_stock": [0., 1.]})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            create_intake_split(frame, ["novel_stock"], 7, .3)


if __name__ == "__main__": unittest.main()
