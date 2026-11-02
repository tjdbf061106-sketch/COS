import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ingredient_group, interpolate_viscosity


class DataContracts(unittest.TestCase):
    def test_power_law_interpolation(self):
        self.assertAlmostEqual(interpolate_viscosity([10, 1000], [1000, 10], 100), 100)

    def test_no_extrapolation(self):
        self.assertTrue(np.isnan(interpolate_viscosity([1, 10], [100, 10], 100)))

    def test_invalid_measurements_are_not_silent_zeros(self):
        with self.assertRaises(ValueError):
            interpolate_viscosity([10, 1000], [0, 10])
        with self.assertRaises(ValueError):
            interpolate_viscosity([10, 10], [20, 10])

    def test_group_holds_out_different_concentrations_of_same_combination(self):
        columns = ["surfactant A", "surfactant B", "polymer"]
        a = {"surfactant A": 1, "surfactant B": 5, "polymer": 0.1}
        b = {"surfactant A": 20, "surfactant B": 2, "polymer": 3}
        self.assertEqual(ingredient_group(a, columns), ingredient_group(b, columns))
        b["polymer"] = 0
        self.assertNotEqual(ingredient_group(a, columns), ingredient_group(b, columns))


if __name__ == "__main__":
    unittest.main()
