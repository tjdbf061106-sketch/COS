import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from build_review import ROOT, build_features, interp_log, read_json


class FeatureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=pd.read_csv(ROOT/"data/ingredient_registry_reviewed.csv")
        cls.stocks=read_json(ROOT/"inputs/stock_feature_names.json")

    def recipe(self):
        row={c:0.0 for c in self.stocks}
        row.update({"Dehyton AB 30":10,"Dehyton PK 45":10,"Luviquat Excellence":2,"Arlypon TT":3})
        return pd.DataFrame([row])

    def test_percent_units_and_no_renormalization(self):
        f,_,_=build_features(self.recipe(),self.registry)
        self.assertAlmostEqual(f.surfactant_active_claim_lower_pct.iloc[0],7.4)
        self.assertAlmostEqual(f.surfactant_active_claim_upper_pct.iloc[0],7.6)
        self.assertAlmostEqual(f.declared_nacl_contribution_pct.iloc[0],0.7)
        self.assertAlmostEqual(f.added_water_balance_pct_estimate.iloc[0],75)
        self.assertAlmostEqual(f.water_known_partial_lower_pct.iloc[0],86.7)
        self.assertAlmostEqual(f.surfactant_pair_balance.iloc[0],1)
        self.assertAlmostEqual(f.interaction_type_grade_amphoteric__luviquat_excellence.iloc[0],0.004)

    def test_dilution_information_survives(self):
        a,_,_=build_features(self.recipe(),self.registry)
        b,_,_=build_features(self.recipe()*0.5,self.registry)
        self.assertAlmostEqual(b.surfactant_active_claim_lower_pct.iloc[0],a.surfactant_active_claim_lower_pct.iloc[0]*0.5)
        self.assertAlmostEqual(b.added_water_balance_pct_estimate.iloc[0],87.5)

    def test_supplier_values_are_a_separate_scenario(self):
        a,_,_=build_features(self.recipe(),self.registry)
        b,_,_=build_features(self.recipe(),self.registry,True)
        self.assertAlmostEqual(b.surfactant_active_claim_lower_pct.iloc[0],6.7)
        self.assertAlmostEqual(b.surfactant_active_claim_upper_pct.iloc[0],7.1)
        pd.testing.assert_frame_equal(a[self.stocks],b[self.stocks])

    def test_no_outcome_or_id_leakage(self):
        d=self.recipe();a,_,_=build_features(d,self.registry)
        d['sample_id']=999;d['stable']=1;d['turbidity_ntu']=1e9;d['rheology_type']='MAGIC';d['rheometer']='OTHER'
        b,_,_=build_features(d,self.registry)
        pd.testing.assert_frame_equal(a,b)

    def test_zero_surfactant_boundary_is_explicit(self):
        d=self.recipe()*0;f,_,_=build_features(d,self.registry)
        self.assertTrue(np.isfinite(f.to_numpy()).all())
        self.assertEqual(f.no_designated_surfactant.iloc[0],1)
        self.assertEqual(f.surfactant_pair_balance.iloc[0],0)

    def test_polyquaternium7_grades_not_collapsed(self):
        d=self.recipe();d['Luviquat Excellence']=0;d['Dehyquart CC7 Benz']=2
        a,_,_=build_features(d,self.registry)
        d['Dehyquart CC7 Benz']=0;d['Salcare Super 7']=2
        b,_,_=build_features(d,self.registry)
        self.assertEqual(a.grade_present_dehyquart_cc7_benz.iloc[0],1)
        self.assertEqual(b.grade_present_salcare_super_7.iloc[0],1)
        self.assertFalse(a.equals(b))

    def test_curve_interpolation_and_no_extrapolation(self):
        self.assertAlmostEqual(interp_log([10,1000],[1000,10],100),100)
        self.assertTrue(np.isnan(interp_log([10,90],[1000,100],100)))
        with self.assertRaises(ValueError):interp_log([10,1000],[1000,0],100)

    def test_invalid_recipe_rejected(self):
        d=self.recipe();d['Dehyton PK 45']=-1
        with self.assertRaises(ValueError):build_features(d,self.registry)


if __name__=='__main__': unittest.main()
