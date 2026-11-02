from __future__ import annotations

import numpy as np
import pandas as pd

from build_review import BASE, ROOT, build_features, digest, interp_log, read_json, write_json


def run():
    checks=[]
    provenance=read_json(ROOT/"input_provenance.json")
    for record in provenance['records']:
        assert digest(ROOT/record['snapshot'])==record['sha256']
    for path,expected in provenance['baseline_files'].items():
        assert digest(BASE/path)==expected,path
    checks.append("Input snapshot hashes and all 46 baseline files unchanged")
    data=pd.read_csv(ROOT/"inputs/formulations_baseline.csv",float_precision='round_trip')
    raw=pd.DataFrame(read_json(ROOT/"inputs/raw_shampoo.json"))
    stocks=read_json(ROOT/"inputs/stock_feature_names.json")
    registry=pd.read_csv(ROOT/"data/ingredient_registry_reviewed.csv")
    features=pd.read_csv(ROOT/"data/features_source_reported_812.csv",float_precision='round_trip').set_index('sample_id')
    scenario=pd.read_csv(ROOT/"data/features_pk45_supplier_sensitivity_812.csv",float_precision='round_trip').set_index('sample_id')
    assert features.index.is_unique and features.index.tolist()==data.sample_id.tolist()
    assert len(features)==812 and features.shape[1]==114
    np.testing.assert_allclose(features[stocks],raw[stocks],atol=1e-12,rtol=0)
    assert np.isfinite(features.to_numpy()).all() and np.isfinite(scenario.to_numpy()).all()
    np.testing.assert_allclose(features[stocks].sum(axis=1)+features.added_water_balance_pct_estimate,100,atol=1e-10)
    for frame in [features,scenario]:
        assert (frame.surfactant_active_claim_lower_pct<=frame.surfactant_active_claim_upper_pct).all()
        np.testing.assert_allclose(frame.surfactant_active_claim_width_pct,frame.surfactant_active_claim_upper_pct-frame.surfactant_active_claim_lower_pct,atol=1e-12)
    calc,_,sets=build_features(data[stocks],registry)
    np.testing.assert_allclose(calc,features,atol=1e-12,rtol=0)
    # Independent perturbation demonstrates that predictors ignore observed outcomes and IDs.
    modified=data.copy();modified['sample_id']=np.arange(812)[::-1];modified['stable']=1-modified.stable
    modified['turbidity_ntu']=np.arange(812)*9999
    altered,_,_=build_features(modified,registry)
    pd.testing.assert_frame_equal(calc,altered)
    assert read_json(ROOT/"data/feature_sets.json")==sets
    prohibited={'sample_id','stable','rheometer','rheology_type','turbidity_ntu','log10p1_turbidity','log10_viscosity_100','ingredient_combination','polymer_thickener_group'}
    assert not prohibited.intersection(features.columns)
    assert len(sets['stock18'])==18 and len(sets['physics_core'])==61 and len(sets['physics_interactions'])==114
    checks.append("812 rows: original stocks and mass balance preserved; 18/61/114 predictor sets finite and unaffected by outcome/ID changes")
    pk=features['Dehyton PK 45']
    np.testing.assert_allclose(scenario.surfactant_active_claim_lower_pct-features.surfactant_active_claim_lower_pct,-0.07*pk,atol=1e-12)
    np.testing.assert_allclose(scenario.surfactant_active_claim_upper_pct-features.surfactant_active_claim_upper_pct,-0.05*pk,atol=1e-12)
    np.testing.assert_allclose(features.loc[pk==0],scenario.loc[pk==0],atol=1e-12)
    changed_columns=(abs(features-scenario)>1e-12).any()
    assert all('active_claim' in c for c in features.columns[changed_columns])
    assert int((pk>0).sum())==140
    checks.append("PK45 scenario: only active-range features change, exactly 140 affected recipes; no silent stock/target corrections")
    targets=pd.read_csv(ROOT/"data/targets_and_groups_812.csv",float_precision='round_trip')
    for c in ['stable','viscosity_100_mPa_s','log10_viscosity_100','turbidity_ntu','log10p1_turbidity']:
        np.testing.assert_allclose(targets[c],data[c],equal_nan=True,atol=1e-12)
    assert digest(ROOT/"data/outer_folds_preserved.csv")==digest(ROOT/"inputs/outer_folds_baseline.csv")
    assert not targets.loc[targets.stable==0,['turbidity_ntu','viscosity_100_mPa_s']].notna().any().any()
    assert targets.viscosity_100_mPa_s.notna().sum()==293 and targets.turbidity_ntu.notna().sum()==294
    checks.append("Targets, missingness and outer folds unchanged; 44-row review does not replace full training cohorts")
    review=pd.read_csv(ROOT/"review/formulations44_comparison.csv",float_precision='round_trip')
    expected=pd.read_csv(ROOT/"inputs/selected44_baseline.csv").sample_id
    assert len(review)==44 and set(review.sample_id)==set(expected)
    component=pd.read_csv(ROOT/"review/formulations44_ingredients_long.csv",float_precision='round_trip')
    assert len(component)==176 and component.groupby('sample_id').size().eq(4).all()
    for sid,g in component.groupby('sample_id'):
        source=data.set_index('sample_id').loc[sid]
        np.testing.assert_allclose(g.stock_pct,[source[n] for n in g.ingredient_id],atol=1e-12)
    lookup={r['ID']:r for r in read_json(ROOT/"inputs/raw_shampoo.json")}
    for row in review.itertuples(index=False):
        curve={k:v for part in lookup[row.sample_id]['Rheology_Data'] for k,v in part.items()}
        for rate in [10,100,500]:
            np.testing.assert_allclose(getattr(row,f'eta{rate}_mPa_s'),interp_log(curve['shear_rate'],curve['avg_viscosity'],rate),atol=1e-10)
        np.testing.assert_allclose(row.eta100_mPa_s,data.set_index('sample_id').loc[row.sample_id,'viscosity_100_mPa_s'],atol=1e-8)
    curves=pd.read_csv(ROOT/"review/formulations44_flow_curves.csv")
    assert set(curves.sample_id)==set(expected)
    representatives=pd.read_csv(ROOT/"review/subsystem_review_representatives.csv")
    assert representatives.polymer_thickener_group.is_unique and set(representatives.sample_id).issubset(set(expected))
    assert len(representatives)==7
    checks.append("44 selected recipes, 176 stock entries, original curves and 7 descriptive representatives traced to source")
    summary=read_json(ROOT/"summary.json")
    assert not summary['model_training_performed'] and not summary['skin_eye_predictions_performed']
    for c,n in summary['review_flag_counts'].items(): assert int(review[c].sum())==n
    dictionary=pd.read_csv(ROOT/"data/feature_dictionary.csv")
    assert dictionary.feature.tolist()==features.columns.tolist()
    assert dictionary.feature.is_unique and dictionary.definition.notna().all()
    result={'status':'PASS','checks':checks,'training_performed':False,'skin_eye_scoring_performed':False}
    write_json(ROOT/"verification.json",result);print(result)


if __name__=='__main__':run()
