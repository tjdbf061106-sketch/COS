"""Verify the shared eye results; optionally replay one original outer fold.

This entry point needs only files inside this directory. The source_snapshot
directory preserves the original code; its original entry points are historical.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def csv(name):
    return pd.read_csv(ROOT / name, low_memory=False, float_precision='round_trip')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def point(d, policy):
    pred = d['y_pred_0_5' if policy == 'fixed_0.5' else 'y_pred_inner_BA']
    tn, fp, fn, tp = confusion_matrix(d.y_true, pred, labels=[0, 1]).ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return dict(n=len(d), positives=int(d.y_true.sum()), negatives=int((1-d.y_true).sum()),
                roc_auc=roc_auc_score(d.y_true, d.score_1),
                average_precision=average_precision_score(d.y_true, d.score_1),
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def verify():
    checks = {}
    manifest = read('manifest.json')
    for name, digest in manifest['sha256'].items():
        assert sha(ROOT/name) == digest, 'Shared file changed: '+name
    checks['shared_file_hashes'] = len(manifest['sha256'])
    data = csv('data/eye_retained699.csv')
    features = read('data/feature_names.json')
    gate = csv('tables/cohort_gate_794.csv').set_index('formulation_id')
    oof = csv('predictions/paired_oof_699.csv')
    metrics = csv('tables/metrics_summary.csv')
    retained_ids = set(gate.index[gate.in_baseline699])
    consistent_ids = set(gate.index[gate.in_exclusion693])
    assert len(data) == 699 and data.formulation_id.is_unique
    assert set(data.formulation_id) == retained_ids
    assert len(consistent_ids) == 693 and len(retained_ids-consistent_ids) == 6
    assert len(features) == 2280 and len(set(features)) == 2280
    assert data.binary_label.sum() == 183
    assert data.substance_type.value_counts().to_dict() == {'Mixture': 606, 'Chemical': 93}
    assert (data.final_label.le(2).astype(int) == data.binary_label).all()
    assert data.groupby('split_group_id').outer_fold.nunique().max() == 1
    checks['cohorts_labels_features_and_group_folds'] = 'PASS'
    for condition, pool_ids in [('retained699',retained_ids),('exclude693',consistent_ids)]:
        scored = oof[oof.condition.eq(condition)]
        assert len(scored) == 699 and set(scored.formulation_id) == retained_ids
        merged = scored.merge(data[['formulation_id','binary_label','outer_fold']],on='formulation_id',suffixes=('_oof','_input'),validate='1:1')
        assert (merged.y_true == merged.binary_label).all()
        assert (merged.outer_fold_oof == merged.outer_fold_input).all()
        assert np.array_equal(scored.y_pred_0_5, scored.score_1.ge(.5).astype(int))
        assert scored.score_1.between(0,1).all()
        for fold in range(5):
            pool = data[data.formulation_id.isin(pool_ids) & data.outer_fold.ne(fold)]
            held = data[data.outer_fold.eq(fold)]
            assert not set(pool.split_group_id) & set(held.split_group_id)
            inner = csv(f'splits/{condition}_fold{fold}_inner.csv')
            assert set(inner.formulation_id) == set(pool.formulation_id)
            assert inner.groupby('split_group_id').inner_fold.nunique().max() == 1
        card = read(f'models/binary_{condition}_card.json')
        assert card['feature_names'] == features
        assert set(card['training_ids']) == pool_ids
        assert card['model_sha256'] == sha(ROOT/f'models/binary_{condition}.joblib')
        assert card['threshold'] == .5 and not card['external_validation_performed']
    checks['paired_evaluation_inner_splits_and_model_cards'] = 'PASS'
    for _, expected in metrics.iterrows():
        frame = oof[oof.condition.eq(expected.condition)]
        if expected.subset == 'common693':
            frame = frame[frame.formulation_id.isin(consistent_ids)]
        elif expected.subset == 'FIFRA606':
            frame = frame[frame.source.eq('FIFRA data')]
        elif expected.subset == 'ICCVAM93':
            frame = frame[frame.source.eq('ICCVAM 2006')]
        elif expected.subset == 'multicomponent':
            frame = frame[frame.comp_count.gt(1)]
        elif expected.subset == 'binary_conflict6':
            frame = frame[frame.formulation_id.isin(retained_ids-consistent_ids)]
        else:
            assert expected.subset == 'all699', expected.subset
        actual = point(frame, expected.threshold_policy)
        for key, value in actual.items():
            assert np.isclose(value, expected[key], atol=1e-12, rtol=1e-10), (key,value,expected[key])
    checks['metric_rows_recomputed'] = len(metrics)
    reg = csv('formulation_reference/CTFA53_formulation_registry.csv')
    materials = csv('formulation_reference/CTFA53_curated_materials_570.csv')
    assert len(reg) == 53 and len(materials) == 570
    assert reg.raw_material_quantity_complete.sum() == 32
    assert not reg.external_validation_ready.any()
    checks['formulation_reference_scope'] = '53 candidates / 32 quantity-complete / 0 external-validation-ready'
    return data, features, gate, oof, checks

def refit_fold(fold, data, features, gate, oof):
    path = ROOT/'source_snapshot/stage3/modeling.py'
    spec = importlib.util.spec_from_file_location('shared_original_modeling', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    checks = {}
    seed = read('protocol.json')['seed_base'] + 100 + fold
    for condition, flag in [('retained699','in_baseline699'),('exclude693','in_exclusion693')]:
        ids = set(gate.index[gate[flag]])
        train = data[data.formulation_id.isin(ids) & data.outer_fold.ne(fold)]
        test = data[data.outer_fold.eq(fold)]
        model = base.make_model({'family':'random_forest','leaf':5}, seed)
        model.fit(train[features], train.binary_label.to_numpy(int))
        actual = model.predict_proba(test[features])[:,1]
        expected = oof[oof.condition.eq(condition)&oof.outer_fold.eq(fold)].set_index('formulation_id').loc[test.formulation_id,'score_1'].to_numpy()
        error = float(np.max(np.abs(actual-expected)))
        assert error < 1e-12, (condition,error)
        checks[condition] = {'fold':fold,'n_train':len(train),'n_test':len(test),'max_abs_score_difference':error}
    return checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit-fold',type=int,choices=range(5),help='Replay both training policies for one outer fold, fixed threshold 0.5.')
    parser.add_argument('--output',type=Path,help='Optional path to save this verification report.')
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        data,features,gate,oof,checks=verify()
        report={'status':'PASS','checks':checks,'scope':'Saved-result verification. Original confidence intervals are supplied, not newly re-estimated. No external validation.'}
        if args.refit_fold is not None:
            report['replayed_outer_fold']=refit_fold(args.refit_fold,data,features,gate,oof)
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__':
    main()
