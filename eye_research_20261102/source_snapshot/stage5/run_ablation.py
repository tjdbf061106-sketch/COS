"""Controlled binary feature ablation; reads the main experiment without modifying it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '4')

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent / '3. modeling'
FOLLOWUP = ROOT.parent / '4. followup'
spec = importlib.util.spec_from_file_location('main_modeling_readonly', MAIN / 'modeling.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SEED = base.SEED
MODELS = {
    'rf_leaf5': dict(name='rf_leaf5', family='random_forest', view='all', leaf=5),
    'lr_C0.1': dict(name='lr_C0.1', family='logistic', view='all', C=.1),
}
VIEW_PREFIXES = {
    'composition': ('comp_',),
    'composition_descriptors': ('comp_', 'desc_'),
    'composition_descriptors_fingerprints': ('comp_', 'desc_', 'fp_'),
    'full_with_aux': ('comp_', 'desc_', 'fp_', 'aux_'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    base.dump_json(path, value)


def save(frame, path):
    base.save_csv(frame, path)


def input_manifest():
    files = [base.DEFAULT_INPUT / name for name in
             ['eye_vectorized.csv', 'eye_vectorized_multionly.csv', 'model_feature_columns.json']]
    files += [p for folder in [MAIN, FOLLOWUP] for p in folder.rglob('*') if p.is_file()]
    return {str(p): sha(p) for p in sorted(set(files))}


def assert_originals_unchanged(manifest):
    for name, digest in manifest.items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Original file changed: {name}')


def frozen_json(path, obj):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != obj:
            raise ValueError(f'Cached run does not match {path.name}; use a new output directory.')
    else:
        dump(path, obj)


def views_for(features):
    # Preserve the main feature order so the full view is exactly the original matrix.
    views = {name: [c for c in features if c.startswith(prefixes)]
             for name, prefixes in VIEW_PREFIXES.items()}
    assert [len(v) for v in views.values()] == [12, 229, 2277, 2280]
    assert views['full_with_aux'] == features
    assert all(not any(token in c for token in ['label', 'source', 'group', 'conflict'])
               for cols in views.values() for c in cols)
    return views


def prepare_data(out):
    _, data, features, _ = base.read_inputs(base.DEFAULT_INPUT)
    reference = pd.read_csv(MAIN / 'splits' / 'outer_folds.csv').set_index('formulation_id')
    assert set(reference.index) == set(data.formulation_id)
    data['outer_fold'] = reference.loc[data.formulation_id, 'outer_fold'].to_numpy(int)
    for column in ['split_group_id', 'binary_label', 'final_label']:
        assert np.array_equal(data[column].to_numpy(), reference.loc[data.formulation_id, column].to_numpy())
    assert data.groupby('split_group_id').outer_fold.nunique().max() == 1
    generated = np.full(len(data), -1)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, test) in enumerate(splitter.split(data, data.final_label, data.split_group_id)):
        generated[test] = fold
    assert np.array_equal(generated, data.outer_fold.to_numpy())
    save(data[base.META + ['outer_fold']], out / 'splits' / 'outer_folds.csv')
    views = views_for(features)
    frozen_json(out / 'feature_views.json', views)
    return data, views


def inner_folds(train, scope, fold, out):
    path = MAIN / 'splits' / f'{scope}_binary_fold{fold}_inner.csv'
    ref = pd.read_csv(path).set_index('formulation_id')
    assert ref.index.is_unique and set(ref.index) == set(train.formulation_id)
    ids = ref.loc[train.formulation_id, 'inner_fold'].to_numpy(int)
    assert set(ids) == {0, 1, 2}
    splits = []
    for k in range(3):
        tr, va = np.flatnonzero(ids != k), np.flatnonzero(ids == k)
        assert not set(train.iloc[tr].split_group_id) & set(train.iloc[va].split_group_id)
        assert set(train.iloc[tr].binary_label) == {0., 1.}
        splits.append((tr, va))
    save(train[['formulation_id', 'split_group_id']].assign(inner_fold=ids),
         out / 'splits' / f'{scope}_fold{fold}_inner.csv')
    return ids, splits


def fitted_predictions(train, valid, columns, model_spec, seed):
    model = base.make_model(model_spec, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(train[columns], train.binary_label.to_numpy(int))
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Logistic regression did not converge; results are not accepted.')
    p = base.probabilities(model, valid[columns], 'binary')[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    imputer = model.named_steps['imputer']
    indicator_count = len(imputer.indicator_.features_)
    used_count = int(model.named_steps['constant_filter'].get_support().sum())
    return p, used_count, indicator_count


def validate_cache(frame, expected, scope, fold, model_name, view, inner=False):
    assert frame.formulation_id.is_unique
    assert frame.formulation_id.tolist() == expected.formulation_id.tolist()
    assert np.array_equal(frame.y_true.to_numpy(int), expected.binary_label.to_numpy(int))
    assert frame.split_group_id.tolist() == expected.split_group_id.tolist()
    for col, value in [('scope', scope), ('outer_fold', fold), ('model', model_name), ('view', view)]:
        assert frame[col].eq(value).all()
    assert np.isfinite(frame.score_1).all() and frame.score_1.between(0, 1).all()
    if not inner:
        assert np.array_equal(frame.y_pred_0_5, (frame.score_1 >= .5).astype(int))
        assert np.array_equal(frame.y_pred_inner_BA, (frame.score_1 >= frame.threshold_inner_BA).astype(int))


def run(out):
    started = time.monotonic()
    for folder in ['checks', 'splits', 'predictions', 'inner_predictions', 'tables', 'figures']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'original_manifest.json'
    if manifest_path.exists():
        originals = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert_originals_unchanged(originals)
    else:
        originals = input_manifest()
        dump(manifest_path, originals)
    frozen_json(out / 'run_code_hashes.json', {
        'run_ablation.py': sha(Path(__file__)), 'main_modeling.py': sha(MAIN / 'modeling.py')})
    packages = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'threadpoolctl']}
    frozen_json(out / 'environment.json', dict(python=sys.version, packages=packages))
    data, views = prepare_data(out)
    protocol = dict(
        task='binary: EPA I-II vs III-IV', seed=SEED,
        design='Post-hoc controlled feature ablation; fixed model configurations, no hyperparameter search.',
        models=MODELS, primary_model='rf_leaf5', robustness_model='lr_C0.1',
        model_settings='Original make_model; RF 200 trees / min leaf 5 / sqrt features; LR C=0.1; balanced weights.',
        view_counts={k: len(v) for k, v in views.items()},
        scopes=['all', 'multionly'], n_all=794, n_multionly=235,
        outer_folds='Original stored 5 group folds; regenerated equality also checked.',
        inner_folds='Original stored 3 group folds for each scope and outer fold.',
        fit_seed='SEED+100+outer_fold for outer fits; +inner_fold for inner fits; shared across feature views.',
        preprocessing='Median imputation with missing indicators and variance filter fitted on train only; LR scaling on train only.',
        primary_metrics=['roc_auc', 'average_precision'],
        primary_operating_point='Fixed 0.5 for all variants; isolates feature effects from threshold selection.',
        supplementary_operating_point='Each view chooses BA-maximizing threshold on inner OOF only; same grid/rule.',
        contrasts=[['composition', 'composition_descriptors'],
                   ['composition_descriptors', 'composition_descriptors_fingerprints'],
                   ['composition', 'composition_descriptors_fingerprints'],
                   ['composition_descriptors_fingerprints', 'full_with_aux']],
        bootstrap='2000 paired cluster resamples of fixed outer OOF predictions, grouped by split_group_id.',
        limitations=[
            'Existing folds and previously inspected data are reused; this is exploratory, not independent confirmation.',
            'Configurations are fixed using existing choices; no claim of optimal performance of each view.',
            'Composition baseline includes structure coverage and missingness; it is not structure-free.',
            'RF sqrt feature-subset policy is fixed, but number of candidate features changes with dimensionality.',
            'Feature effects are conditional on model and representation, not a causal or model-independent chemical contribution.',
            'FIFRA and mixture type are confounded; source-specific evaluation does not isolate all source effects.',
            'Reduced views can produce identical vectors in different groups; original full-feature grouping is retained for paired comparison.',
            'Bootstrap intervals condition on fitted OOF models; no retraining, split uncertainty or multiple-comparison correction.',
            'No cosmetic external validation, no final deployment model replacement.'])
    frozen_json(out / 'protocol.json', protocol)
    outputs = []
    for scope in ['all', 'multionly']:
        scoped = data if scope == 'all' else data[data.comp_count.gt(1)]
        for fold in range(5):
            train = scoped[scoped.outer_fold.ne(fold)].reset_index(drop=True)
            test = scoped[scoped.outer_fold.eq(fold)].reset_index(drop=True)
            assert not set(train.split_group_id) & set(test.split_group_id)
            ids, splits = inner_folds(train, scope, fold, out)
            for model_name, model_spec in MODELS.items():
                for view, cols in views.items():
                    tag = f'{scope}_fold{fold}_{model_name}_{view}'
                    cache = out / 'predictions' / f'{tag}.csv'
                    inner_cache = out / 'inner_predictions' / f'{tag}.csv'
                    if cache.exists() and inner_cache.exists():
                        rec = pd.read_csv(cache)
                        inc = pd.read_csv(inner_cache)
                        validate_cache(rec, test, scope, fold, model_name, view)
                        validate_cache(inc, train, scope, fold, model_name, view, inner=True)
                        assert np.array_equal(inc.inner_fold, ids)
                        threshold = base.threshold_from_oof(inc.y_true.to_numpy(int), inc.score_1.to_numpy())
                        assert np.allclose(rec.threshold_inner_BA, threshold, atol=1e-12, rtol=0)
                        outputs.append(rec)
                        print('RESUME', tag, flush=True)
                        continue
                    tick = time.monotonic()
                    seed = SEED + 100 + fold
                    inner_oof = np.full(len(train), np.nan)
                    for k, (tr, va) in enumerate(splits):
                        inner_oof[va], _, _ = fitted_predictions(
                            train.iloc[tr], train.iloc[va], cols, model_spec, seed + k)
                    threshold = base.threshold_from_oof(train.binary_label.to_numpy(int), inner_oof)
                    inc = train[['formulation_id', 'split_group_id']].copy()
                    inc = inc.assign(scope=scope, outer_fold=fold, model=model_name, view=view,
                                     inner_fold=ids, y_true=train.binary_label.to_numpy(int), score_1=inner_oof)
                    score, used_count, indicator_count = fitted_predictions(train, test, cols, model_spec, seed)
                    rec = test[base.META].copy()
                    rec = rec.assign(scope=scope, outer_fold=fold, model=model_name, view=view,
                                     y_true=test.binary_label.to_numpy(int), score_1=score,
                                     y_pred_0_5=(score >= .5).astype(int), threshold_inner_BA=threshold,
                                     y_pred_inner_BA=(score >= threshold).astype(int),
                                     n_train=len(train), n_features=len(cols),
                                     n_features_after_preprocessing=used_count, n_missing_indicators=indicator_count)
                    validate_cache(rec, test, scope, fold, model_name, view)
                    save(inc, inner_cache)
                    save(rec, cache)
                    outputs.append(rec)
                    print('DONE', tag, 'train', len(train), 'test', len(test),
                          f'{time.monotonic()-tick:.1f}s', flush=True)
    combined = pd.concat(outputs, ignore_index=True)
    assert len(combined) == (794 + 235) * len(MODELS) * len(views)
    assert not combined.duplicated(['scope', 'model', 'view', 'formulation_id']).any()
    save(combined, out / 'predictions' / 'oof_predictions.csv')
    assert_originals_unchanged(originals)
    dump(out / 'checks' / 'training_complete.json', dict(
        status='PASSED', total_prediction_rows=len(combined), outer_fits=80, inner_fits=240,
        outer_folds_match_main=True, inner_folds_match_main=True, group_overlap=0,
        original_files_unchanged=True, no_convergence_warnings=True,
        elapsed_seconds=time.monotonic()-started))
    print('TRAINING COMPLETE', len(combined), f'{time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.output_dir.resolve())
