import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from common_v2 import ROOT, MAIN, PROJECT, inputs, modules, read_json, dump, save, freeze, sha

CONDITIONS = {'retained699': 'in_baseline699', 'exclude693': 'in_exclusion693'}


def point(frame, pred_column):
    y = frame.y_true.to_numpy(int)
    pred = frame[pred_column].to_numpy(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    sens = tp / (tp+fn) if tp+fn else np.nan
    spec = tn / (tn+fp) if tn+fp else np.nan
    return dict(n=len(frame), positives=int(y.sum()), negatives=int((1-y).sum()),
                roc_auc=roc_auc_score(y, frame.score_1) if both else np.nan,
                average_precision=average_precision_score(y, frame.score_1) if both else np.nan,
                sensitivity=sens, specificity=spec, balanced_accuracy=(sens+spec)/2,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def cache_valid(name, lock_hash, expected_test, expected_train):
    meta_path = ROOT / f'cache/{name}.json'
    if not meta_path.exists():
        return False
    meta = read_json(meta_path)
    assert meta['training_lock_sha256'] == lock_hash
    assert meta['test_ids'] == expected_test and meta['train_ids'] == expected_train
    for rel, digest in meta['sha256'].items():
        assert sha(ROOT / rel) == digest, f'Cache modified: {rel}'
    return True


def main():
    ab, aa, base = modules()
    data, features = inputs()
    gate = pd.read_csv(ROOT / 'tables/cohort_gate_794.csv').set_index('formulation_id')
    lock_hash = freeze()
    plan = read_json(ROOT / 'protocol.json')
    evaluation = data[data.formulation_id.isin(gate.index[gate.in_baseline699])].copy()
    all_predictions = []
    for condition, flag in CONDITIONS.items():
        pool = data[data.formulation_id.isin(gate.index[gate[flag]])].copy()
        for fold in range(5):
            train = pool[pool.outer_fold.ne(fold)].reset_index(drop=True)
            test = evaluation[evaluation.outer_fold.eq(fold)].reset_index(drop=True)
            assert not set(train.split_group_id) & set(test.split_group_id)
            ref = pd.read_csv(MAIN / f'splits/all_binary_fold{fold}_inner.csv').set_index('formulation_id')
            inner_fold = ref.loc[train.formulation_id, 'inner_fold'].to_numpy(int)
            split = train[['formulation_id', 'split_group_id']].assign(inner_fold=inner_fold)
            assert split.groupby('split_group_id').inner_fold.nunique().max() == 1
            name = f'{condition}_fold{fold}'
            outer_path, inner_path = f'predictions/{name}.csv', f'inner_predictions/{name}.csv'
            if cache_valid(name, lock_hash, test.formulation_id.tolist(), train.formulation_id.tolist()):
                rec = pd.read_csv(ROOT / outer_path)
                print('CACHE', name, flush=True)
            else:
                scores_inner = np.full(len(train), np.nan)
                for k in range(3):
                    tr, va = train.iloc[np.flatnonzero(inner_fold != k)], train.iloc[np.flatnonzero(inner_fold == k)]
                    assert not set(tr.split_group_id) & set(va.split_group_id)
                    scores_inner[inner_fold == k], _, _ = ab.fitted_predictions(
                        tr, va, features, ab.MODELS['rf_leaf5'], plan['seed_base']+100+fold+k)
                threshold = base.threshold_from_oof(train.binary_label.to_numpy(int), scores_inner)
                scores, used, missing = ab.fitted_predictions(
                    train, test, features, ab.MODELS['rf_leaf5'], plan['seed_base']+100+fold)
                rec = test[['formulation_id', 'source', 'substance_type', 'comp_count', 'split_group_id', 'outer_fold']].copy()
                rec['condition'] = condition
                rec['y_true'] = test.binary_label.to_numpy(int)
                rec['score_1'] = scores
                rec['y_pred_0_5'] = (scores >= .5).astype(int)
                rec['threshold_inner_BA'] = threshold
                rec['y_pred_inner_BA'] = (scores >= threshold).astype(int)
                rec['n_train'] = len(train)
                rec['outer_seed'] = plan['seed_base']+100+fold
                rec['features_after_preprocessing'] = used
                rec['missing_indicators'] = missing
                inc = split.assign(y_true=train.binary_label.to_numpy(int), score_1=scores_inner,
                                   seed=plan['seed_base']+100+fold+inner_fold)
                save(outer_path, rec)
                save(inner_path, inc)
                save(f'splits/{name}_inner.csv', split)
                files = [outer_path, inner_path, f'splits/{name}_inner.csv']
                dump(f'cache/{name}.json', dict(training_lock_sha256=lock_hash,
                     test_ids=test.formulation_id.tolist(), train_ids=train.formulation_id.tolist(),
                     sha256={f: sha(ROOT / f) for f in files}))
                print('FIT', name, 'train', len(train), 'test', len(test), flush=True)
            all_predictions.append(rec)
        # Save a research model, with the same fixed threshold and transparent identity manifest.
        model_path = ROOT / f'models/binary_{condition}.joblib'
        card_path = ROOT / f'models/binary_{condition}_card.json'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if card_path.exists():
            card = read_json(card_path)
            assert card['training_lock_sha256'] == lock_hash and sha(model_path) == card['model_sha256']
        else:
            model = base.make_model(ab.MODELS['rf_leaf5'], plan['final_fit_seed'])
            model.fit(pool[features], pool.binary_label.to_numpy(int))
            joblib.dump(model, model_path)
            dump(f'models/binary_{condition}_card.json', dict(
                role='research_reference_baseline' if condition == 'retained699' else 'research_conflict_exclusion_sensitivity',
                condition=condition, n_train=len(pool), positives=int(pool.binary_label.sum()), n_features=len(features),
                training_ids=pool.formulation_id.tolist(), feature_names=features,
                model_spec=plan['model'], seed=plan['final_fit_seed'], threshold=.5,
                threshold_origin='fixed_in_protocol_not_optimized_on_outer_evaluation',
                input_semantics='legacy_partial_declared_composition_not_complete_formulation',
                external_validation_performed=False, cosmetics_transfer_performed=False,
                use_as_cosmetics_safety_model=False, training_lock_sha256=lock_hash, model_sha256=sha(model_path)))
    oof = pd.concat(all_predictions, ignore_index=True)
    save('predictions/paired_oof_699.csv', oof)
    summaries, differences, intervals = [], [], []
    retained = oof[oof.condition.eq('retained699')].sort_values('formulation_id').reset_index(drop=True)
    excluded = oof[oof.condition.eq('exclude693')].sort_values('formulation_id').reset_index(drop=True)
    assert retained.formulation_id.tolist() == excluded.formulation_id.tolist()
    consistent_ids = set(gate.index[gate.in_exclusion693])
    masks = {'all699': np.ones(len(retained), dtype=bool),
             'common693': retained.formulation_id.isin(consistent_ids).to_numpy(),
             'FIFRA606': retained.source.eq('FIFRA data').to_numpy(),
             'ICCVAM93': retained.source.eq('ICCVAM 2006').to_numpy(),
             'multicomponent': retained.comp_count.gt(1).to_numpy(),
             'binary_conflict6': ~retained.formulation_id.isin(consistent_ids).to_numpy()}
    for subset, mask in masks.items():
        frames = {'retained699': retained[mask].reset_index(drop=True), 'exclude693': excluded[mask].reset_index(drop=True)}
        # Counts are printed for six ambiguous rows, without pretending the tiny set is an adequate test.
        weights = np.vstack([np.ones(mask.sum()), aa.bootstrap_weights(frames['retained699'])]) if subset != 'binary_conflict6' else None
        for policy, col in [('fixed_0.5', 'y_pred_0_5'), ('inner_BA', 'y_pred_inner_BA')]:
            vals = {}
            for condition, frame in frames.items():
                summaries.append(dict(subset=subset, condition=condition, threshold_policy=policy, **point(frame, col)))
                if weights is not None:
                    vals[condition] = aa.weighted_metrics(frame, weights, col)
                    for metric, values in vals[condition].items():
                        lo, hi, n = aa.ci(values[1:])
                        intervals.append(dict(subset=subset, condition=condition, threshold_policy=policy,
                                              metric=metric, value=values[0], ci_low=lo, ci_high=hi, valid_bootstraps=n))
            if weights is not None:
                for metric in vals['retained699']:
                    delta = vals['exclude693'][metric] - vals['retained699'][metric]
                    lo, hi, n = aa.ci(delta[1:])
                    differences.append(dict(subset=subset, threshold_policy=policy, metric=metric,
                                            contrast='exclude693_minus_retained699', delta=delta[0],
                                            ci_low=lo, ci_high=hi, valid_bootstraps=n))
    save('tables/metrics_summary.csv', pd.DataFrame(summaries))
    save('tables/confidence_intervals.csv', pd.DataFrame(intervals))
    save('tables/paired_differences.csv', pd.DataFrame(differences))
    dump('training_summary.json', dict(status='FITTED', conditions=list(CONDITIONS),
         training_pool_sizes={'retained699': 699, 'exclude693': 693}, common_evaluation_rows=699,
         conditional_evaluation_rows=693, shared_seed_policy=plan['shared_outer_seed'],
         final_research_models=2, primary_threshold=.5, no_outer_threshold_optimization=True,
         external_validation=False, cosmetics_transfer=False, training_lock_sha256=lock_hash))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
