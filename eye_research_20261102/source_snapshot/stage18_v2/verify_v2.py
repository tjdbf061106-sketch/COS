import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from common_v2 import ROOT, STAGE, PROJECT, ARCHIVE, PREP, MAIN, read_json, inputs, modules, dump, sha, freeze
from cohort_v2 import conflict_flags


def main():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    data, features = inputs()
    ab, aa, base = modules()
    lock_hash = freeze()
    gate = pd.read_csv(ROOT / 'tables/cohort_gate_794.csv').set_index('formulation_id')
    # Regression cases detect the precise semantic bug, independent of old flags.
    toy = pd.DataFrame(dict(comp_group_id=['a','a','b','b','c','c','d','d'],
                            source=['kept']*7+['Yamaguchi et al. 2016'],
                            final_label=[1,2,3,4,2,3,1,4], binary_label=[1,1,0,0,1,0,1,0]))
    keep, binary, multi = conflict_flags(toy)
    check('I_vs_II_not_a_binary_conflict', not binary.iloc[:2].any() and multi.iloc[:2].all())
    check('III_vs_IV_not_a_binary_conflict', not binary.iloc[2:4].any() and multi.iloc[2:4].all())
    check('II_vs_III_is_a_binary_conflict', binary.iloc[4:6].all())
    check('conflicts_recomputed_after_retracted_source_removal', not binary.iloc[6:].any() and not keep.iloc[7])
    raw = data[~data.source.eq('Yamaguchi et al. 2016')]
    true_groups = set(raw.groupby('comp_group_id').binary_label.nunique().loc[lambda s:s>1].index)
    true_ids = set(raw.loc[raw.comp_group_id.isin(true_groups), 'formulation_id'])
    baseline_ids = set(gate.index[gate.in_baseline699])
    exclusion_ids = set(gate.index[gate.in_exclusion693])
    check('baseline699_equals_all_nonretracted_labels', baseline_ids == set(raw.formulation_id) and len(baseline_ids)==699)
    check('693_exclusion_matches_actual_binary_labels', baseline_ids-exclusion_ids == true_ids and len(exclusion_ids)==693)
    check('three_binary_conflict_groups_six_rows', len(true_groups)==3 and len(true_ids)==6)
    restored = gate[gate.restored_from_old679]
    check('14_binary_consistent_records_restored', len(restored)==14 and restored.in_exclusion693.all()
          and not restored.binary_composition_conflict.any() and int(restored.binary_label.sum())==6)
    check('no_retracted_training_rows', not data[data.formulation_id.isin(baseline_ids|exclusion_ids)].source.str.startswith('Yamaguchi').any())
    check('existing_conflicts_are_already_in_same_folds', gate.loc[list(true_ids)].groupby('comp_group_id').outer_fold.nunique().eq(1).all())
    matrix = data.set_index('formulation_id')[features]
    check('binary_conflict_feature_pairs_really_identical', all(
        np.allclose(matrix.loc[g.index].iloc[0], matrix.loc[g.index].iloc[1], equal_nan=True)
        for _, g in gate.loc[list(true_ids)].groupby('comp_group_id')))
    check('label_definitions_unchanged', np.array_equal(gate.loc[data.formulation_id,'binary_label'].to_numpy(),data.binary_label.to_numpy())
          and np.array_equal(gate.loc[data.formulation_id,'final_label'].to_numpy(),data.final_label.to_numpy()))

    paired = pd.read_csv(ROOT / 'predictions/paired_oof_699.csv')
    all_split_checks = all_threshold_checks = cache_ok = seed_ok = train_scope_ok = True
    for cond, ids in [('retained699',baseline_ids),('exclude693',exclusion_ids)]:
        frame = paired[paired.condition.eq(cond)]
        check(cond+'_699_unique_evaluation_rows', len(frame)==699 and frame.formulation_id.is_unique and set(frame.formulation_id)==baseline_ids)
        for fold in range(5):
            name = f'{cond}_fold{fold}'
            out = frame[frame.outer_fold.eq(fold)]
            tr = data[data.formulation_id.isin(ids) & data.outer_fold.ne(fold)]
            inc = pd.read_csv(ROOT / f'inner_predictions/{name}.csv')
            cache = read_json(ROOT / f'cache/{name}.json')
            cache_ok &= cache['training_lock_sha256']==lock_hash and all(sha(ROOT/f)==h for f,h in cache['sha256'].items())
            train_scope_ok &= set(inc.formulation_id)==set(tr.formulation_id) and out.n_train.eq(len(tr)).all()
            all_split_checks &= not bool(set(tr.split_group_id)&set(out.split_group_id))
            all_split_checks &= inc.groupby('split_group_id').inner_fold.nunique().eq(1).all()
            seed_ok &= out.outer_seed.eq(ab.SEED+100+fold).all()
            seed_ok &= inc.seed.eq(ab.SEED+100+fold+inc.inner_fold).all()
            threshold=base.threshold_from_oof(inc.y_true.to_numpy(int),inc.score_1.to_numpy())
            all_threshold_checks &= np.allclose(out.threshold_inner_BA,threshold)
            all_threshold_checks &= out.y_pred_0_5.eq(out.score_1.ge(.5).astype(int)).all()
            all_threshold_checks &= out.y_pred_inner_BA.eq(out.score_1.ge(threshold).astype(int)).all()
    check('outer_and_inner_groups_do_not_cross', all_split_checks)
    check('same_seed_schedule_in_both_training_policies',seed_ok)
    check('inner_thresholds_use_training_rows_only',train_scope_ok and all_threshold_checks)
    check('all_ten_fold_caches_bound_to_exact_code_and_inputs',cache_ok)
    check('paired_rows_labels_and_folds_match', all(
        paired[paired.condition.eq('retained699')].sort_values('formulation_id')[c].reset_index(drop=True).equals(
        paired[paired.condition.eq('exclude693')].sort_values('formulation_id')[c].reset_index(drop=True))
        for c in ['formulation_id','y_true','outer_fold','split_group_id','outer_seed']))

    # All five baseline folds reproduce the pre-existing fixed-seed stage8 run.
    previous = pd.concat([pd.read_csv(PROJECT/f'8. source_sensitivity/predictions/removed_fold{k}.csv') for k in range(5)]).set_index('formulation_id')
    current = paired[paired.condition.eq('retained699')].set_index('formulation_id')
    baseline_maxdiff=float(np.max(np.abs(current.score_1-previous.loc[current.index,'score_1'])))
    check('all_baseline_scores_reproduce_stage8_same_seed',baseline_maxdiff<1e-12)
    # One corrected condition fold independently replayed; the other folds are checked for inputs, seeds and provenance.
    tr=data[data.formulation_id.isin(exclusion_ids)&data.outer_fold.ne(0)].reset_index(drop=True)
    te=data[data.formulation_id.isin(baseline_ids)&data.outer_fold.eq(0)].reset_index(drop=True)
    replay,_,_=ab.fitted_predictions(tr,te,features,ab.MODELS['rf_leaf5'],ab.SEED+100)
    cached=paired[paired.condition.eq('exclude693')].set_index('formulation_id').loc[te.formulation_id,'score_1'].to_numpy()
    replay_maxdiff=float(np.max(np.abs(replay-cached)))
    check('corrected_fold0_independent_replay',replay_maxdiff<1e-12)

    metrics=pd.read_csv(ROOT/'tables/metrics_summary.csv')
    metrics_ok=True
    for _,r in metrics.iterrows():
        f=paired[paired.condition.eq(r.condition)]
        masks={'all699':np.ones(len(f),dtype=bool),'common693':f.formulation_id.isin(exclusion_ids),
               'FIFRA606':f.source.eq('FIFRA data'),'ICCVAM93':f.source.eq('ICCVAM 2006'),
               'multicomponent':f.comp_count.gt(1),'binary_conflict6':f.formulation_id.isin(true_ids)}
        f=f[masks[r.subset]]
        col='y_pred_0_5' if r.threshold_policy=='fixed_0.5' else 'y_pred_inner_BA'
        tn,fp,fn,tp=confusion_matrix(f.y_true,f[col],labels=[0,1]).ravel()
        metrics_ok &= np.isclose(roc_auc_score(f.y_true,f.score_1),r.roc_auc) and len(f)==r.n
        metrics_ok &= (tn,fp,fn,tp)==(r.tn,r.fp,r.fn,r.tp)
    check('all_reported_auc_and_confusion_counts_recomputed',metrics_ok)
    a=paired[paired.condition.eq('retained699')].sort_values('formulation_id').reset_index(drop=True)
    b=paired[paired.condition.eq('exclude693')].sort_values('formulation_id').reset_index(drop=True)
    weights=aa.bootstrap_weights(a)
    delta=[]
    for w in weights:
        if w[a.y_true.eq(1)].sum() and w[a.y_true.eq(0)].sum():
            delta.append(roc_auc_score(b.y_true,b.score_1,sample_weight=w)-roc_auc_score(a.y_true,a.score_1,sample_weight=w))
    diff=pd.read_csv(ROOT/'tables/paired_differences.csv')
    r=diff.query("subset=='all699' and threshold_policy=='fixed_0.5' and metric=='roc_auc'").iloc[0]
    check('paired_auc_group_bootstrap_independently_recomputed',np.allclose(np.quantile(delta,[.025,.975]),[r.ci_low,r.ci_high],atol=1e-12)
          and r.valid_bootstraps==len(delta))
    model_checks=True
    for cond,ids in [('retained699',baseline_ids),('exclude693',exclusion_ids)]:
        card=read_json(ROOT/f'models/binary_{cond}_card.json')
        model_path=ROOT/f'models/binary_{cond}.joblib'
        model=joblib.load(model_path)
        params=model.named_steps['classifier'].get_params()
        model_checks &= card['model_sha256']==sha(model_path) and card['training_lock_sha256']==lock_hash
        model_checks &= set(card['training_ids'])==ids and card['n_train']==len(ids) and card['feature_names']==features
        model_checks &= params['random_state']==ab.SEED+100 and params['n_estimators']==200 and params['min_samples_leaf']==5
        model_checks &= card['threshold']==.5 and not card['use_as_cosmetics_safety_model']
    check('both_model_cards_match_artifacts_scope_threshold_and_training_ids',model_checks)

    reg=pd.read_csv(PREP/'CTFA53_formulation_registry.csv')
    mat=pd.read_csv(PREP/'CTFA53_curated_materials_570.csv')
    cls=pd.read_csv(ROOT/'tables/cosmetics_material_class_summary.csv')
    positive_ids=set(reg.loc[reg.epa_binary_I_II_positive.eq(1),'formulation_id'])
    counts_ok=True
    for _,r in cls.iterrows():
        ids=set(mat.loc[mat.material_class.eq(r.material_class),'formulation_id'])
        counts_ok &= r.n_formulations==len(ids) and r.positive_formulations==len(ids&positive_ids)
        counts_ok &= r.negative_formulations==len(ids-positive_ids)
    check('positive_formulation_counts_use_unique_ids_all_classes',counts_ok)
    check('exact_nonwater_81_rows_are_19_positive_formulations',cls.set_index('material_class').loc['exact_nonwater','positive_formulations']==19)
    check('fragrance_4_rows_are_3_positive_formulations',cls.set_index('material_class').loc['undisclosed_fragrance_mixture','positive_formulations']==3)
    cos=read_json(ROOT/'cosmetics_summary.json')
    check('cosmetics_diagnostics_use51_recipe_groups_and_are_not_validation',cos['recipe_groups']==51 and not cos['new_cosmetics_model_fitted'])
    check('no_cosmetics_rows_in_training',not bool(set(reg.formulation_id)&baseline_ids))
    actual=pd.read_csv(ROOT/'tables/historical_cosmetics_loo_scores.csv').set_index('formulation_id')
    old=pd.read_csv(STAGE/'tables/cosmetics_loo_scores.csv').set_index('formulation_id')
    check('historical_cosmetics_scores_preserved',all(np.allclose(actual[c],old.loc[actual.index,c])
          for c in ['score_year_only','score_ingredients_only','score_year_plus_ingredients']))
    # Check actual artifacts against dated archives; existence alone is not preservation evidence.
    archive_rows=read_json(ARCHIVE/'file_manifest.json')
    check('old_stage18_archive_hashes_match',all(sha(ARCHIVE/r['snapshot'])==r['sha256'] for r in archive_rows))
    prior=read_json(PROJECT/'시행착오/2026-10-31_통합보관/file_manifest.json')
    protected=[r for r in prior if r['source'].startswith(tuple(f'{i}. ' for i in range(13,18)))
               or r['source'] in ['3. modeling/models/all_binary_final.joblib','8. source_sensitivity/tables/metrics_summary.csv']]
    check('stages13_17_and_original_model_hashes_unchanged',all(sha(PROJECT/r['source'])==r['sha256'] for r in protected))
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,n_checks=len(checks),
                baseline_max_absolute_difference=baseline_maxdiff,corrected_fold0_max_absolute_difference=replay_maxdiff,
                training_lock_sha256=lock_hash,code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
                protected_upstream_files=len(protected),prior_stage18_archived_files=len(archive_rows))
    dump('checks/verification.json',result)
    print('VERIFICATION',result['status'],sum(checks.values()),'/',len(checks),flush=True)
    assert all(checks.values()),[k for k,v in checks.items() if not v]


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
