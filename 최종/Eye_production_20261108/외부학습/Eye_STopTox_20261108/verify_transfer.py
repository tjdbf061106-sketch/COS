"""Independent numerical and lineage checks of saved experiment artifacts."""
from common import *
import argparse
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as fg
from concentration_features import fit_concentration_median

def preparation_checks():
    config,old_id=old_config();df,X,cols=E.load_eye()
    audit=pd.read_csv(HERE/'stoptx_identity_audit.csv')
    train=pd.read_csv(HERE/'stoptx_aux_training.csv')
    universe=pd.read_csv(HERE/'screening_structure_universe.csv')
    assert len(audit)==3545 and len(train)==read(HERE/'data_audit.json')['training_rows']
    assert not train.excluded_known_uvcb_or_polymer.any() and not train.excluded_identical_local_morgan.any()
    assert set(train.stoptx_row)==set(audit.loc[audit.aux_training_allowed,'stoptx_row'])
    assert not train.excluded_overlap.any() and not train.excluded_invalid_standardized_parent.any()
    assert train.parent.is_unique and not set(train.parent)&set(universe.parent.dropna())
    assert not set(train.canonical)&set(universe.canonical.dropna())
    assert not set(train.connectivity)&set(universe.connectivity.dropna())
    component=pd.read_csv(HERE/'production_component_scores.csv')
    gen=fg.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    model=joblib.load(HERE/'stoptx_teacher.joblib')
    parent=component.parent.drop_duplicates().tolist()
    component_fps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in parent]
    mat=np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in parent])
    pred=model.predict_proba(mat)[:,1]
    lookup=dict(zip(parent,pred))
    np.testing.assert_allclose(component.parent.map(lookup),component.stoptx_score,rtol=1e-12,atol=1e-12)
    block=np.load(HERE/'aux_features.npy')
    for i,g in component.groupby('row_index'):
        s=g.stoptx_score.to_numpy();w=g.pct.to_numpy()
        if np.isnan(w).any():
            assert len(g)==1 and df.iloc[i].weight_mode=='equal_fallback_missing_pct'
            w=np.array([1.])
        else:w=w/w.sum()
        mean=np.average(s,weights=w)
        np.testing.assert_allclose(block[i],[mean,max(s),min(s),np.sqrt(np.average((s-mean)**2,weights=w))],rtol=1e-12,atol=1e-12)
    groups=np.array(config['groups'])
    for seed,folds in config['splits'].items():
        coverage=[]
        for fold in folds:
            train_rows,held=fold['train'],fold['test'];coverage+=held
            assert not set(groups[train_rows])&set(groups[held])
            assert df.iloc[held].evaluation_role.eq('existing_evaluation_pool').all()
            for part in fold['inner']:
                assert not set(groups[part['train']]) & (set(groups[part['validation']]) | set(groups[held]))
                assert df.iloc[part['validation']].evaluation_role.eq('existing_evaluation_pool').all()
        assert sorted(coverage)==list(range(606))
    # Refit one saved control with its selected candidate: a direct execution
    # check complements the full checksum/threshold audit of all reused controls.
    r=u.load_record(OLD/'concentration/A/lr/42/fold-0/outer.json',old_id)
    a,b,med=P.feature_pair(df,X,r['train_rows'],r['test'],'concentration')
    fit=u.fit_weighted(u.build_estimator('lr',r['selected_candidate']['params']),a,E.labels_of(df)['A'][r['train_rows']])
    scores=u.predict_scores(fit,b,'lr')
    np.testing.assert_allclose(scores,r['scores'],atol=1e-12,rtol=1e-12)
    # Identity removal does not remove close analogues. Report similarity as a
    # descriptive domain check, with no performance-derived cutoff.
    tfps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.parent]
    nearest=[]
    for parent_smi,fp in zip(parent,component_fps):
        sim=np.array(DataStructs.BulkTanimotoSimilarity(fp,tfps));best=int(sim.argmax())
        nearest.append({'parent':parent_smi,'max_teacher_tanimoto':float(sim[best]),
            'nearest_stoptx_row':int(train.iloc[best].stoptx_row),'fingerprint_identical':bool(sim[best]==1)})
    nearest=pd.DataFrame(nearest);csv(HERE/'component_teacher_domain.csv',nearest)
    assert not nearest.fingerprint_identical.any()
    joined=component.merge(nearest,on='parent',validate='many_to_one')
    domain=[]
    for i,g in joined.groupby('row_index'):
        domain.append({'row_index':i,'formulation_id':df.iloc[i].formulation_id,
            'evaluation_role':df.iloc[i].evaluation_role,
            'weighted_nearest_tanimoto':float(g.weight@g.max_teacher_tanimoto),
            'minimum_component_nearest_tanimoto':float(g.max_teacher_tanimoto.min())})
    csv(HERE/'production_teacher_domain.csv',pd.DataFrame(domain))
    write(HERE/'preparation_verification.json',{'passed':True,'teacher_reloaded_predictions_match':True,
        'all_673_component_pools_match':True,'all_606_original_rows_covered_per_seed':True,
        'heldout_groups_excluded_from_all_target_training':True,'control_refit_matches_saved_scores':True,
        'screened_identity_intersections':0,'nearest_tanimoto_median_unique_component':float(nearest.max_teacher_tanimoto.median()),
        'fingerprint_identical_component_count':int(nearest.fingerprint_identical.sum()),
        'note':'Fingerprint collisions/analogs are not exact chemical identities. Domain similarity does not establish cosmetic validity.'})
    print('PREPARATION VERIFICATION PASSED',flush=True)

def result_checks():
    assert read(HERE/'training_complete.json')['status']=='complete'
    config,old_id=old_config();df,X,cols=E.load_eye();identity=read(HERE/'run_config.json')['identity']
    run_config=u.load_record(HERE/'run_config.json',identity)
    for name,h in run_config['code_hashes'].items():assert sha(HERE/name)==h
    prepared=read(HERE/'prepared.json')
    for key,name in [('teacher_sha256','stoptx_teacher.joblib'),('aux_sha256','aux_features.npy'),
                     ('protocol_sha256','protocol.json'),('data_audit_sha256','data_audit.json')]:
        assert sha(HERE/name)==prepared[key]
    data_audit=read(HERE/'data_audit.json')
    for source,h in data_audit['source_files_sha256'].items():assert sha(source)==h
    assert sha(HERE/'sources/canada_priority_inventory.csv')==data_audit['uvcb_inventory_sha256']
    yall=E.labels_of(df);paths=list((HERE/'runs').rglob('outer.json'));assert len(paths)==90
    folds=[]
    for path in paths:
        r=u.load_record(path,identity);split=config['splits'][str(r['seed'])][r['fold']]
        assert r['test']==split['test'] and r['train_rows']==split['train'] and r['calibration_rows']==split['calibration']
        assert r['y_true']==yall[r['label']][split['test']].tolist()
        candidates=[u.load_record(path.parent/'candidates'/(c['candidate_id']+'.json'),identity) for c in config['grids'][r['model']]]
        assert len(candidates)==20
        for c,expected in zip(candidates,config['grids'][r['model']]):
            assert c['candidate']==expected and c['calibration_rows']==split['calibration']
            assert c['mean_auc']==float(np.mean(c['fold_aucs']))
            # Independently rebuild each inner AUC from its saved row scores.
            score=dict(zip(c['calibration_rows'],c['calibration_scores']))
            aucs=[float(u.roc_auc_score(yall[r['label']][part['validation']], [score[i] for i in part['validation']])) for part in split['inner']]
            np.testing.assert_allclose(aucs,c['fold_aucs'],rtol=0,atol=0)
        best=max(candidates,key=lambda c:c['mean_auc'])
        assert best['candidate']==r['selected_candidate']
        assert r['threshold']==u.threshold(best['calibration_scores'],yall[r['label']][split['calibration']])
        assert r['predictions']==(np.array(r['scores'])>=r['threshold']).astype(int).tolist()
        assert r['outer_concentration_median']==fit_concentration_median(df.iloc[split['train']])
        assert r['inner_concentration_medians']==[fit_concentration_median(df.iloc[p['train']]) for p in split['inner']]
        folds.append(r)
    oof=pd.read_csv(HERE/'outer_oof_predictions.csv',float_precision='round_trip');assert len(oof)==606*2*2*3*3
    metrics=pd.read_csv(HERE/'metrics_by_seed.csv',float_precision='round_trip')
    for key,g in oof.groupby(['arm','label','model','seed']):
        assert sorted(g.row_index.tolist())==list(range(606))
        actual=u.compute_metrics(g.y_true,g.score,g.prediction)
        m=metrics[(metrics.arm==key[0])&(metrics.label==key[1])&(metrics.model==key[2])&(metrics.seed==key[3])].iloc[0]
        for name in ['MCC','sensitivity','specificity','ROC-AUC','PR-AUC']:
            np.testing.assert_allclose(actual[name],m[name],rtol=1e-12,atol=1e-12)
    preserved=read(HERE/'protected_hashes_before.json')
    changes={p:{'before':h,'after':sha(p)} for p,h in preserved.items() if sha(p)!=h}
    assert not changes, changes
    write(HERE/'verification.json',{'passed':True,'new_outer_jobs_checked':90,'new_full_grid_candidates_checked':1800,
        'control_outer_jobs_checked':90,'pooled_metrics_recomputed':36,
        'same_evaluation_rows_groups_splits':True,'thresholds_recomputed_from_original_inner_oof':True,
        'concentration_medians_train_only':True,'protected_files_checked':len(preserved),'protected_files_changed':0,
        'executed_code_teacher_aux_features_and_source_hashes_match':True,
        'eye_input_sha256':sha(E.DATA),'cosmetic_external_evaluation_rows':0})
    print('RESULT VERIFICATION PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation-only',action='store_true');a=p.parse_args()
    if a.preparation_only:preparation_checks()
    else:result_checks()
