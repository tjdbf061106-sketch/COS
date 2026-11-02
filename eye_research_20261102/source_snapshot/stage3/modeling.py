"""Reproducible, group-nested eye irritation modeling. See README.txt and report.html."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / '2. vectorized' / '\ucd5c\uc885'
SEED = 20261030
META = ['formulation_id', 'source', 'substance_type', 'comp_count', 'split_group_id',
        'final_label', 'binary_label', 'undeclared_pct', 'structure_coverage',
        'has_missing_pct', 'test_concentration', 'test_concentration_units',
        'identical_input_binary_conflict']

def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x:
        x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def read_inputs(directory):
    d = pd.read_csv(directory / 'eye_vectorized.csv', low_memory=False)
    multi = pd.read_csv(directory / 'eye_vectorized_multionly.csv', low_memory=False)
    schema = json.loads((directory / 'model_feature_columns.json').read_text(encoding='utf-8'))
    assert len(schema) == len(set(schema)) and set(schema).issubset(d.columns)
    assert d.formulation_id.is_unique and d.split_group_id.notna().all()
    assert set(multi.formulation_id) == set(d.loc[d.comp_count.gt(1), 'formulation_id'])
    pd.testing.assert_frame_equal(d.set_index('formulation_id').loc[multi.formulation_id, schema],
                                 multi.set_index('formulation_id')[schema], check_dtype=False)
    for col in ['final_label', 'binary_label', 'split_group_id', 'comp_count']:
        pd.testing.assert_series_equal(d.set_index('formulation_id').loc[multi.formulation_id, col],
                                      multi.set_index('formulation_id')[col], check_dtype=False)
    assert d.loc[d.binary_label.notna(), 'final_label'].isin([1, 2, 3, 4]).all()
    assert (d.loc[d.binary_label.notna(), 'binary_label'].astype(int).to_numpy() ==
            d.loc[d.binary_label.notna(), 'final_label'].le(2).astype(int).to_numpy()).all()
    # Retrieval provenance is not chemistry; label-derived conflict fields never enter X.
    features = [c for c in schema if c != 'aux_smiles_from_pubchem']
    numeric = d[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    d = pd.concat([d.drop(columns=features), numeric], axis=1).copy()
    a = d.loc[d.binary_label.notna()].reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(a[schema], index=False)
    assert a.assign(_hash=hashes).groupby('_hash').split_group_id.nunique().max() == 1
    for col in ['comp_group_id', 'structure_group_id']:
        assert a.groupby(col).split_group_id.nunique().max() == 1
    return d, a, features, schema

def candidate_specs():
    return [dict(name='dummy', family='dummy', view='composition'),
        dict(name='composition_lr', family='composition', view='composition', C=1.),
        *[dict(name=f'lr_C{c}', family='logistic', view='all', C=c) for c in [.1, 1.]],
        *[dict(name=f'rf_leaf{leaf}', family='random_forest', view='all', leaf=leaf) for leaf in [2, 5]],
        *[dict(name=f'et_leaf{leaf}', family='extra_trees', view='all', leaf=leaf) for leaf in [2, 5]],
        *[dict(name=f'hgb_leaves{leaves}', family='hist_boosting', view='descriptors', leaves=leaves) for leaves in [7, 15]]]

def feature_view(features, view):
    if view == 'composition': return [c for c in features if c.startswith('comp_')]
    if view == 'descriptors': return [c for c in features if not c.startswith('fp_')]
    return features

def make_model(spec, seed):
    fam = spec['family']
    steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
             ('constant_filter', VarianceThreshold())]
    # A source-held-out training set may consist entirely of neat chemicals, so
    # every composition field can be constant. Scaling retains valid zero columns;
    # the composition LR then learns only an intercept and the dummy still works.
    if fam in ['composition', 'dummy']:
        steps = steps[:1]
    if fam in ['logistic', 'composition']:
        steps.append(('scaler', StandardScaler()))
        clf = LogisticRegression(C=spec['C'], class_weight='balanced', solver='lbfgs', max_iter=5000,
                                 random_state=seed)
    elif fam in ['random_forest', 'extra_trees']:
        cls = RandomForestClassifier if fam == 'random_forest' else ExtraTreesClassifier
        clf = cls(n_estimators=200, min_samples_leaf=spec['leaf'], max_features='sqrt',
                  class_weight='balanced', n_jobs=4, random_state=seed)
    elif fam == 'hist_boosting':
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
            max_leaf_nodes=spec['leaves'], min_samples_leaf=15, l2_regularization=5.,
            class_weight='balanced', early_stopping=False, random_state=seed)
    else:
        clf = DummyClassifier(strategy='prior')
    return Pipeline(steps + [('classifier', clf)])

def labels(data, task):
    return data['binary_label' if task == 'binary' else 'final_label'].astype(int).to_numpy()

def probabilities(model, X, task):
    p = model.predict_proba(X)
    classes = np.array([0, 1] if task == 'binary' else [1, 2, 3, 4])
    assert np.array_equal(model.classes_, classes), model.classes_
    return p

def threshold_from_oof(y, p, sensitivity_target=None):
    thresholds = np.unique(np.r_[0., np.linspace(.01, .99, 99), 1.])
    pred = p[:, None] >= thresholds[None, :]
    sens = pred[y == 1].mean(axis=0)
    spec = (~pred[y == 0]).mean(axis=0)
    if sensitivity_target is None:
        score = (sens + spec) / 2
        possible = np.flatnonzero(np.isclose(score, score.max()))
    else:
        eligible = np.flatnonzero(sens >= sensitivity_target)
        possible = eligible[np.isclose(spec[eligible], spec[eligible].max())]
    j = possible[np.argmin(np.abs(thresholds[possible] - .5))]
    return float(thresholds[j])

def predictions(p, task, threshold=.5):
    return (p[:, 1] >= threshold).astype(int) if task == 'binary' else p.argmax(axis=1) + 1

def metrics(y, pred, p, task, weights=None):
    classes = [0, 1] if task == 'binary' else [1, 2, 3, 4]
    cm = confusion_matrix(y, pred, labels=classes, sample_weight=weights)
    denom = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), denom, out=np.full(len(classes), np.nan), where=denom != 0)
    result = dict(n=len(y), accuracy=accuracy_score(y, pred, sample_weight=weights),
        balanced_accuracy=float(np.nanmean(recall)),
        macro_f1=f1_score(y, pred, labels=classes, average='macro', zero_division=0, sample_weight=weights))
    if task == 'binary':
        tn, fp, fn, tp = cm.ravel()
        result.update(sensitivity=recall[1], specificity=recall[0],
            f1=f1_score(y, pred, zero_division=0, sample_weight=weights),
            precision=tp/(tp+fp) if tp+fp else 0., npv=tn/(tn+fn) if tn+fn else 0.,
            mcc=matthews_corrcoef(y, pred, sample_weight=weights),
            roc_auc=roc_auc_score(y,p[:,1],sample_weight=weights) if len(np.unique(y))==2 else np.nan,
            average_precision=average_precision_score(y,p[:,1],sample_weight=weights) if np.any(y==1) else np.nan,
            prevalence=float(np.average(y,weights=weights)),
            brier=brier_score_loss(y,p[:,1],sample_weight=weights), tn=tn,fp=fp,fn=fn,tp=tp)
    else:
        result.update(ordinal_mae=float(np.average(np.abs(y-pred),weights=weights)),
                      quadratic_kappa=cohen_kappa_score(y,pred,labels=classes,weights='quadratic',sample_weight=weights))
        result.update({f'recall_epa_{c}':recall[c-1] for c in classes})
    return result

def check_split(data, train, test):
    assert len(set(train) & set(test)) == 0
    assert not set(data.iloc[train].split_group_id) & set(data.iloc[test].split_group_id)
    assert set(data.iloc[train].final_label.astype(int)) == {1,2,3,4}

def tune(data, features, task, seed, output, tag):
    y = labels(data, task)
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
        data, data.final_label.astype(int), data.split_group_id))
    split_log = []
    for k,(tr,va) in enumerate(splits):
        check_split(data,tr,va)
        for j in va: split_log.append(dict(formulation_id=data.iloc[j].formulation_id,inner_fold=k))
    save_csv(pd.DataFrame(split_log), output/'splits'/f'{tag}_inner.csv')
    results=[]
    for spec in candidate_specs():
        cols=feature_view(features,spec['view']); oof=np.zeros((len(data),2 if task=='binary' else 4))
        fold_scores=[]
        for k,(tr,va) in enumerate(splits):
            model=make_model(spec,seed+k)
            model.fit(data.iloc[tr][cols],y[tr]); p=probabilities(model,data.iloc[va][cols],task)
            oof[va]=p
            fold_scores.append(average_precision_score(y[va],p[:,1]) if task=='binary' else
                               f1_score(y[va],p.argmax(axis=1)+1,labels=[1,2,3,4],average='macro',zero_division=0))
        t=threshold_from_oof(y,oof[:,1]) if task=='binary' else .5
        ts=threshold_from_oof(y,oof[:,1],.8) if task=='binary' else .5
        results.append(dict(spec=spec,score=float(np.mean(fold_scores)),threshold=t,
                            screen_threshold=ts,inner_scores=fold_scores))
    dump_json(output/'tuning'/f'{tag}.json',results)
    winners=[]
    for family in dict.fromkeys(s['family'] for s in candidate_specs()):
        winners.append(max((r for r in results if r['spec']['family']==family),key=lambda r:r['score']))
    eligible=[r for r in winners if r['spec']['family'] not in ['dummy','composition']]
    chosen=max(eligible,key=lambda r:r['score'])
    return winners, chosen

def tanimoto(a,b):
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    dot=a@b.T; denom=(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-dot
    return np.divide(dot,denom,out=np.zeros_like(dot),where=denom>0).clip(0,1)

def applicability(train,test,fp_cols):
    within=tanimoto(train[fp_cols],train[fp_cols])
    same=train.split_group_id.to_numpy()[:,None] == train.split_group_id.to_numpy()[None,:]
    within[same]=-1.
    cut=float(np.quantile(within.max(1),.05))
    sim=tanimoto(test[fp_cols],train[fp_cols]); near=sim.argmax(1); best=sim.max(1)
    return dict(max_tanimoto=best,ad_cutoff=cut,in_structural_ad=best>=cut,
                nearest_training_id=train.formulation_id.to_numpy()[near],
                group_seen_in_training=test.split_group_id.isin(train.split_group_id).to_numpy())

def evaluate_fold(train,test,features,task,seed,out,tag,scope,fold):
    winners, chosen=tune(train,features,task,seed,out,tag)
    y=labels(train,task); result=[]; importances=[]
    ad=applicability(train,test,[c for c in features if c.startswith('fp_')])
    selected_model=None
    for best in winners:
        spec=best['spec']; cols=feature_view(features,spec['view'])
        model=make_model(spec,seed); model.fit(train[cols],y)
        p=probabilities(model,test[cols],task)
        rec=test[META].reset_index(drop=True).copy()
        rec['task']=task; rec['scope']=scope; rec['outer_fold']=fold
        rec['model']=spec['family']; rec['configuration']=spec['name']
        rec['y_true']=labels(test,task); rec['y_pred']=predictions(p,task,best['threshold'])
        rec['threshold']=best['threshold']; rec['inner_score']=best['score']
        for j,c in enumerate([0,1] if task=='binary' else [1,2,3,4]): rec[f'score_{c}']=p[:,j]
        if task=='binary':
            rec['y_pred_0_5']=predictions(p,task,.5)
            rec['screen_threshold']=best['screen_threshold']
            rec['y_pred_screen80']=predictions(p,task,best['screen_threshold'])
        for key,value in ad.items():rec[key]=value
        rec['high_undeclared_fraction']=rec.undeclared_pct.gt(50)
        rec['partial_structure']=rec.structure_coverage.lt(1)
        result.append(rec)
        if spec['name']==chosen['spec']['name']:
            selected_model=model
            selected=rec.copy();selected['model']='nested_selected';result.append(selected)
            # Block permutation is evaluated only on the held-out outer fold.
            rng=np.random.default_rng(seed)
            base=average_precision_score(labels(test,task),p[:,1]) if task=='binary' else f1_score(
                labels(test,task),predictions(p,task),labels=[1,2,3,4],average='macro',zero_division=0)
            for block in ['fingerprints','descriptors','composition','structure_aux']:
                bcols=[c for c in cols if c.startswith({'fingerprints':'fp_','descriptors':'desc_',
                                'composition':'comp_','structure_aux':'aux_'}[block])]
                if not bcols:continue
                for rep in range(5):
                    xp=test[cols].copy();xp[bcols]=xp[bcols].to_numpy()[rng.permutation(len(xp))]
                    pp=probabilities(model,xp,task)
                    score=average_precision_score(labels(test,task),pp[:,1]) if task=='binary' else f1_score(
                        labels(test,task),predictions(pp,task),labels=[1,2,3,4],average='macro',zero_division=0)
                    importances.append(dict(task=task,scope=scope,outer_fold=fold,block=block,
                        repeat=rep,importance=base-score,selected_configuration=spec['name']))
    return pd.concat(result,ignore_index=True),pd.DataFrame(importances),chosen,selected_model

def audit(d,a,features,schema,input_dir,out):
    manifest={name:dict(path=str(input_dir/name),sha256=hashlib.sha256((input_dir/name).read_bytes()).hexdigest())
        for name in ['eye_vectorized.csv','eye_vectorized_multionly.csv','model_feature_columns.json']}
    previous=out/'input_manifest.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        if any(old.get(k,{}).get('sha256') != v['sha256'] for k,v in manifest.items()):
            raise ValueError('Inputs changed. Use a new --output-dir instead of reusing cached folds.')
    dump_json(previous,manifest)
    missing=d[features].isna().sum(); groups=a.groupby('split_group_id')
    info=dict(total_rows=len(d),labeled_rows=len(a),unlabeled_rows=int(d.binary_label.isna().sum()),
        multi_rows=int(d.comp_count.gt(1).sum()),all_feature_count=len(schema),used_feature_count=len(features),
        excluded_features={'aux_smiles_from_pubchem':'Data acquisition provenance; not a chemical property.'},
        label_counts=a.final_label.value_counts().sort_index().to_dict(),
        binary_counts=a.binary_label.value_counts().sort_index().to_dict(),
        multi_label_counts=a.loc[a.comp_count.gt(1),'final_label'].value_counts().sort_index().to_dict(),
        labeled_groups=groups.ngroups,multi_groups=a.loc[a.comp_count.gt(1),'split_group_id'].nunique(),
        single_declared_mixture_rows=int(((d.substance_type=='Mixture')&d.comp_count.eq(1)).sum()),
        labeled_chemicals=int(a.substance_type.eq('Chemical').sum()),
        binary_conflicting_groups=int(groups.binary_label.nunique().gt(1).sum()),
        exact_input_conflict_rows=int(a.identical_input_binary_conflict.sum()),
        missing_features=missing[missing.gt(0)].to_dict(),
        test_concentration_known=int(a.test_concentration.notna().sum()),
        input_validation='PASSED: subset equality, unique IDs, EPA mapping, group containment, exact-vector grouping')
    dump_json(out/'data_audit.json',info)
    save_csv(pd.crosstab(a.source,a.final_label).reset_index(),out/'tables'/'source_label_counts.csv')
    save_csv(a.loc[a.identical_input_binary_conflict,META+['component_names']],out/'tables'/'label_conflicts.csv')
    dump_json(out/'feature_columns.json',features)
    return info

def run(input_dir,out):
    started=time.time()
    verified=out/'checks'/'verification.json'
    if verified.exists():
        recorded=json.loads(verified.read_text(encoding='utf-8')).get('code_sha256',{}).get('modeling.py')
        current=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if recorded and recorded != current:
            raise ValueError('Modeling code changed after verification. Use a new --output-dir.')
    for sub in ['splits','tuning','predictions','models','tables','figures','checks']: (out/sub).mkdir(parents=True,exist_ok=True)
    d,a,features,schema=read_inputs(input_dir); audit(d,a,features,schema,input_dir,out)
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text(encoding='utf-8'))
        if old['seed'] != SEED or old['candidates'] != candidate_specs():
            raise ValueError('Seed/candidates changed. Use a new --output-dir for a new experiment.')
    dump_json(protocol_path,dict(seed=SEED,outer_folds=5,inner_folds=3,
        group_column='split_group_id',outer_stratification='EPA 4-class',
        binary_selection='mean inner-fold average precision; threshold maximizes pooled inner-OOF balanced accuracy',
        multiclass_selection='mean inner-fold macro F1',candidates=candidate_specs(),
        selection_eligible=['logistic','random_forest','extra_trees','hist_boosting'],
        structural_ad='5th percentile of train nearest other-group weighted-fingerprint Tanimoto',
        auxiliary_quality_flag='undeclared_pct > 50 (heuristic; not a validated reliability cutoff)',
        no_external_cosmetics_validation=True))
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib','threadpoolctl']}
    dump_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=versions))
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(a,a.final_label.astype(int),a.split_group_id))
    fold_ids=np.full(len(a),-1)
    for fold,(tr,te) in enumerate(splits):check_split(a,tr,te);fold_ids[te]=fold
    split_table=a[META].copy();split_table['outer_fold']=fold_ids;save_csv(split_table,out/'splits'/'outer_folds.csv')
    all_preds=[]; all_importances=[]; selections=[]
    for scope in ['all','multionly']:
        for task in ['binary','multiclass']:
            for fold,(tr,te) in enumerate(splits):
                tag=f'{scope}_{task}_fold{fold}'; cache=out/'predictions'/f'{tag}.csv'
                if cache.exists():
                    all_preds.append(pd.read_csv(cache));all_importances.append(pd.read_csv(out/'tables'/f'{tag}_importance.csv'))
                    print('RESUME',tag,flush=True);continue
                train=a.iloc[tr].copy();test=a.iloc[te].copy()
                if scope=='multionly':train=train[train.comp_count.gt(1)];test=test[test.comp_count.gt(1)]
                print('START',tag,'train',len(train),'test',len(test),flush=True)
                pred,imp,chosen,_=evaluate_fold(train.reset_index(drop=True),test.reset_index(drop=True),features,
                    task,SEED+100+fold,out,tag,scope,fold)
                save_csv(pred,cache);save_csv(imp,out/'tables'/f'{tag}_importance.csv')
                all_preds.append(pred);all_importances.append(imp)
                print('DONE',tag,'selected',chosen['spec']['name'],'elapsed',round(time.time()-started,1),flush=True)
            # Deployment refit uses fresh 3-fold group tuning on all eligible records.
            tag=f'{scope}_{task}_final'; model_path=out/'models'/f'{tag}.joblib'
            if model_path.exists(): print('RESUME',tag,flush=True);continue
            train=a if scope=='all' else a[a.comp_count.gt(1)].copy()
            print('REFIT',tag,flush=True)
            _,chosen=tune(train.reset_index(drop=True),features,task,SEED+1000,out,tag)
            spec=chosen['spec'];cols=feature_view(features,spec['view']);model=make_model(spec,SEED)
            model.fit(train[cols],labels(train,task))
            fp=[c for c in features if c.startswith('fp_')]
            bundle=dict(model=model,features=cols,fp_features=fp,task=task,scope=scope,
                threshold=chosen['threshold'],screen_threshold=chosen['screen_threshold'],configuration=spec,
                score_note='Uncalibrated model scores; not experimentally validated probabilities.',
                train_ad=train[['formulation_id','split_group_id']+fp].copy(),
                train_ids=train.formulation_id.tolist(),versions=versions)
            joblib.dump(bundle,model_path,compress=3)
            dump_json(out/'models'/f'{tag}_metadata.json',{k:v for k,v in bundle.items() if k not in ['model','train_ad']})
            if scope=='all':
                un=d[d.binary_label.isna()].copy();p=probabilities(model,un[cols],task)
                rec=un[META].copy();rec['predicted_label']=predictions(p,task,chosen['threshold'])
                for j,c in enumerate([0,1] if task=='binary' else [1,2,3,4]):rec[f'score_{c}']=p[:,j]
                for k,v in applicability(train,un,fp).items():rec[k]=v
                rec['is_experimental_label']=False
                save_csv(rec,out/'predictions'/f'unlabeled_{task}_exploratory.csv')
    save_csv(pd.concat(all_preds,ignore_index=True),out/'predictions'/'oof_predictions.csv')
    save_csv(pd.concat(all_importances,ignore_index=True),out/'tables'/'block_permutation_importance.csv')
    # Source-held-out evaluation: all matching training groups are purged before tuning.
    source_preds=[]
    for source in sorted(a.source.unique()):
        tag='source_'+source.split()[0].lower(); cache=out/'predictions'/f'{tag}.csv'
        if cache.exists():source_preds.append(pd.read_csv(cache));continue
        test=a[a.source.eq(source)].copy(); train=a[~a.source.eq(source)&~a.split_group_id.isin(test.split_group_id)].copy()
        print('SOURCE HOLDOUT',source,'train',len(train),'test',len(test),flush=True)
        _,chosen=tune(train.reset_index(drop=True),features,'binary',SEED+2000,out,tag)
        spec=chosen['spec'];cols=feature_view(features,spec['view']);model=make_model(spec,SEED)
        model.fit(train[cols],labels(train,'binary'));p=probabilities(model,test[cols],'binary')
        rec=test[META].copy();rec['heldout_source']=source;rec['n_train']=len(train)
        rec['selected_configuration']=spec['name'];rec['y_true']=labels(test,'binary')
        rec['y_pred']=predictions(p,'binary',chosen['threshold']);rec['score_0']=p[:,0];rec['score_1']=p[:,1]
        rec['threshold']=chosen['threshold']
        for k,v in applicability(train,test,[c for c in features if c.startswith('fp_')]).items():rec[k]=v
        assert not rec.group_seen_in_training.any()
        save_csv(rec,cache);source_preds.append(rec)
    save_csv(pd.concat(source_preds,ignore_index=True),out/'predictions'/'source_holdout_predictions.csv')
    dump_json(out/'checks'/'training_complete.json',dict(completed=True,elapsed_seconds=time.time()-started,
        oof_group_overlap=0,source_holdout_group_overlap=0))
    print('TRAINING COMPLETE',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.input_dir,args.output_dir)
