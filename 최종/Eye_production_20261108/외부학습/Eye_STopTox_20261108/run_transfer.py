"""Exact production-fold paired comparison with newly learned STopTox features."""
from common import *
import argparse, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from concentration_features import fit_concentration_median

_CACHE=None

def data():
    global _CACHE
    if _CACHE is None:
        assert sha(E.DATA)==EXPECTED
        df,X,cols=E.load_eye()
        aux=np.load(HERE/'aux_features.npy')
        assert aux.shape==(673,4) and np.isfinite(aux).all()
        _CACHE=(df,X,aux)
    return _CACHE

def pair(df,X,aux,tr,va):
    a,b,median=P.feature_pair(df,X,tr,va,'concentration')
    return np.column_stack([a,aux[tr]]),np.column_stack([b,aux[va]]),median

def run_fold(job):
    started=time.time();folder=Path(job['folder']);identity=job['identity']
    if (folder/'outer.json').exists():return u.load_record(folder/'outer.json',identity)
    df,X,aux=data();fold=job['split'];y=E.labels_of(df)[job['label']];model=job['model']
    inside=[(part,*pair(df,X,aux,part['train'],part['validation'])) for part in fold['inner']]
    candidates=[]
    for candidate in u.candidate_grid(model):
        path=folder/'candidates'/(candidate['candidate_id']+'.json')
        if path.exists():
            result=u.load_record(path,identity);assert result['candidate']==candidate
        else:
            scores=np.full(len(df),np.nan);aucs=[]
            for part,a,b,median in inside:
                est=u.fit_weighted(u.build_estimator(model,candidate['params']),a,y[part['train']])
                s=u.predict_scores(est,b,model)
                scores[part['validation']]=s
                aucs.append(float(u.roc_auc_score(y[part['validation']],s)))
            cal=fold['calibration'];assert np.isfinite(scores[cal]).all()
            result={'candidate':candidate,'mean_auc':float(np.mean(aucs)),
                'fold_aucs':aucs,'calibration_rows':cal,'calibration_scores':scores[cal].tolist()}
            u.save_record(path,identity,result)
        candidates.append(result)
    best=max(candidates,key=lambda x:x['mean_auc'])
    cut=u.threshold(best['calibration_scores'],y[fold['calibration']])
    a,b,median=pair(df,X,aux,fold['train'],fold['test'])
    est=u.fit_weighted(u.build_estimator(model,best['candidate']['params']),a,y[fold['train']])
    scores=u.predict_scores(est,b,model)
    result={k:job[k] for k in ['arm','label','model','seed','fold']}
    result.update(test=fold['test'],y_true=y[fold['test']].tolist(),scores=scores.tolist(),
        predictions=(scores>=cut).astype(int).tolist(),threshold=float(cut),
        selected_candidate=best['candidate'],candidates_evaluated=len(candidates),
        outer_concentration_median=median,inner_concentration_medians=[v[3] for v in inside],
        train_rows=fold['train'],calibration_rows=fold['calibration'],elapsed_seconds=time.time()-started)
    u.save_record(folder/'outer.json',identity,result)
    return result

def audit_control(config,old_id,df):
    """Reuse only after checking all 20 candidates, selected cutoff and row identity."""
    results=[];checked=[]
    yall=E.labels_of(df)
    for label in ['A','B']:
        for model in MODELS:
            for seed in E.SEEDS:
                for fi,fold in enumerate(config['splits'][str(seed)]):
                    folder=OLD/'concentration'/label/model/str(seed)/f'fold-{fi}'
                    r=u.load_record(folder/'outer.json',old_id)
                    assert r['test']==fold['test'] and r['train_rows']==fold['train']
                    assert r['calibration_rows']==fold['calibration']
                    assert r['y_true']==yall[label][fold['test']].tolist()
                    assert r['candidates_evaluated']==20
                    candidates=[u.load_record(folder/'candidates'/(c['candidate_id']+'.json'),old_id) for c in config['grids'][model]]
                    for c,expected in zip(candidates,config['grids'][model]):
                        assert c['candidate']==expected and c['calibration_rows']==fold['calibration']
                        assert c['mean_auc']==float(np.mean(c['fold_aucs']))
                    best=max(candidates,key=lambda v:v['mean_auc'])
                    assert r['selected_candidate']==best['candidate']
                    assert r['threshold']==u.threshold(best['calibration_scores'],yall[label][fold['calibration']])
                    assert r['predictions']==(np.array(r['scores'])>=r['threshold']).astype(int).tolist()
                    assert r['outer_concentration_median']==fit_concentration_median(df.iloc[fold['train']])
                    assert r['inner_concentration_medians']==[fit_concentration_median(df.iloc[f['train']]) for f in fold['inner']]
                    r=dict(r,arm='control')
                    results.append(r)
                    checked.append({'path':str(folder/'outer.json'),'sha256':sha(folder/'outer.json')})
    return results,checked

def summarize(results,df):
    evaluation=np.flatnonzero(df.evaluation_role.eq('existing_evaluation_pool')).tolist()
    rows=[];oof=[];fmetrics=[]
    frame=pd.DataFrame(results)
    for key,g in frame.groupby(['arm','label','model','seed'],sort=True):
        parts=g.sort_values('fold');ix=np.concatenate(parts.test.to_list())
        assert sorted(ix.tolist())==evaluation
        yt=np.concatenate(parts.y_true.to_list());score=np.concatenate(parts.scores.to_list());pred=np.concatenate(parts.predictions.to_list())
        rows.append(dict(zip(['arm','label','model','seed'],key),**u.compute_metrics(yt,score,pred)))
        for r in parts.to_dict('records'):
            fmetrics.append({k:r[k] for k in ['arm','label','model','seed','fold','threshold']}|u.compute_metrics(r['y_true'],r['scores'],r['predictions']))
            for i,y,s,p in zip(r['test'],r['y_true'],r['scores'],r['predictions']):
                oof.append(dict(zip(['arm','label','model','seed'],key),fold=r['fold'],
                    row_index=i,formulation_id=df.iloc[i].formulation_id,y_true=y,score=s,prediction=p,threshold=r['threshold']))
    metrics=pd.DataFrame(rows);paired=metrics.pivot(index=['label','model','seed'],columns='arm',values='MCC').reset_index()
    paired['delta_MCC']=paired.stoptx_aux-paired.control
    summary=paired.groupby(['label','model']).agg(control_MCC_mean=('control','mean'),
        control_MCC_sd=('control','std'),aux_MCC_mean=('stoptx_aux','mean'),aux_MCC_sd=('stoptx_aux','std'),
        delta_MCC_mean=('delta_MCC','mean'),delta_MCC_sd=('delta_MCC','std'),
        delta_MCC_min=('delta_MCC','min'),delta_MCC_max=('delta_MCC','max'),
        positive_seed_count=('delta_MCC',lambda x:int((x>0).sum()))).reset_index()
    for name,d in [('metrics_by_seed.csv',metrics),('paired_delta_by_seed.csv',paired),
        ('paired_summary.csv',summary),('outer_oof_predictions.csv',pd.DataFrame(oof)),('metrics_by_fold.csv',pd.DataFrame(fmetrics))]:
        csv(HERE/name,d)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();assert 1<=args.workers<=12
    lock=HERE/'transfer.running'
    # Exclusive lock prevents an accidental duplicate invocation; stale locks are
    # preserved and require checking the recorded process before any recovery.
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    started=time.time()
    try:
        config,old_id=old_config();df,X,aux=data();prepared=read(HERE/'prepared.json')
        assert prepared['aux_sha256']==sha(HERE/'aux_features.npy')
        assert prepared['protocol_sha256']==sha(HERE/'protocol.json')
        controls,checked=audit_control(config,old_id,df)
        run_config={'parent_config_sha256':sha(OLD/'config.json'),'prepared':prepared,
            'code_hashes':{p.name:sha(p) for p in [HERE/'common.py',HERE/'prepare.py',HERE/'run_transfer.py']},
            'models':MODELS,'splits':config['splits'],'groups':config['groups'],
            'grids':{m:u.candidate_grid(m) for m in MODELS},'teacher_sha256':sha(HERE/'stoptx_teacher.joblib')}
        identity=u.hash_json(run_config);u.save_record(HERE/'run_config.json',identity,run_config)
        u.save_record(HERE/'control_reuse_audit.json',identity,{'checked_outer_jobs':len(controls),
            'checked_candidates':len(controls)*20,'sources':checked,'all_checks_passed':True})
        jobs=[]
        # Interleave model families so all are active without changing candidates.
        for label in ['A','B']:
            for seed in E.SEEDS:
                for fi,fold in enumerate(config['splits'][str(seed)]):
                    for model in MODELS:
                        jobs.append(dict(identity=identity,arm='stoptx_aux',label=label,model=model,
                            seed=seed,fold=fi,split=fold,folder=str(HERE/'runs'/label/model/str(seed)/f'fold-{fi}')))
        print('START',len(jobs),'new outer jobs; reused',len(controls),'audited controls',flush=True)
        results=list(controls)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_fold,j):j for j in jobs}
            for n,f in enumerate(as_completed(futures),1):
                r=f.result();results.append(r)
                print('DONE',n,'/',len(jobs),r['label'],r['model'],r['seed'],r['fold'],
                    round(time.time()-started,1),'sec',flush=True)
        summary=summarize(results,df)
        write(HERE/'training_complete.json',{'status':'complete','new_outer_jobs':len(jobs),
            'reused_control_outer_jobs':len(controls),'new_candidate_jobs':len(jobs)*20,
            'new_target_model_fits':len(jobs)*61,'elapsed_seconds':time.time()-started,
            'evaluation_rows_per_arm_seed':606,'independent_external_evaluation_rows':0})
        print(summary.to_string(index=False),flush=True)
    finally:
        lock.unlink()

if __name__=='__main__':
    main()
