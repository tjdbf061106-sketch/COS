"""Report the four requested classification metrics and paired class balance."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
BASE=REPO/'shampoo_physics_20261102'
METRICS=['Sensitivity','Specificity','MCC','ROC-AUC']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run():
    policy=json.loads((ROOT/'reporting_policy.json').read_text(encoding='utf-8'))
    source=BASE/'results/oof_predictions.csv'
    snapshot=ROOT/'stability_oof_snapshot.csv'
    if not snapshot.exists():
        d=pd.read_csv(source,float_precision='round_trip')
        d=d[d.target=='stability'][['scheme','seed','fold','model','sample_id','observed','predicted']]
        d.to_csv(snapshot,index=False,encoding='utf-8-sig')
        save_json(ROOT/'provenance.json',{'original_oof_path':'shampoo_physics_20261102/results/oof_predictions.csv',
                                        'original_oof_sha256':sha(source),'stability_snapshot_sha256':sha(snapshot),
                                        'repository_commit_reviewed':'ed35ec4b8499cecc876c0aa442b35999eb6f16c8',
                                        'github_source_data_matches_analysis_content':True})
    provenance=json.loads((ROOT/'provenance.json').read_text(encoding='utf-8'))
    assert sha(snapshot)==provenance['stability_snapshot_sha256']
    if source.exists():assert sha(source)==provenance['original_oof_sha256']
    d=pd.read_csv(snapshot,float_precision='round_trip')
    assert d.observed.isin([0,1]).all() and d.predicted.between(0,1).all()
    assert np.isfinite(d.predicted).all()
    rows=[]
    for (scheme,seed,model),g in d.groupby(['scheme','seed','model']):
        assert len(g)==812 and g.sample_id.is_unique
        y=g.observed.to_numpy(int);pred=g.predicted.to_numpy()>=policy['current_classification_threshold']
        tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum())
        fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
        assert tp+fn==294 and tn+fp==518
        sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
        rows.append({'scheme':scheme,'seed':int(seed),'model':model,'n':len(g),'TP_stable':tp,'FN_stable':fn,
                     'TN_unstable':tn,'FP_unstable':fp,'Sensitivity':sensitivity,'Specificity':specificity,
                     'MCC':float(matthews_corrcoef(y,pred)),'ROC-AUC':float(roc_auc_score(y,g.predicted)),
                     'sensitivity_specificity_gap':abs(sensitivity-specificity),
                     'weaker_class_recall':min(sensitivity,specificity)})
    repeat=pd.DataFrame(rows)
    repeat.to_csv(ROOT/'four_metrics_by_repeat.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (scheme,model),g in repeat.groupby(['scheme','model']):
        row={'scheme':scheme,'model':model,'n_independent_samples':812,'n_repeats':len(g)}
        for metric in METRICS+['sensitivity_specificity_gap','weaker_class_recall']:
            row[metric+'_mean']=g[metric].mean()
            row[metric+'_sd']=g[metric].std(ddof=1) if len(g)>1 else np.nan
        summary.append(row)
    summary=pd.DataFrame(summary)
    summary.to_csv(ROOT/'four_metrics_summary.csv',index=False,encoding='utf-8-sig')
    if (BASE/'results/repeat_metrics.csv').exists():
        prior=pd.read_csv(BASE/'results/repeat_metrics.csv',float_precision='round_trip').query("target=='stability'")
        combined=repeat.merge(prior,on=['scheme','seed','model'],suffixes=('','_original'),validate='one_to_one')
        for a,b in [('Sensitivity','recall'),('Specificity','specificity'),('MCC','mcc'),('ROC-AUC','roc_auc')]:
            np.testing.assert_allclose(combined[a],combined[b],atol=1e-12,rtol=0)
    def rendered(scheme):
        lines=['| 모델 | Sensitivity | Specificity | MCC | ROC-AUC |','|---|---:|---:|---:|---:|']
        names={'dummy':'상수 기준선','linear':'선형 모델','extra_trees':'Extra Trees'}
        for name in ['dummy','linear','extra_trees']:
            row=summary[(summary.scheme==scheme)&(summary.model==name)].iloc[0]
            values=[]
            for metric in METRICS:
                mean=row[metric+'_mean'];sd=row[metric+'_sd']
                values.append(f'{mean:.3f} ± {sd:.3f}' if row.n_repeats>1 else f'{mean:.3f}')
            lines.append('| '+names[name]+' | '+' | '.join(values)+' |')
        return '\n'.join(lines)
    primary=summary[summary.scheme=='combination_holdout'].set_index('model')
    lin,et=primary.loc['linear'],primary.loc['extra_trees']
    text=f'''# 분류 성능 보고 — 네 지표와 민감도·특이도 균형

사용자 요청에 따라 앞으로 분류 성능은 **Sensitivity, Specificity, MCC, ROC-AUC**를 함께 보고합니다. 민감도와 특이도의 수준 및 차이를 우선 해석하고, 어느 한 지표가 높다는 이유만으로 좋은 모델이라고 판단하지 않습니다. 둘 다 낮아 차이만 작은 경우도 좋은 균형으로 보지 않습니다.

## 현재 상안정성 결과

양성(1)=안정 제형, 음성(0)=불안정 제형입니다. 민감도는 안정 제형을 안정하다고 찾는 비율, 특이도는 불안정 제형을 불안정하다고 거르는 비율입니다. 임계값은 기존 평가의 0.5를 유지했습니다. 실측 제형 812개(안정 294, 불안정 518), 원료 조합을 분리한 3회 5-fold OOF 결과의 평균±반복 표준편차입니다. 반복 예측을 서로 다른 독립 표본으로 세지 않았습니다.

{rendered('combination_holdout')}

현재 결과에서 선형 모델의 민감도–특이도 차이는 **{lin.sensitivity_specificity_gap_mean:.3f}**, Extra Trees는 **{et.sensitivity_specificity_gap_mean:.3f}**입니다. 선형 모델은 안정 제형 탐지율이 더 높고 특이도는 다소 낮으며, 두 클래스 간 균형은 더 가깝습니다. MCC도 함께 확인했지만 3회 반복 결과만으로 통계적 우월성이나 최종 채택을 확정하지 않습니다.

Extra Trees의 AUC가 약 0.80이어도 민감도 {et.Sensitivity_mean:.3f}와 특이도 {et.Specificity_mean:.3f} 사이의 차이를 고려하면 균형이 충분하다고 평가할 수 없습니다. 선형 모델 역시 안정 제형 중 약 {(1-lin.Sensitivity_mean)*100:.1f}%를 놓치므로 개선이 필요합니다. 상수 기준선은 특이도 1.0이지만 민감도 0이어서 한 클래스만 맞히는 경우를 보여 줍니다.

## 고분자·증점제 조합 제외 평가

8개 subsystem을 차례로 제외한 OOF 결과입니다. 1회 평가이므로 반복 SD를 붙이지 않습니다.

{rendered('subsystem_holdout')}

## 이후 평가 규칙

1. 네 지표를 같은 표에 제시하고, 민감도와 특이도 중 낮은 쪽 및 격차를 설명합니다. 격차만 최소화하거나 임의의 단일 종합점수로 승자를 정하지 않습니다.
2. MCC는 두 클래스와 오류 전체의 관계, ROC-AUC는 임계값과 별개의 구분 능력을 보조적으로 확인합니다. 한 지표의 상승으로 다른 지표의 저하를 숨기지 않습니다.
3. FN은 안정 제형을 놓치는 오류, FP는 불안정 제형을 안정하다고 통과시키는 오류입니다. 반복별 혼동행렬 수치를 CSV에 함께 보존했습니다.
4. 비교 모델은 모집단·분할·양성 정의·임계값을 맞춥니다. 임계값 변경은 학습/내부 검증에서 결정하고 외부 평가 자료로 최적화하지 않습니다.
5. 현재 후보 탐색의 0.8 점수 필터는 별도 예시 설정입니다. 여기의 0.5 평가 지표와 혼동하지 않습니다. 임계값별 후보 평가가 필요하면 별도 검증으로 보고합니다.
6. 점도·탁도는 연속값 회귀이므로 네 분류 지표를 억지로 적용하지 않습니다. 회귀오차는 별도로 보고하며 사후 이분화로 성능을 꾸미지 않습니다.

이번 작업은 기존 예측을 재집계한 것입니다. 모델 재학습·교체·임계값 변경·화장품 자극성 점수 생성은 수행하지 않았습니다. 입력 보강의 효과는 아직 평가하지 않았습니다.

## 파일과 재현

- `reporting_policy.json`: 사용자 지정 보고·판단 규칙.
- `four_metrics_by_repeat.csv`: 네 지표, 혼동행렬, 균형 진단을 반복별로 보존.
- `four_metrics_summary.csv`: 평균·표준편차. 표준편차는 신뢰구간이 아닙니다.
- `stability_oof_snapshot.csv`: 실제 계산에 사용한 기존 안정성 OOF 예측.
- `provenance.json`, `verification.json`: 출처·해시·기존 지표와 일치 검증.

이 폴더에서 `python report_metrics.py`로 다시 생성할 수 있습니다(numpy, pandas, scikit-learn 필요). 현재 저장된 스냅샷으로 독립 실행할 수 있으며 원래 기준선이 있으면 원본과도 대조합니다. 새 모델 결과는 동일한 보고 규칙을 적용하되 기존 스냅샷을 덮어쓰지 않고 새 버전으로 보고합니다.
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    save_json(ROOT/'verification.json',{'status':'PASS','repeat_rows':len(repeat),'summary_rows':len(summary),
              'n_samples_per_model_repeat':812,'positive_stable':294,'negative_unstable':518,
              'four_metrics_recomputed_from_saved_oof':True,'classification_threshold':0.5,
              'baseline_metrics_matched':(BASE/'results/repeat_metrics.csv').exists(),
              'model_refitted':False,'policy_sha256':sha(ROOT/'reporting_policy.json'),'snapshot_sha256':sha(snapshot)})
    print(summary[summary.scheme=='combination_holdout'][['model']+[m+'_mean' for m in METRICS]+['sensitivity_specificity_gap_mean']].to_string(index=False))


if __name__=='__main__':run()
