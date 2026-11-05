# skin_rf — skin 자극성 production 모델 (RandomForest)

2026-11-05: RF vs 신경망(NN) 정면 비교(`skin irritation/rf_feature_compare/`) 결과 RF 채택.
skin/eye는 각자 최적 모델을 독립적으로 쓰기로 한 팀 결정에 따라, `modeling/src`의 멀티태스크
NN harness와 완전히 분리된 skin 전용 production 파이프라인이다.

## 비교 근거

`skin irritation/rf_feature_compare/`의 세 스크립트(`skin_compare.py`, `rf_vs_nn_head2head.py`,
`rf_vs_nn_balanced.py`) 참조. 핵심: 각 모델을 자기 균형점 threshold에서, 반복 30회로 비교한 결과
RF가 Sensitivity·Specificity·MCC·ROC-AUC 전부 NN과 같거나 높았고, 최적 threshold도 NN보다
10배 이상 일관됨(nested 방식으로도 재확인).

## `modeling/src`(멀티태스크 NN)와의 관계

`modeling/src`는 삭제하지 않는다. NTP ICE 원 도메인 내부에서 Early Stopping·과적합·data
leakage 영향을 보여주는 내부 검증·발표자료용으로 그대로 남겨둔다. skin의 실제 production
(배포용) 모델은 이 폴더(`skin_rf`)다.

## 실행

```
py train.py                         # 학습 + checkpoint 저장
py evaluate.py --checkpoint skin_rf_checkpoint.joblib   # 테스트셋 평가
```

## 평가 원칙

- group-aware 분할(`comp_group_id`)로 동일 조성 leakage 방지
- feature 선택(nz20)과 threshold는 train/val 내부에서만 결정, test는 전혀 보지 않음(nested)
- 성능은 Sensitivity/Specificity/MCC/ROC-AUC를 함께 보고, 단일 지표로 판단하지 않음
- `train.py` 실행 시 나오는 테스트셋 수치는 "배포 체크포인트 하나를 만든 것"이지 공식 성능
  보고가 아님 — 공식 수치는 `rf_feature_compare/`의 반복 group 5-fold 결과를 기준으로 한다.

## checkpoint 구성 (`skin_rf_checkpoint.joblib`)

| 키 | 내용 |
|---|---|
| `model` | 학습된 RandomForestClassifier (trainval 전체로 재학습된 배포용) |
| `nz_filter_keep` | nz20 컬럼 마스크(trainval에서 fit) |
| `threshold` | 검증셋에서 고른 균형점 threshold (test는 안 봄) |
| `feature_cols` | 원본 feature 컬럼 이름(디버깅용) |
| `test_idx` | 평가용 held-out 테스트 인덱스 |
