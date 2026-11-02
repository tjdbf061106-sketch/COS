# 샴푸 물성 연구 공유 패키지 — 2026-11-02

812개 실측 제형의 분석 데이터, 기준선 모델 3개, 평가 예측, 44개 처방 검토표, 보강 입력과 재현 코드를 함께 공유합니다. **최신 성능 해석과 실행 경로는 이 문서를 기준으로 합니다.** 상위 `shampoo_demo`의 기존 파일과 설명은 수정하지 않았습니다.

| 필요한 자료 | 위치 |
|---|---|
| 812개 제형·18개 원료 투입량·물성·그룹 | [formulations.csv](shampoo_physics_20261102/data/formulations.csv) |
| 입력 열 순서와 추론 규칙 | [prediction_contract.json](prediction_contract.json) |
| 상안정성·점도·탁도 모델과 모델 카드 | [models/](shampoo_physics_20261102/models/) |
| 학습 코드·고정 분석계획·평가 분할 | [src/](shampoo_physics_20261102/src/), [protocol.json](shampoo_physics_20261102/protocol.json), [outer_folds.csv](shampoo_physics_20261102/results/outer_folds.csv) |
| 원래 OOF 예측과 회귀 성능 | [oof_predictions.csv](shampoo_physics_20261102/results/oof_predictions.csv), [metrics_summary.csv](shampoo_physics_20261102/results/metrics_summary.csv) |
| 네 분류 지표·양성 정의별 혼동행렬 | [반복별](metrics_by_positive_class.csv), [평균·SD](metrics_summary_by_positive_class.csv), [보고 원칙](classification_reporting/reporting_policy.json) |
| 실측 출발 처방 44개 비교 | [엑셀](shampoo_inputs_review_20261102/샴푸_44개처방비교_입력보강.xlsx), [비교표 CSV](shampoo_inputs_review_20261102/review/formulations44_comparison.csv), [검토 보고서](shampoo_inputs_review_20261102/report.html) |
| 18/61/114개 입력 변수와 정의 | [feature_sets.json](shampoo_inputs_review_20261102/data/feature_sets.json), [feature_dictionary.csv](shampoo_inputs_review_20261102/data/feature_dictionary.csv) |
| 812개 제형 보강 입력·공급사 시나리오 | [원논문 기준](shampoo_inputs_review_20261102/data/features_source_reported_812.csv), [PK45 별도 시나리오](shampoo_inputs_review_20261102/data/features_pk45_supplier_sensitivity_812.csv) |
| 미측정 보간 후보 | [candidate_predictions.csv](shampoo_physics_20261102/results/candidate_predictions.csv) |
| 출처·해시·패키지 검증 | [package_provenance.json](package_provenance.json), [manifest.json](manifest.json), [verification.json](verification.json) |

HTML 보고서는 저장소를 내려받아 브라우저로 열 수 있습니다. GitHub에서는 이 README와 CSV를 먼저 확인하면 됩니다. 각 하위 폴더는 기존 산출물 97개를 바이트 그대로 보존한 스냅샷입니다. 하위 README의 과거 경로·다음 작업·AUC 중심 요약보다 이 문서와 네 지표 표를 우선합니다.

## 현재 완료 범위

- 상안정성: 812개, 안정 294 / 불안정 518. 원료 조합 475개를 묶은 3회 5-fold OOF 평가, 고분자·증점제 조합 8개를 차례로 제외한 추가 평가.
- 물성 회귀: 실측 안정 제형 중 점도 293개, 탁도 294개. S151의 비양수 유변곡선 때문에 점도만 제외했고 안정성·탁도는 유지했습니다. 불안정 제형의 미측정 물성을 0으로 채우지 않았습니다.
- 실측 출발 처방 44개: 원료 등급·활성분 범위·유변곡선·측정 변동·7개 조합별 검토 대표 처방을 정리했습니다. 44개는 목표를 충족한 관측 사례이며 별도 학습 모집단이나 독립 검증셋이 아닙니다.
- 입력 보강: 전체 812개에 대해 stock18 / physics_core 61 / physics_interactions 114를 만들고 기존 분할을 보존했습니다. **보강 입력으로 새 모델을 학습하거나 성능 개선을 확인하지는 않았습니다.**
- 공급사 정보가 원논문과 다른 PK45는 원본 값을 보존하고 별도 시나리오로 구성했습니다. 140개 제형이 영향을 받으며 실제 실험 배치의 활성분이 확인된 것은 아닙니다.

동일·근접 제형의 모든 누수나 외부 독립성을 확정한 결과는 아닙니다. 현재 주평가의 그룹은 **활성 원료 ID 조합 전체의 일치**로 정했으며, 다른 조합 사이의 근접성까지 배제하는 별도 평가가 완료된 것은 아닙니다. 독립 외부검증도 아직 없습니다.

## 분류 성능: 네 지표와 양쪽 예측의 균형

모든 표는 평가 임계값 0.5의 같은 OOF 예측에서 계산했습니다. 아래는 812개, 3회 그룹 분할의 평균입니다. 반복별 수치·혼동행렬 및 반복 SD는 위 CSV에 있습니다. 반복 SD는 신뢰구간이 아니며 반복 예측을 독립 표본으로 더하지 않습니다.

상위 데모 문서에 맞춘 **양성=불안정** 기준입니다. Sensitivity는 불안정 제형 탐지율, Specificity는 안정 제형 유지율입니다.

| 모델 | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---:|---:|---:|---:|
| 상수 기준선 | 1.000 | 0.000 | 0.000 | 0.498 |
| 선형 모델 | 0.832 | 0.608 | 0.450 | 0.811 |
| Extra Trees | 0.870 | 0.546 | 0.445 | 0.802 |

원래 학습·모델 카드·하위 보고서는 **양성=안정** 기준입니다. 이 기준에서는 Sensitivity가 안정 제형 탐지율, Specificity가 불안정 제형 배제율입니다.

| 모델 | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---:|---:|---:|---:|
| 상수 기준선 | 0.000 | 1.000 | 0.000 | 0.498 |
| 선형 모델 | 0.608 | 0.832 | 0.450 | 0.811 |
| Extra Trees | 0.546 | 0.870 | 0.445 | 0.802 |

선형 모델은 두 클래스 탐지율의 차이가 0.224, Extra Trees는 0.324입니다. 선형 모델이 이 평가에서 더 가까운 균형을 보이지만 안정 제형의 약 39%를 놓치며, Extra Trees는 약 45%를 놓칩니다. AUC 약 0.80이나 한쪽 탐지율만 보고 충분한 성능이라고 판단하지 않습니다. 두 값이 모두 낮아서 차이만 작은 경우도 좋은 균형이 아닙니다. MCC·AUC는 함께 확인하는 보조 근거이며 이 결과만으로 통계적 우월성이나 최종 모델 채택을 확정하지 않습니다.

저장 모델 3개는 사전에 고정한 **Extra Trees 기준선**입니다. 선형 모델의 비교 결과도 전부 보존했으며 모델을 자동 교체하지 않았습니다. 회귀는 별도 로그 MAE/RMSE/R²로 평가합니다. Extra Trees의 점도 R²는 0.364, 탁도 R²는 0.239로, 정밀 처방 확정용 성능은 아닙니다.

## 모델을 연결할 때의 입력·출력 규칙

1. 입력은 [열 순서](shampoo_physics_20261102/data/feature_names.json) 그대로 **18개 상용 원료의 전체 제형 대비 투입 질량%**입니다. 알려진 원료의 합을 100으로 재정규화하지 않습니다. 같은 INCI라도 공급사·등급이 다르면 동일 입력으로 간주할 수 없습니다. 61/114개 보강 변수를 기존 모델에 바로 넣으면 안 됩니다.
2. `stability.predict_proba(X)[:, 1]`은 **안정 점수**입니다. 별도 확률 보정은 수행하지 않았습니다. 안정 판정은 `p_stable >= 0.5`, 불안정 판정은 이 판정의 반대입니다. 데모의 불안정 점수는 `1 - p_stable`이며 0.5 동점은 안정으로 유지합니다. 따라서 불안정 판정은 `p_unstable > 0.5`입니다.
3. 양성 정의를 바꾸면 Sensitivity와 Specificity가 서로 바뀌고 MCC·ROC-AUC는 같습니다. AUC 재계산은 반대 순위를 정확히 보존하도록 `-p_stable`을 사용합니다. `1-p_stable` 연산의 부동소수점 반올림으로 매우 가까운 점수가 동점으로 합쳐지는 것을 피하는 처리이며, 음수 값은 확률이 아닌 순위 점수입니다.
4. 점도 모델 출력은 `log10(η100 / mPa·s)`이며 실제 단위로는 `10 ** prediction`입니다. 조건은 25℃, 전단속도 100 s⁻¹입니다. 탁도 모델은 `log10(1+NTU)`이며 `10 ** max(prediction, 0) - 1`로 복원합니다. 두 회귀는 **실측 안정 제형에 조건부**로 학습됐습니다.
5. 상안정성은 원 연구의 pH 5.8±0.2 조정과 최종 36시간 관찰 조건의 단기 결과입니다. 장기 안정성·유통기한·피부/안자극성 결과가 아닙니다. 상용 원료 내부 분자 조성이 완전히 확보된 자료도 아닙니다.
6. 예시 후보 필터의 안정 점수 0.8은 성능표의 0.5 임계값과 별개입니다. 실측 목표 충족 44개와 미측정 보간 후보 990개를 구분합니다. 후자는 점추정 목표 충족 99개, 기술적 오차대까지 충족 0개이며 실험 확인이 필요합니다. 오차대는 보정된 예측구간이나 보증이 아닙니다.

## 재현 검사

COS 저장소 전체를 내려받은 뒤 이 폴더에서 실행합니다. 기존 skin·eye 파일은 원래 성분 감사에 사용한 해시를 대조할 때만 읽고, 물성 모델의 학습 라벨이나 checkpoint로 사용하지 않습니다.

```powershell
cd shampoo_demo/research_package_20261102
python -m pip install -r requirements.txt
python -B verify_package.py --refit
```

Python 3.14와 기록된 의존성 버전에서 확인했습니다. 재학습 대조를 생략하는 빠른 검사는 `python -B verify_package.py`입니다. 원본 검증기가 결과 파일을 쓰기 때문에 래퍼가 임시 복사본에서 실행하고 저장된 산출물은 보존합니다. 실제 COS 경로를 자동 연결하므로 과거 스크립트의 최상위 폴더 경로 가정을 직접 수정할 필요가 없습니다. 다른 위치에서 실행할 경우 `--repo <COS 경로>`를 지정합니다.

검사는 파일 해시, 현재 데모 원자료와의 일치, 분할·타깃·후보·모델 입력 순서, 네 지표와 양성 정의 환산, 기존 테스트 12개를 확인합니다. `--refit`은 세 타깃의 첫 분할 재학습 예측도 대조합니다. 전 분할을 다시 학습한 검사는 아닙니다. 필요하면 `--output <패키지 밖 JSON 경로>`로 새 검증 기록을 저장합니다.

전체 모델 실험을 다시 할 때는 스냅샷을 별도 실행 폴더로 복사하고 하위 학습 코드를 사용합니다. 현재 학습 잠금·분할·OOF는 유지하며, 보강 입력 비교나 임계값 선택은 별도 버전의 분석계획에 따라 수행해야 합니다. 모델 파일과 연결 정보만 공유했으며 앱 추론 화면 구현은 포함하지 않았습니다.

## 출처와 보존

논문과 공개 원자료는 기존 [논문 정보](../docs/papers/chitre_2024.md), [원자료 출처](../docs/sources/provenance.json), 패키지의 [Figshare v1 원본·해시](shampoo_physics_20261102/sources/manifest.json)를 참고합니다. 공개 원자료의 라이선스는 CC0입니다. 공급사 보강 근거는 [공식 자료 URL·페이지·검토 사항](shampoo_inputs_review_20261102/sources/supplier_evidence.json)에 기록했으며 공급사 PDF 자체는 포함하지 않았습니다.

`frozen_inventory.json`은 기존 산출물 97개의 해시, `manifest.json`은 이 공유 패키지 파일들의 해시를 보존합니다. `.gitattributes`는 운영체제별 줄바꿈 변환으로 스냅샷 해시가 달라지는 것을 방지합니다. 원본 보고서의 이전 요약을 재작성하지 않고 최신 판단을 이 진입 문서와 네 지표 CSV에 추가했습니다.
