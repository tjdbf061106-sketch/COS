# 샴푸 실측 44개 검토와 입력 보강

요청한 2번(실측 제형 검토), 3번(입력 보강)을 완료한 별도 분석 폴더입니다. 기존 기준선은 그대로 유지했습니다. **모델을 재학습하지 않았으며 성능 개선을 주장하지 않습니다.**

## 열어볼 파일

- [보고서](report.html): 전체 결과와 그림이 포함된 단독 HTML.
- [엑셀 비교표](샴푸_44개처방비교_입력보강.xlsx): 44개 비교, 원료 투입량, 176개 원료 행, 원측정 곡선, 원료 규격, 입력 정의, 근거 출처.
- [전체 곡선 PDF](figures/observed44_flow_curves.pdf).
- [입력 세트 정의](data/feature_sets.json), [검증 결과](verification.json).

## 핵심 결과

- 실측 제형 44개, 원료 행 176개, 고분자·증점제 7조합 검토. 전단박화 41개, 뉴턴 유체 3개(원문 분류 유지).
- 탁도 상대오차 30% 이상: S17, S153, S154, S335, S639, S736. 이는 검토 플래그이며 자동 제외 사유가 아닙니다.
- S335의 탁도 86 + 원문오차 39 = 125 NTU로, 예시 목표 100을 넘습니다. S337·S349는 점도 목표 100 mPa·s 근처입니다.
- 조합별 검토 출발 제형 7개: S496, S337, S765, S272, S166, S542, S693. 선정 기준은 원문 탁도 상대오차, 100 s⁻¹ 근처 CV, ID 순입니다. 최적 처방/안전성 순위가 아닙니다.
- 전체 812개에 18/61/114개 입력을 생성했습니다. 44개만을 모델 학습에 사용하지 않습니다.
- Dehyton PK 45 활성분: 연구 원문 44–46%와 BASF 공개 자료 37–41%가 다릅니다. 140/812개, 14/44개 제형에 영향이 있어 별도 시나리오로 분리했습니다. 배치·원산지·등급 동등성은 미확인입니다.

## 실행

```powershell
python -m pip install -r requirements.txt
python src/build_review.py
python -m unittest discover -s tests -v
python src/verify_review.py
python src/make_deliverables.py
```

같은 저장소의 `shampoo_physics_20261102/` 기준선과 나란히 둡니다. `inputs/`에는 필요한 원본 사본을 포함했고, 검증은 기존 기준선 파일이 보존됐는지도 확인합니다. 공급사 PDF 전체는 재배포하지 않으며, 근거 숫자·페이지·URL·확인한 문서 해시를 `sources/`에 기록했습니다.

## 데이터 파일

| 파일 | 내용 |
|---|---|
| `review/formulations44_comparison.csv` | 제형별 처방, 10/100/500 s⁻¹ 점도, 측정오차, 검토 플래그 |
| `review/formulations44_stock_amounts.csv` | 원료 18개 투입량과 조정 전 물 잔량 추정 |
| `review/formulations44_ingredients_long.csv` | 성분별 상업 원료 투입량·공개 활성분 범위·등급 |
| `review/formulations44_flow_curves.csv` | 원래의 전단속도·평균 점도·표준편차 |
| `review/same_combination_pairs.csv` | 같은 원료 조합 안의 처방 차이 25쌍; 인과/반복시험 비교 아님 |
| `data/ingredient_registry_reviewed.csv` | 18개 원료의 원문 명칭, 검토 명칭, 범위, 등급, 공급사 차이 |
| `data/features_source_reported_812.csv` | ID + 114개 입력. 원문 범위 유지 |
| `data/features_pk45_supplier_sensitivity_812.csv` | 같은 구조의 PK45 공급사 범위 민감도 시나리오 |
| `data/feature_dictionary.csv` | 모든 입력의 단위·공식·한계 |
| `data/feature_sets.json` | 18/61/114개 열 목록. ID 제외 |
| `data/targets_and_groups_812.csv` | 라벨·결측·그룹을 별도 파일로 보존 |
| `data/outer_folds_preserved.csv` | 기존 3회 5-fold와 8개 subsystem 제외 분할, 바이트 단위 보존 |
| `data/supplier_context_not_core_features.csv` | Luviquat 고형분 등 과거 공급사 참고값. 주 입력에 포함하지 않음 |

## 해석·입력 규칙

- `source-claimed active`는 공개 CSV 열에서 환산한 값입니다. 실험 당시 배치의 활성분 실측이 아닙니다. 원료 투입량 10%에 활성분 44–46%면 제형 기여분은 4.4–4.6%입니다.
- Dehyton PK45 시나리오는 활성분 관련 열만 바꿉니다. 원료 투입량·실측 결과·원문의 물 함량은 그대로이며, 새로운 완전 조성으로 복원한 것이 아닙니다.
- 물·염·산·보존제 열은 공개된 부분 기여분입니다. 누락 성분을 0이라고 단정하지 않습니다. 고분자/증점제 내부 물, 적정액, 기타 미공개 성분은 미확인으로 남깁니다.
- 5종 계면활성제 분류는 저자의 타입 구분입니다. pH 5.8에서 측정한 전하량이 아닙니다. 종류별 투입량 곱은 상호작용을 학습하기 위한 사전 지정 대리변수이고 물리적 상호작용 실측이 아닙니다.
- Polyquaternium-7 두 등급은 분자량·공중합 조성이 다르므로 합치지 않습니다. 고분자 미확인 분자량을 임의 수치로 채우지 않습니다.
- Luviquat 38–42%는 2012년 자료의 고형분 참고값입니다. 순수 고분자 활성분과 같다고 가정하지 않고 주 입력에서 제외했습니다.
- Dehyquart A-CA의 검토 INCI는 공급사 표기 Cetrimonium Chloride입니다. 원문의 Potassium 포함 명칭은 별도 열로 보존했습니다. Dehyton AB30 대표 SMILES 전하 이슈는 아직 확정 수정하지 않았습니다.
- 투입량을 공개 성분 합으로 재정규화하지 않습니다. ID, 라벨, 물성 측정값, 장비, 표본 순서, 목표 충족 여부는 예측 입력에 없습니다.
- 희석·단위 환산·목표값 변경·grade 구분·0분모·보간 범위와 원본 보존을 검증했습니다. 모델 학습에 맞춘 정규화·결측대체·변수 선택은 이 단계에서 하지 않았습니다.
- 모든 통계·플래그는 44개에 대한 기술적 검토입니다. 이 제형들은 실측 물성으로 선택됐으므로 외부검증셋, 무작위 표본 또는 전체 학습셋이 아닙니다.

## 다음 단계

보존한 분할에서 18/61/114개 입력과 활성분 두 시나리오를 비교합니다. 모든 선택과 스케일링은 학습 폴드 안에서 수행합니다. 목표에 맞는 제형만 골라 학습하거나, 전체 자료의 변수 빈도로 먼저 필터링하지 않습니다. 현재 결과는 입력 준비 완료이지 정확도 향상 확인이 아닙니다.

GitHub `f049106`의 연구 방향을 반영했습니다. 화장품 자극성 전이학습과 피부/안자극 점수 생성은 수행하지 않았습니다. 기존 skin·eye, 기준선, 팀 문서는 수정하지 않았고 이번 자료를 GitHub에 게시하지 않았습니다.

## 출처

- [Chitre et al. (2024)](https://www.nature.com/articles/s41597-024-03573-w) 및 [샴푸 데이터](https://springernature.figshare.com/articles/software/Liquid_Formulations_Dataset/25451878), [원료 정보](https://springernature.figshare.com/articles/dataset/Surfactants_Information/25451926), [원료 구조](https://springernature.figshare.com/articles/figure/Structures_of_Formulations_Ingredients/25451929). Figshare 원본은 CC0.
- [BASF Standard Surfactants & Alcohols, 6쪽](https://promo.basf.com/campaign/TechCell/arquivos/Standard%20Surfactants%20%26%20Alcohols.pdf). 문서 발행시점·실험 당시 배치 동등성은 미확인.
- [BASF Luviquat Polymer Grades, 2012-05, 4쪽](https://promo.basf.com/campaign/Projetos/CaringForYou/Documentos/Geral/Luviquat%C2%AE%20Ultracare%20AT%201.pdf).
- [BASF Dehyquart A-CA 제품 정보](https://care360.basf.com/global/en/Product-Finder/30528457).
