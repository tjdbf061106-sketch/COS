# 샴푸 물성 설계 연구 — 원자료와 첫 기준선

이 폴더는 COS의 B안(샴푸 물성 설계 연구)을 위한 공개 원자료를 원본 파일명 그대로 보관한다. 자극성 예측·화장품 전이학습과는 별도 트랙이며, skin/eye 라벨·모델 가중치를 물성 모델 학습에 사용하지 않는다.

## 포함한 공개 자료

| 경로 | 내용 | 출처 |
|---|---|---|
| `source_data/liquid_formulations/LiquidFormulationsDataset_2023.json` | 812개 샴푸 제형의 조성·단기 상안정성·탁도·유변 측정 원자료 | Figshare 25451878 v1 |
| `source_data/surfactant_information/BASF Surfactants Information.csv` | 계면활성제 상용 원료의 INCI, 대표 SMILES, 활성분·수분·기타 함량 참고 정보 | Figshare 25451926 v1 |
| `source_data/ingredient_structures/BASF Formulation Ingredient Structures.pptx` | 18개 제형 원료의 구조 참고 자료 | Figshare 25451929 v1 |

원 연구: Chitre A. et al., *Accelerating Formulation Design via Machine Learning: Generating a High-throughput Shampoo Formulations Dataset*, Scientific Data 11, 728 (2024). https://doi.org/10.1038/s41597-024-03573-w

## 모델링 범위

1. 조성만으로 단기 상안정성(안정/불안정)을 분류한다.
2. 실제로 안정하고 측정값이 있는 제형에 한해 탁도와 사전 고정한 전단률의 점도를 각각 회귀한다.
3. 후보 제안은 학습 원료·농도 범위 안에서만 수행하며, 후속 실험 대상이다.

장기 보관 안정성, 모든 상용 샴푸에 대한 일반화, 피부·안자극성, 안전성 또는 규제 적합성을 이 자료로 판정하지 않는다.

## 다음 작업

- 원료 ID·투입량·단위·결측·반복 측정·제형 조합을 감사하고 분석용 테이블을 별도로 생성한다.
- group-aware 평가 계획을 고정한 뒤 상안정성 분류 기준선을 재현한다.
- 안정 제형만 대상으로 탁도·점도 회귀와 오류 분석을 수행한다.
- 생성된 분석용 데이터·코드·모델·결과는 원자료를 변경하지 않고 이 폴더의 하위 디렉터리에 추가한다.

파일별 다운로드 URL, 버전, SHA-256은 `provenance.json`에 기록했다.
