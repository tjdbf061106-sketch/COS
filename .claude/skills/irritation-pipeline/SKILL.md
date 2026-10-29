---
name: irritation-pipeline
description: COS팀 학술제 프로젝트(NTP ICE 혼합물 기반 화장품 피부·안 자극성 예측) 작업 시 항상 적용. 데이터 스키마, 라벨링 규칙, 모델 아키텍처 결정사항을 팀원 누구나 일관되게 쓰도록 하는 프로젝트 컨텍스트.
---

# 학술제 — NTP ICE 자극성 예측 파이프라인

팀 COS, 2026학년도 미래인재대학 학술제. 작업 시작 전 항상 `세팅\진행상황.md`를 먼저 읽어 최신 상태를 확인할 것 — 이 스킬은 "지금 정해진 규칙"만 요약하고, 시간순 이력/미결 논의는 진행상황.md가 단일 출처(source of truth)다.

## 연구 목적 (한 줄 요약)
ICE mixture(농약 제형) 성분조성 → skin/eye EPA Classification 예측 모델을 학습해, 화장품 제형에 전이학습으로 적용하고 적용범위(AD)를 정량화.

## 데이터 (②단계 산출물, 고정된 입력)
- **`skin irritation\skin_vectorized.csv`** — 544행×2,289컬럼. 메타 16개 + feature 2,273개(공통 8개 MolWt/LogP/TPSA/HBD/HBA/RotatableBonds/AromaticProportion/ESOL_LogS + 전체 RDKit descriptor 217개(`desc_*`) + Morgan FP 2048비트(`fp_w_*`)). 염 제거(salt stripping) 적용됨(scalar descriptor는 염 제거 후 최대 프래그먼트 기준, FP는 염 포함 원본 분자 기준 — eye와 동일 원칙). `binary_label` 존재.
  - ⚠ **이전 CASRN one-hot 버전은 완전히 폐기됨**(2026-10-06). 과거 세션 기록의 "skin_vectorized_v2.csv"/"DEPRECATED" 파일명은 전부 지금 이 파일을 가리킴. 재구성 과정은 `skin_feature_rebuild.ipynb`(8개 descriptor 버전) → `skin_feature_expand_full.ipynb`(217+2048 확장) 순서로 누적 반영됨.
  - harness(`modeling/src/datasets.py`)의 `skin_dataset()`은 feature 전체에 `dropna` 적용 — 구조 완전 미해결 1건(`MIX306`) + 무기물 성분의 일부 desc_ 컬럼 NaN(Gasteiger 전하 미정의) 영향 2건, 합계 3건 제외되어 **541건으로 학습**(binary_label 511/30, 2026-10-29 MIX839/840 worst-case 통일 반영).
- **`eye irritation\eye_vectorized.csv`** — 841행. 메타 다수 + feature(공통 개념은 skin과 동일하나 컬럼명은 RDKit 표준 이름에 내장: `desc_*` 217개 + `fp_w_*` 2048개, ESOL은 `aux_ESOL_logS`로 별도 보관). **반드시 `substance_type == 'Mixture'`만 필터링해서 쓸 것**(순물질 Chemical 235건은 모집단이 달라 제외 — 진행상황.md 참조). `binary_label` 존재(GHS만 있고 EPA라벨 없는 47건은 NaN, 자동 제외).
  - harness `eye_dataset()` 적용 시 Mixture+라벨있음+feature 전부 유효 기준 **604건**(binary 491/113)으로 학습.
- 이분류 라벨 변환 규칙(skin/eye 공통): Category 1~2 → 1(독성/자극), Category 3~4 → 0(비독성/무자극).
- train/test 분할은 `comp_group_id`(skin) / `group_id`(eye) 기준 **group-aware split 필수**(동일조성 쌍이 train/test에 갈라지면 data leakage). 성능 보고/비교는 단일분할 대신 `splits.py`의 `grouped_kfold_splits()`(StratifiedGroupKFold) 사용 권장 — 단일분할은 변동폭이 매우 큼(검증 완료, 진행상황.md 참조).

## 모델 아키텍처 (2026-09-18 교수님 피드백으로 확정, 기존 RF/XGBoost+4단계 계획은 폐기)
- **Multi-task PyTorch, 이분류 단일**(4단계 다분류 안 함)
- skin/eye는 입력 차원이 다르므로(skin 2,273 vs eye 2,265, 2026-10-06 팀 논의로 양쪽 다 고차원 feature로 통일) 각각 **task-specific encoder**로 받고 shared_trunk에서 공통표현 학습 후 task별 head로 분기(hard parameter sharing)
- ⚠ feature 차원(2천대) 대비 샘플 수(skin 541·eye 604)가 매우 작아 과적합 위험이 큼 — 정규화(dropout/L2)·차원축소(PCA) 적용을 강하게 권장(skin_feature_expand_full.ipynb 한계점 참조)
- "α·skin + β·eye 가중합산"은 **샘플 단위 벡터합이 아니라 task loss 가중합**(`total_loss = α·skin_loss + β·eye_loss`)으로 구현 — skin/eye가 서로 다른 Formulation_ID라 벡터를 더할 짝이 없기 때문. 근거는 `세팅\멀티태스크_모델링_정리.md` 5절.
- Loss: CrossEntropyLoss. 클래스 불균형 대응은 **class-weighted CE 우선(지도교수 피드백으로 채택 확정, 2026-10-29)**, 1:1 언더샘플링/SMOTE는 비교 검토했으나 최종 미채택(skin 양성 30건뿐이라 데이터 손실 크거나 Morgan FP 이진값과 상성 문제). `modeling/src/train.py`에 구현됨 — task별 독립적으로 inverse-frequency 가중치 계산.
- 코드 스켈레톤: `modeling\src\` (datasets.py/model.py/splits.py/train.py/evaluate.py), 상세 사용법은 `modeling\README.md`.

## 작업 규칙
- Python 실행은 `py` 런처 사용(`python`/`python3` alias 불안정한 환경).
- 엑셀 대량 반영은 5개 이하씩 개별 저장(openpyxl 타임아웃으로 파일손상 경험있음).
- 새 산출물은 `C:\Users\UserK\학술제` 폴더 내 해당 서브폴더(skin irritation / eye irritation / modeling / 세팅)에 저장(2026-10-06부터 기본 경로 — Desktop 하위 경로 아님).
- 수치·스키마 확인이 필요하면 추측하지 말고 반드시 파일 직접 열어서 확인(pandas로 shape/컬럼/분포 직접 조회) — 문서 기억만으로 단정 금지.
- 미결 사항이나 스코프 불일치를 발견하면 조용히 넘어가지 말고 `세팅\진행상황.md`에 기록하고 사용자에게 알릴 것.
