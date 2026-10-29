# NTP ICE 혼합물 기반 화장품 피부·안 자극성 예측 모델 개발
### — 도메인 전이 및 적용범위(Applicability Domain) 정량화

2026학년도 미래인재대학 학술제 참가 프로젝트 (팀 COS: 김서율 · 장다예 · 전수진 · 한세음)

## 이 연구가 뭘 하는 건가

화장품 피부·안 자극성 평가는 전통적으로 동물실험(Draize test)에 의존해왔지만, EU 등 주요 시장에서 화장품 원료 동물실험이 금지되면서 in vitro·in silico 대체 평가법 수요가 커지고 있습니다.

이 연구는 **NTP ICE(Integrated Chemical Environment)**의 농약 혼합물 피부·안 자극성 데이터(EPA Classification 4단계 등급)로 성분 조성(SMILES 기반 분자구조 feature)으로부터 제형 전체의 자극성을 예측하는 모델을 만들고, 이를 **화장품 제형에 전이학습(transfer learning)으로 적용**하는 것을 목표로 합니다. 핵심 질문은 두 가지입니다.

1. 농약 제형 데이터로 학습한 모델이 화장품처럼 전혀 다른 조성의 제형에도 쓸 수 있을까?
2. 쓸 수 있다면 어디까지(적용범위, AD) 믿을 수 있고, 어디서부터는 확인시험이 필요할까?

## 지금까지 한 일 (요약)

- **skin/eye 데이터 벡터화 완료**: NTP ICE 원본에서 Mixture+EPA Classification 조건으로 필터링 → CASRN 결측/라벨 충돌 등 데이터 정제 → SMILES 기반 RDKit 분자 descriptor(공통 8개 + 전체 descriptor 217개 + Morgan fingerprint 2,048비트)로 벡터화. (skin 544건, eye 841건)
- **Multi-task PyTorch 모델 설계 및 1차 학습**: skin/eye를 하나의 신경망으로 묶어 task-specific encoder + shared trunk + task별 head 구조로 설계. 클래스 불균형(skin 독성:비독성 ≈ 1:17) 대응 방식을 class-weighted loss / SMOTE 등으로 비교 검토 후 class-weighted loss 채택.
- **중간보고서 작성 완료**(`서류/중간보고서_COS.docx`).
- 현재 단계: 모델 성능 개선(feature 정비, 하이퍼파라미터 튜닝) 진행 중. 화장품 데이터 적용(전이학습)은 다음 단계.


## 폴더 구조

```
학술제/
├── skin irritation/   # skin 트랙 — 데이터 전처리·벡터화·feature 실험
├── eye irritation/     # eye 트랙 — 데이터 전처리·벡터화
├── modeling/           # Multi-task PyTorch 모델 harness (skin+eye 통합 학습/평가)
├── 서류/               # 연구계획서, 중간보고서, 학교 제출 서식
├── 선행연구/            # 팀원별 선행연구 리뷰 (4편)
├── 세팅/               # 진행상황.md 등 프로젝트 운영 문서
└── 주차 별 진행상황/     # 주차별 결과요약 PDF
```

### `skin irritation/` — skin 트랙

| 파일 | 설명 |
|---|---|
| `skin_irritation.xlsx` | NTP ICE 원본 데이터 |
| `skin_filtered.csv` | Mixture + EPA Classification 필터링 결과(1,799행) |
| `skin_formulation_level_with_comp_group.csv` | 제형 단위 라벨 + 동일조성 그룹(comp_group_id) 정리본 |
| `casrn_smiles_cache.csv` | CASRN → SMILES 매핑(PubChem 조회) |
| `casrn_descriptors_full.csv` | 성분별 RDKit descriptor 계산 결과(264종 × 2,273차원) |
| `skin_vectorized.csv` | **최종 벡터화 결과**(544행 × 2,289컬럼, 모델 입력용) |
| `skin_feature_rebuild.ipynb` / `skin_feature_expand_full.ipynb` | feature 생성 과정 노트북(염 제거, descriptor 계산, 시각화 포함) |
| `skin_addons.ipynb` | EPI Suite 대조용, 메타변수 검증, 그룹분할 검증 |
| `skin_aggregation_comparison.ipynb` | 성분→제형 집계 방식(가중평균/최댓값/최솟값 등) 비교 실험 |
| `skin_harness_cv_upgrade.ipynb` | 단일분할 vs group k-fold 교차검증 안정성 비교 |
| `skin_classweight_demo.ipynb` / `skin_smote_demo.ipynb` | 클래스 불균형 대응 방식(class-weighted loss vs SMOTE) 비교 데모 |
| `epi_check_template.csv` | EPI Suite 실측값 대조용 템플릿(30종, 입력 대기 중) |
| `XGB.xlsx` / `descriptor.pdf` | 선행연구(Kang et al. 2023) 벤치마크 데이터·descriptor 정의 참고자료 |

### `eye irritation/` — eye 트랙

| 파일 | 설명 |
|---|---|
| `eye_irritation.xlsx` | NTP ICE 원본 데이터 |
| `eye_filtered.csv` | 혼합물/순물질/시험법별 분리 결과 |
| `eye_vectorized.csv` | **최종 벡터화 결과**(841행, skin과 동일한 descriptor+fingerprint 체계) |
| `eye_feature_rebuild.ipynb` | feature 생성 과정 노트북 |
| `eye_벡터화_방법.md` | eye 트랙 벡터화 방법론 설명 |
| `DRD_Barroso2017_extracted.xlsx`, `DRD_usable.csv`, `ICE_Eye_Irritation_CAP.xlsx`, `eye_invitro_vitrigel.csv` | 보강데이터 후보(검토 결과는 진행상황.md 참조) |

### `modeling/` — 통합 모델링 harness

skin과 eye를 하나의 Multi-task 신경망으로 학습·평가하는 PyTorch 코드.

| 파일 | 설명 |
|---|---|
| `src/datasets.py` | skin/eye CSV를 PyTorch Dataset으로 로드 |
| `src/model.py` | Multi-task 모델 구조(task-specific encoder + shared trunk + head, 학습 가능한 α/β 손실가중치) |
| `src/splits.py` | 동일조성 제형이 train/test에 갈라지지 않도록 하는 group-aware 분할 |
| `src/train.py` | 학습 스크립트(클래스 불균형 대응 포함) |
| `src/evaluate.py` | 학습된 모델 평가(F1, ROC-AUC, 혼동행렬) |
| `checkpoint.pt` | 최근 학습된 모델 가중치 |

### `서류/` — 문서

연구계획서, 중간보고서(`중간보고서_COS.docx`), 학교 지정 작성양식, 참가신청서 등 제출용 서류.

### `선행연구/` — 선행연구 리뷰

팀원별로 분담해 검토한 선행연구 4편(총 5개 논문): STopTox(Borba 2022), PFAS 전이학습(Feinstein 2021), AttentiveSkin(Huang 2024) + AI-Nano 리뷰(Zhou 2026), Kang·Kim·Lim(2023) XGBoost 벤치마크.

### `세팅/` — 프로젝트 운영 문서

- **`진행상황.md`**: 이 프로젝트의 단일 진실 출처(source of truth). 모든 결정사항·실험결과·미결항목이 날짜순으로 기록됨.
- `멀티태스크_모델링_정리.md`: 모델 아키텍처 설계 배경
