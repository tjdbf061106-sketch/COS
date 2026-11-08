# Eye production 검증 및 통합 노트북 — 2026-11-08

먼저 **[eye_all_experiments_v2.ipynb](eye_all_experiments_v2.ipynb)** 를 열어 5부와 6부를 읽으세요.
최신 production 결과와 갱신된 종합 결론이 저장된 출력으로 포함되어 있습니다.

## 검증 범위

- 원본 673행 = **기존 606행 평가 + 증강 67행 train-only**.
- baseline(fp2048 + desc217) vs 농도 피처 4개 추가(2269개). 원본 벡터 재스케일 없음.
- 6모델 × A/B × seed 42/202/777 × outer 5-fold × 2 arm = **360 outer 결과**.
- inner 3-fold, 모델마다 20후보 전체 격자. 농도 중앙값은 해당 학습 fold에서만 계산.
- outer test/inner validation 그룹과 연결된 증강행은 해당 학습에서 제외.
- threshold는 기존행 inner OOF의 민감도·특이도 균형점에서 선택.
- seed별 606행 pooled OOF 지표를 계산한 뒤 3seed 평균 ± 표본 표준편차(ddof=1)를 보고.
- 동일 데이터 내부 검증이며 외부 도메인 성능·통계적 유의성·동등성 검정 결과가 아님.

## 주요 결과

| 라벨·모델 | baseline MCC | 농도 MCC | 짝지은 ΔMCC 평균 ± 표준편차 |
|---|---:|---:|---:|
| A / LR | 0.336051 | 0.337392 | +0.001342 ± 0.002324 |
| B / XGBoost | 0.182657 | 0.223927 | +0.041270 ± 0.075155 |
| B / RandomForest | 0.199990 | 0.215147 | +0.015156 ± 0.033631 |
| B / LightGBM | 0.184002 | 0.211707 | +0.027706 ± 0.029325 |

A의 LR 변화는 작고, B의 개선은 모델·seed에 따라 달라집니다. 평균 순위만으로 최종 모델을
확정하지 않습니다. 그룹화 실험03의 2판 ΔMCC −0.014205는 별도의 673행 CV 결과로,
지정 기준상 중간 구간입니다. 과거 노트북의 '채택' 문구는 당시 탐색 판단으로 보존했습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `eye_all_experiments_v2.ipynb` | 최신 통합본. 1~4부는 역사적 출력, 5~6부는 최신 production 검증/결론 |
| `eye_all_experiments.ipynb` | 수정하지 않은 이전 통합본 |
| `data/eye_vectorized_production_673.csv` | 이 검증에 사용한 Eye 입력 snapshot |
| `claude/results/concentration_production/f91f3c88787bec6b/` | config, 지표, paired 비교, export, 360개 outer 예측 |
| `claude/results/03_grouping_fix*` | 그룹화 1판/2판 결과, 비교와 판정 |
| `claude/results/continuation_20261108T111518/` | 전체 CV 완료·누수·체크섬 검증 기록과 ΔMCC CSV |
| `claude/results/pipeline_v2_20261108T144904/` | 기능 테스트 28개 및 실제 LR 두 arm 예측 일치 검증 기록 |
| `claude/results/notebook_production_v2_20261108T154155/` | 노트북 추가 셀 실행 검증 및 ΔMCC 그래프 |
| `claude/*.py`, `claude/PRODUCTION_V2.txt` | 검증 당시 코드 snapshot 및 학습·추론 사용 안내 |
| `upload_manifest.json` | 이 폴더 파일별 SHA256, 출처와 업로드 범위 |

## 학습 없이 결과 재검증

검증 환경은 Python 3.14.6입니다. 필요한 보고서용 패키지는 `requirements-report.txt`에 있습니다.
이 폴더에서 실행합니다.

```bash
python -m pip install -r requirements-report.txt
python verify_results.py
```

파일 SHA256와 원본 입력 673행의 역할을 확인하고, 새 노트북의 **5부 코드 셀 5개만**
별도 커널에서 실행합니다. 360개 outer에서 72 OOF 세트 × 606행의 성능지표를 재집계해
저장 지표와 비교합니다. 학습 실행기를 호출하거나 노트북 파일을 덮어쓰지 않습니다.
Jupyter에서도 1~4부를 실행하지 않고 5부 첫 코드 셀부터 순서대로 실행할 수 있습니다.

## 보존 및 재현성

- 저장소의 기존 `eye irritation/eye_vectorized.csv`(841행)와 이 입력(673행)은 서로 다른
  모집단입니다. 기존 파일을 교체하지 않았습니다. Skin 자료도 이번 추가 범위에 없습니다.
- `eye_vectorized_concentration.csv`는 **전체 학습용 파생 export**이며 CV 입력으로 사용하지
  않았습니다. 노트북은 이 CSV의 feature를 읽지 않고 파일 checksum만 확인합니다.
- 학습 코드·결과·노트북은 검증 당시 bytes를 유지했습니다. `.gitattributes`에서 줄바꿈 변환을
  막아 SHA256을 보존합니다. 원래 run identity는
  `f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b`입니다.
- **학습 코드의 개인 절대경로는 원본에 남아 있습니다.** 이 폴더는 완료 결과와 원본 코드의
  인계 패키지입니다. 보고서 5부와 `verify_results.py`는 폴더 기준 경로로 검증했습니다.
  새 컴퓨터에서 학습기를 직접 실행하려면 원본을 보존한 별도 버전에서 경로를 설정해야 합니다.
  기존 runner의 sibling Skin 파일 checksum 참조도 그대로 보존했으며, Skin CSV는 복사하지
  않았습니다. 이 참조를 처리하지 않은 상태에서 학습기 실행 가능하다고 주장하지 않습니다.
- 1~4부의 경로·원료 자료·GJC/EPA 외부 의존까지 이식한 것은 아닙니다. 과거 저장 출력은
  볼 수 있으나 전체 노트북을 무조건 Run All 하는 대신 5부를 독립 실행하세요.
- 7,200개 후보별 중간 checkpoint는 로컬 복구용으로 보존하고 이 폴더에서는 제외했습니다.
  전체 후보 완료 사실은 기존 완료 검증 기록에, 최종 outer 예측은 360개 JSON에 있습니다.
- 28개 코드 테스트 통과와 360 outer 성능 검증은 구별합니다. **전체 673행 최종 후보 모델
  학습, 배포 또는 모델 교체는 수행하지 않았습니다.** 검증용 fold 모델도 배포 모델로 포함하지 않습니다.

## 저장소 작업 기록

2026-11-08 사용자 요청에 따라 `최종/` 아래 새 폴더로 Eye 자료만 추가했습니다.
기존 저장소 파일과 원본 로컬 자료는 변경하지 않았습니다. 기존 결과 재검증만 수행했고
학습 재실행은 없었습니다. 최신 프로젝트 진행 이력을 확인했으며, 예전 skill의 544행 Skin/
841행 Eye 설명은 이 독립 production 673행 snapshot의 평가 규약을 대체하지 않습니다.
