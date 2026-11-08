# Eye production·STopTox 외부 보조학습 통합 결과

[eye_all_experiments_v2.ipynb](eye_all_experiments_v2.ipynb)의 **5~7부**를 먼저 확인하세요.
5부는 기존 production, 6부는 실제 외부 보조학습과 화장품 자료 검토, 7부는 갱신된 종합 결론입니다.

**외부학습은 완료됐습니다. 현재 비교에서 일관된 개선은 확인되지 않았으며,
독립 화장품 EPA 외부 성능은 미검증입니다(적격 0행).**

## 기존 production — 5부

- 기존 606행 평가, 증강 67행 train-only. validation/test 그룹에 연결된 증강은 학습에서 제외.
- baseline 2,265개 vs 농도 피처 추가 2,269개; LR/RF/ExtraTrees/SVM/XGBoost/LightGBM.
- A: EPA I/II, B: EPA I/II/III; seed 42/202/777, outer grouped 5-fold·inner 3-fold, 모델별 20후보.
- threshold는 기존행 inner OOF의 민감도·특이도 차 최소화 규약. 전처리·중앙값은 학습 fold에서만 계산.
- 완료된 **360 outer**, 원래 identity `f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b` 보존.
- A–LR: MCC 0.336051 → 0.337392, Δ +0.001342 ± 0.002324.
  B–XGBoost: 0.182657 → 0.223927, Δ +0.041270 ± 0.075155. ±는 3seed 표본 SD입니다.

## 실제 외부 보조학습 — 6부

STopTox Table S4의 관측 3,545행에서 중복·구조 표현 문제 등 284행을 제외한
**3,261행(양성 982/음성 2,279)**으로 새 RF 성분 보조모델을 학습했습니다.
GHS 1–2B 관측 라벨을 EPA 정답으로 합치지 않았습니다. 고정 보조모델의 성분 점수를
가중평균·최대·최소·가중 표준편차 4개 특성으로 전달했습니다. 기존 ChemBERTa 재사용 실험이 아닙니다.

대조군 2,269 vs 보조 특성 추가 2,273개, **같은 기존 606행·356그룹·분할·threshold 선정 규칙**으로
LR/SVM/LightGBM 3종을 비교했습니다. 평가 제외행은 0입니다.
신규 **90 outer·1,800후보 작업·downstream 적합 5,490회**를 완료하고 기존 대조군 90 outer를 감사 후 재사용했습니다.
3seed는 EPA 분할을 변경하며 보조모델·downstream estimator 난수는 42로 고정했습니다.

| 라벨·모델 | 대조군 MCC | 외부학습 MCC | 짝지은 ΔMCC 평균 ± SD | 개선 seed |
|---|---:|---:|---:|---:|
| A / LR | 0.337392 | 0.342578 | +0.005186 ± 0.008982 | 1/3 |
| A / SVM | 0.240446 | 0.216945 | −0.023501 ± 0.042925 | 1/3 |
| A / LightGBM | 0.251454 | 0.245476 | −0.005978 ± 0.030556 | 2/3 |
| B / LR | 0.160393 | 0.156165 | −0.004228 ± 0.007368 | 1/3 |
| B / SVM | 0.131451 | 0.137997 | +0.006546 ± 0.012181 | 2/3 |
| B / LightGBM | 0.211707 | 0.228089 | +0.016381 ± 0.020367 | 2/3 |

평균 개선은 3/6조합, 세 seed 모두 개선은 0/6조합입니다. SD는 분할 민감도이며 신뢰구간이 아닙니다.
원시험 ID 연결과 UVCB 목록 범위에는 한계가 남습니다. STopTox 출처 내부 CV를 독립 외부 성능으로 보고하지 않습니다.

## 화장품 외부검증 검토

Baby bath 2025의 39제품, 58원료 시험 조건, 5새 조합 조건과 IIVS의 14번호 제형을 검토했습니다.
전체 배합비·농도 기준·상업 원료 구조·EPA 호환 원시험 정답이 부족해 정량 검증 적격은 **0행**입니다.
원문 사례 분석은 남겼지만 Draize 등급/BCOP IVIS를 EPA로 임의 변환하지 않았습니다.
미공개 잔량을 물로 채우거나 조건·반복을 독립 제품으로 늘리지 않았습니다.
후보 자료는 학습·튜닝·threshold에 사용하지 않았고 저자에게 연락하지 않았습니다.
ChemExpo 함량규칙 역시 Eye 시험 정답으로 평가하지 않았습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `eye_all_experiments_v2.ipynb` | 갱신 통합본, 표·그래프 저장 출력 포함 |
| `eye_all_experiments.ipynb` | 과거 통합본 보존 |
| `claude/results/concentration_production/f91f3c88787bec6b/` | 기존 production 360 outer와 지표·분할·config |
| `외부학습/Eye_STopTox_20261108/` | 외부학습 완료 결과, 새 RF 보조모델, OOF, 사례 감사표, 원 관측 CSV, 핵심 학습 코드, 체크포인트 ZIP |
| `data/eye_vectorized_production_673.csv` | 보호된 Eye 입력 snapshot |
| `publication/` | GitHub 배포 경로 조정·파일 선별·출처 기록 |
| `source_file_manifest.json`, `upload_manifest.json` | 출처·변환 여부 및 현재 배포 파일 SHA256 |
| `verify_results.py` | 5부 5셀 + 6부 10셀을 학습 없이 독립 재검증 |

## 학습 없이 결과 재검증

Python 3.14.6에서 검증했습니다. 이 폴더를 현재 디렉터리로 실행합니다.

```bash
python -m pip install -r requirements-report.txt
python verify_results.py
```

파일 해시·673행의 역할·기존 360 outer/72 OOF 집계와 외부학습 비교 36 OOF/18 paired Δ를 검증합니다.
학습기를 호출하거나 저장된 노트북을 수정하지 않습니다. 원 작업 PC의 다른 폴더 없이 검사할 수 있습니다.
Jupyter에서는 **5부와 6부를 각각 첫 코드 셀부터** 실행할 수 있습니다.
1~4부의 과거 경로·GJC/EPA 원료 의존까지 이식하지 않았으므로 Run All은 재현 범위에 포함되지 않습니다.

## 보존·실행 범위

- 수정 파일은 같은 경로의 최신 내용으로 교체했습니다. 이전 내용은 Git 이력에 있습니다.
  원래 로컬 실험 파일과 기존 production 데이터·결과는 변경하지 않았습니다.
- `data/eye_vectorized_production_673.csv` SHA256:
  `811268d4813fcdd68e85458641bf5823239cae04e1fa871766c77e04f0221e7f`.
  저장소의 기존 `eye irritation/eye_vectorized.csv` 841행과 서로 다른 snapshot입니다.
- 원래 6모델 production 360 outer와 새 3모델 보조학습 90 outer는 구별합니다.
  과거 그룹화 실험03의 673행 CV도 현재 606행 비교와 합치지 않습니다.
- 선택한 원본 학습 코드·외부 데이터·모델은 byte 그대로 복사했습니다. 외부 checksum 목록은 선택 파일 기준으로 만들었습니다. `.gitattributes`로 줄바꿈 변환을 막았습니다.
  통합 노트북은 업로드 폴더의 상대 링크·입력 경로만 조정하고 6부 출력을 재실행했습니다. 5부 셀·출력은 보존했습니다.
- 학습 snapshot의 로컬 절대경로, 전체 ICE/성분 출처·Skin checksum 의존은 남아 있습니다.
  **이 패키지의 이식 검증 범위는 보고서 재검증**이며 새 PC에서 전체 학습이 바로 된다는 뜻은 아닙니다.
  [외부학습 패키지 안내](외부학습/README.md)를 함께 읽으세요.
- `build_eye_all_experiments_v2.py`는 과거 production 절 생성기의 원본 snapshot입니다.
  외부학습까지 통합한 최신본은 제공된 notebook을 사용하고, 검증은 `verify_results.py`로 수행합니다.
- 과거 7,200개 production 후보 중간 checkpoint는 포함하지 않았습니다.
  새 외부학습의 90 outer·1,800후보 JSON은 `검증체크포인트.zip`에 보존했습니다.
- `eye_vectorized_concentration.csv`는 전체 학습용 파생 export이며 CV feature 입력에 쓰지 않았습니다.
- 전체 673행 최종 EPA 후보 학습·배포·기존 모델 교체는 하지 않았습니다.
  포함한 `stoptx_teacher.joblib`은 성분 보조모델이며 완성된 화장품 배포 모델이 아닙니다.

이번 업로드에서는 완료 결과를 검증·게시했으며 모델을 재학습하지 않았습니다.

사용자의 최소 업로드 요청에 따라 외부 관련 파일은 **34개**로 선별했습니다. 원문 PDF, 다운로드 로그,
중복 노트북·PDF 그래프·보고서, 이동/백업 기록과 보조 분석 중간 파일은 로컬에만 보존합니다.
기존 production 자료는 유지하고 이번 추가 범위만 줄였습니다.
