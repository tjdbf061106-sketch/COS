# eval_b — B 파트(입력 feature · 평가) 재현 하네스

작성 2026-10-30. 1단계(모델·학습 개선, A 담당)에 이어 **4번 feature 차원축소 / 5번 threshold 조정 /
6번 group 5-fold 재검증**을 돌린 코드와 결과다.

`modeling/src`의 `datasets.py` / `splits.py` / `train.py`를 **수정 없이 그대로 import**하고,
A가 notebook에서 쓴 평가 프로토콜(skin-only MLP, StratifiedGroupKFold 5-fold OOF, fold별
StandardScaler, class-weighted CE, early stopping patience=15)을 그대로 따른다.

## 실행

```
pip install -r ../requirements.txt     # torch, scikit-learn, pandas, matplotlib
cd modeling/eval_b
python run_baseline.py                 # A 채택 설정 1회 재현 (~10초)
python run_replicate_A.py              # A의 1~3단계를 A 코드 그대로 재현 (~10초)
python run_esdiag.py                   # 조기종료 진단: epoch별 loss/AUC (~20초)
python run_b4.py                       # feature 12종 비교, 반복 10회 (~10분)
python run_b5.py                       # threshold sweep + nested threshold (~15분)
python run_b6.py                       # 단계별 반복 CV 변동폭 (~1분)
python run_final.py                    # 차원축소 후보 짝지은 비교 20회 (~5분)
python make_figdata.py && python make_figures.py   # figs/*.png 재생성
```

경로는 전부 이 파일 기준 상대경로라 레포 어디에 clone해도 그대로 돌아간다.
`skin_vectorized.csv`는 `../../skin irritation/`에서 읽는다.

## 파일

| 파일 | 역할 |
|---|---|
| `bharness.py` | 공용 하네스. feature 변환(fold 내부 fit), 학습 1회, CV, 반복 CV, 지표 |
| `run_baseline.py` | A 채택 설정 단일 재현 |
| `run_replicate_A.py` | A notebook 함수를 그대로 복제해 1~3단계 재현 |
| `run_esdiag.py` | 조기종료가 epoch 1~3에 멈추는 원인 진단 |
| `run_b4.py` | feature 선별/차원축소 12종 비교 → `b4_results.json` |
| `run_b5.py` | threshold sweep(OOF) + nested threshold |
| `run_b6.py` | 단계별 반복 CV 변동폭 |
| `run_final.py` | 차원축소 후보 짝지은 비교 → `final_results.json` |
| `make_figdata.py` / `make_figures.py` | `figdata.json` → `figs/*.png` |

## 결과 요약

자세한 수치와 해석은 **`claude/B_feature_threshold_cv_results.md`** 참조. 그림 4장:

| 그림 | 내용 |
|---|---|
| `figs/fig1_stage_mcc_variance.png` | A의 1~3단계 차이(0.011~0.029)가 측정 변동폭(±0.03~0.05) 안에 있음 |
| `figs/fig2_threshold_tradeoff.png` | threshold를 내려도 recall이 공짜로 안 생김 / MCC 최대점은 0.95 |
| `figs/fig3_feature_reduction_paired.png` | 차원축소는 MCC를 못 올리지만 격차·Sensitivity를 개선 |
| `figs/fig4_earlystopping_diagnosis.png` | train loss 5 epoch 수렴, val loss epoch 2부터 단조 증가 |

핵심 세 줄:

1. **단일 분할 1회 수치로는 설정 간 우열을 판정할 수 없다.** A 코드를 같은 시드로 돌려도 이 환경에서는
   단계별 순서가 뒤집힌다. 앞으로 성능 보고는 **반복 group 5-fold 평균±표준편차**로.
2. **차원축소 권고: `nonzero<20` 컬럼 제거(2,266 → 444차원).** MCC는 그대로지만 Sens +0.063,
   Sens-Spec 격차 -0.097, ROC-AUC +0.019 (20회 짝지은 비교), 변동폭도 감소.
3. **threshold는 0.5 유지 권고.** 검증셋에서 고른 threshold가 0.05~0.95로 흔들려(sd 0.30)
   nested로 적용하면 고정 0.5보다 나빠진다. 양성 30건에서는 튜닝할 근거가 부족하다.

## `modeling/src`에 반영 제안 (이 폴더는 src를 건드리지 않음)

1. `train.py`의 조기종료용 내부 val 분할이 `rng.permutation`이라 **group-aware가 아니다**.
   조성 쌍이 train/val로 갈리므로 `splits.group_train_test_split_from_subset`으로 교체.
2. 검증셋 양성이 0건이면 val loss가 조기종료 신호로 쓸모없어진다 → 재분할 폴백
   (`bharness.fit_one`에 구현해둔 것 참고).
3. feature 필터는 `datasets.py`가 아니라 **fold 내부 파이프라인**에 넣을 것.
   전체 데이터에 fit하면 test 정보가 새어 들어간다.
4. 성능 보고 함수를 반복 CV 평균±표준편차 형태로 통일.
