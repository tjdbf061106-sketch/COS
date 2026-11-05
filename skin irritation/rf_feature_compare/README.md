# skin 입력 비교 스크립트

## 뭘 하는가

1. 네 가지 입력(기술자+물성 / 핑거프린트 / nz20 / 전체)을 **같은 분할·같은 모델**로 비교
2. **짝지은 비교** — 같은 seed끼리 차이를 봐서, 평균 비교로는 묻히는 차이를 드러냄
3. 가장 좋은 입력에서 **threshold 전체 곡선**

## 왜 짝지은 비교인가

평균끼리 비교하면 seed마다 생기는 공통 변동이 그대로 남아서, 양성 30건 규모에서는
작은 차이가 전부 변동폭(±0.07)에 묻힌다. 같은 seed·같은 분할로 짝지으면 그 공통
변동이 상쇄되고 입력 차이만 남는다.

`t` 값이 ±2를 넘고 `우세`가 한쪽으로 쏠리면 실제 차이, 아니면 노이즈다.

## 실행

```
python skin_compare.py        # 20 seed
python skin_compare.py 30     # 30 seed
```

스크립트 상단의 `BASE`를 `skin irritation` 폴더 경로로 바꿀 것.

## 필요 파일

- `skin_filtered.csv`
- `casrn_smiles_cache.csv`
- `casrn_descriptors_full.csv`
- `skin_vectorized.csv` (라벨·comp_group_id)

## 설정 바꾸는 곳

- `AGG` — 성분을 제형으로 접는 방식 (wmean/top/max/min)
- `THRESHOLDS` — 쓸어볼 판정 기준값
- 모델은 `oof_probs()` 안의 `RandomForestClassifier` 한 줄

신경망과 비교하려면 그 한 줄만 바꾸면 나머지(분할·집계·지표)는 동일하게 유지된다.

## 주의

- 분할은 `comp_group_id` 기준 그룹 분할. 같은 조성이 train/test에 갈리지 않게 하기 위함
- `nz20` 필터는 전체 데이터 기준이라 엄밀히는 fold 내부에서 해야 한다(미세한 leakage).
  입력 비교 목적에서는 네 조건에 동일하게 적용되므로 비교 자체는 유효
