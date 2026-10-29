# 멀티태스크 독성예측 모델 (harness)

skin/eye ICE 벡터화 데이터로 Multi-task PyTorch 이분류 모델을 학습/평가하는 코드 스켈레톤.
설계 배경은 `../세팅/멀티태스크_모델링_정리.md` 참조.

## 환경 설정
```
pip install -r requirements.txt
```

## 구조
```
modeling/
  src/
    datasets.py   skin/eye CSV -> PyTorch Dataset (binary_label/epa_class 라벨 처리)
    model.py      MultiTaskIrritationModel (skin_encoder/eye_encoder + shared_trunk + task head)
    splits.py     comp_group_id/group_id 기준 group-aware train/test split (leakage 방지)
    train.py      학습 루프 (skin_loss*alpha + eye_loss*beta)
    evaluate.py   held-out 테스트셋 F1/ROC-AUC/혼동행렬
  requirements.txt
```

## 실행
```
cd src
py train.py --epochs 50 --balance weighted
py evaluate.py --checkpoint checkpoint.pt
```

## 알려진 전제/미결 사항 (실행 전 확인)
- eye는 `substance_type == 'Mixture'`만 사용(순물질 Chemical 241건 제외) — `datasets.py`에 이미 반영됨
- eye `binary_label` 컬럼이 없어 `epa_class`(Category 1~4)에서 자동 파생(1~2→1, 3~4→0) — skin과 동일 기준
- 클래스 불균형: 기본은 `--balance weighted`(class-weighted CE), 성능 안 나오면 `--balance 1to1` 시도 (단 skin 양성 29건뿐이라 데이터 손실 큼 — 최후수단)
- alpha/beta는 벡터합이 아니라 skin_loss/eye_loss 가중합(멀티태스크 loss weighting)으로 구현됨 — 근거는 정리 문서 5절 참조
- PyTorch 미설치 상태로 코드 syntax만 확인함(`py -m py_compile`) — 실제 학습 실행 전 `pip install -r requirements.txt` 필요
