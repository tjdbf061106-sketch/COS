"""
skin 자극성 — RandomForest production 학습.

2026-11-05: RF vs 신경망(NN) 정면 비교 결과(skin irritation/rf_feature_compare/) RF 채택.
skin/eye는 각자 최적 모델을 독립적으로 쓰기로 한 팀 결정에 따라, 기존 멀티태스크 NN
harness(modeling/src)와 완전히 분리된 skin 전용 파이프라인이다. 멀티태스크 NN은
원 도메인(NTP ICE) 내부 검증·발표자료용으로 modeling/src에 그대로 남겨둔다.

사용법:
    py train.py                    # 기본 설정
    py train.py --save-path my.joblib

평가 원칙(기존과 동일):
  - group-aware 분할(comp_group_id 기준)로 동일 조성 leakage 방지
  - feature 선택(nz20)·threshold는 train(또는 val) 내부에서만 결정, test는 보지 않음
  - 성능은 Sensitivity/Specificity/MCC/ROC-AUC를 함께 보고, 단일 지표로 판단하지 않음
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from datasets import SKIN_CSV, skin_dataset  # noqa: E402
from splits import group_train_test_split, group_train_test_split_from_subset  # noqa: E402

THRESHOLD_GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


class NonZeroFilter:
    """train 샘플 기준 nonzero 개수가 min_count 미만인 feature 컬럼 제거(nz20).
    modeling/src/train.py의 동일 클래스와 같은 역할 — skin RF 파이프라인이 NN harness와
    독립적이어야 해서 별도로 둔다. 반드시 train(또는 trainval) 인덱스에서만 fit."""

    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X):
        self.keep_ = (X != 0).sum(axis=0) >= self.min_count
        if self.keep_.sum() == 0:
            self.keep_[:] = True
        return self

    def transform(self, X):
        return X[:, self.keep_]


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                mcc=matthews_corrcoef(y, pred), roc_auc=roc_auc_score(y, proba),
                threshold=thr)


def balance_threshold(y, proba):
    """|Sens-Spec|이 가장 작은 threshold. 동점이면 MCC가 높은 쪽(rf_vs_nn_balanced.py와 동일 규칙)."""
    cand = [metrics_at(y, proba, t) for t in THRESHOLD_GRID]
    best = min(cand, key=lambda d: (round(d["gap"], 4), -d["mcc"]))
    return best["threshold"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin-csv", default=SKIN_CSV)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--nz-min-count", type=int, default=20)
    parser.add_argument("--val-size", type=float, default=0.2, help="threshold 선택용 검증셋 비율")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="skin_rf_checkpoint.joblib")
    args = parser.parse_args()

    ds = skin_dataset(args.skin_csv)
    print(f"skin: {len(ds)}건, feature {ds.input_dim}차원")

    trainval_idx, test_idx = group_train_test_split(ds, args.test_size, args.seed)
    train_idx, val_idx = group_train_test_split_from_subset(ds, trainval_idx, args.val_size, args.seed)

    # 1) train만으로 nz20 fit, threshold 선택용 모델 학습(검증셋 자체도 test 정보는 안 봄)
    filt = NonZeroFilter(args.nz_min_count).fit(ds.X[train_idx])
    X_train = filt.transform(ds.X[train_idx])
    X_val = filt.transform(ds.X[val_idx])
    X_test = filt.transform(ds.X[test_idx])

    tuner = RandomForestClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                    random_state=args.seed, n_jobs=-1)
    tuner.fit(X_train, ds.y[train_idx])
    val_proba = tuner.predict_proba(X_val)[:, 1]
    threshold = balance_threshold(ds.y[val_idx], val_proba)
    print(f"검증셋 기준 선택된 threshold: {threshold:.3f}")

    # 2) 배포용 최종 모델은 trainval(train+val) 전체로 재학습 (threshold는 위에서 고정한 값 그대로 사용)
    filt_final = NonZeroFilter(args.nz_min_count).fit(ds.X[trainval_idx])
    X_trainval = filt_final.transform(ds.X[trainval_idx])
    X_test_final = filt_final.transform(ds.X[test_idx])

    model = RandomForestClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                    random_state=args.seed, n_jobs=-1)
    model.fit(X_trainval, ds.y[trainval_idx])

    test_proba = model.predict_proba(X_test_final)[:, 1]
    test_metrics = metrics_at(ds.y[test_idx], test_proba, threshold)
    print(f"테스트셋(n={len(test_idx)}) 단일분할 평가(참고용, 공식 보고 수치 아님):")
    print(f"  Sens {test_metrics['sensitivity']:.3f} / Spec {test_metrics['specificity']:.3f} / "
          f"격차 {test_metrics['gap']:.3f} / MCC {test_metrics['mcc']:.3f} / "
          f"ROC-AUC {test_metrics['roc_auc']:.3f}")

    checkpoint = {
        "model": model,
        "nz_filter_keep": filt_final.keep_,
        "threshold": threshold,
        "feature_cols": ds.feature_cols,
        "test_idx": test_idx,
        "n_estimators": args.n_estimators,
        "nz_min_count": args.nz_min_count,
        "seed": args.seed,
    }
    joblib.dump(checkpoint, args.save_path)
    print(f"저장 완료: {args.save_path} (evaluate.py에서 로드해 테스트셋 평가)")


if __name__ == "__main__":
    main()
