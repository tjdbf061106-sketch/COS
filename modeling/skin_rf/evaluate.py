"""
train.py가 저장한 skin RF checkpoint로 held-out 테스트셋 평가.

사용법: py evaluate.py --checkpoint skin_rf_checkpoint.joblib
"""
import argparse

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from datasets import SKIN_CSV, skin_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="skin_rf_checkpoint.joblib")
    parser.add_argument("--skin-csv", default=SKIN_CSV)
    args = parser.parse_args()

    ckpt = joblib.load(args.checkpoint)
    ds = skin_dataset(args.skin_csv)

    X = ds.X[:, ckpt["nz_filter_keep"]]
    test_idx = ckpt["test_idx"]
    X_test = X[test_idx]
    y_test = ds.y[test_idx]

    proba = ckpt["model"].predict_proba(X_test)[:, 1]
    thr = ckpt["threshold"]
    pred = (proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"=== skin 테스트셋 (n={len(y_test)}, threshold={thr:.3f}) ===")
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    print(f"격차: {abs(sens - spec):.3f}")
    print(f"MCC: {matthews_corrcoef(y_test, pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print("혼동행렬 [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["non-toxic(0)", "toxic(1)"]))
    print("참고: 이 단일분할 수치는 배포 체크포인트 확인용이며, 공식 성능 보고는 반복 group "
          "5-fold(skin irritation/rf_feature_compare/)를 기준으로 함.")


if __name__ == "__main__":
    main()
