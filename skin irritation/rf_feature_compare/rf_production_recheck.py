# -*- coding: utf-8 -*-
"""B5배치(7건) 추가 후 production RF 성능 재검증 — 반복 group 5-fold, 균형점 threshold.
rf_vs_nn_balanced.py의 RF 부분만 분리(NN은 이미 폐기된 비교 대상이라 불필요).

사용법:
    python rf_production_recheck.py        # 기본 20회
    python rf_production_recheck.py 30
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "modeling", "eval_b"))
sys.path.insert(0, os.path.join(REPO, "modeling", "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix

import bharness as B
from splits import grouped_kfold_splits

N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
KIND = "nz20"
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)

ds = B.load_skin()


def at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(thr=thr, sensitivity=sens, specificity=spec,
                gap=abs(sens - spec), mcc=matthews_corrcoef(y, pred))


def balance_point(y, proba):
    cand = [at(y, proba, t) for t in GRID]
    best = min(cand, key=lambda d: (round(d["gap"], 4), -d["mcc"]))
    return best


def run_rf(rep):
    folds = grouped_kfold_splits(ds, n_splits=5, random_state=rep)
    oof = np.zeros(len(ds.y))
    for tr, te in folds:
        tf = B.make_transform(KIND, ds.feature_cols)
        Xtr = tf.fit_transform(ds.X[tr], ds.y[tr])
        Xte = tf.transform(ds.X[te])
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=rep, n_jobs=-1)
        m.fit(Xtr, ds.y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def main():
    print(f"skin: {len(ds)}건(binary_label 분포 {dict(zip(*np.unique(ds.y, return_counts=True)))}), "
          f"{ds.input_dim}차원, feature={KIND}, 반복={N_REPS}")

    rows = []
    aucs = []
    thrs = []
    for rep in range(N_REPS):
        oof = run_rf(rep)
        aucs.append(roc_auc_score(ds.y, oof))
        bp = balance_point(ds.y, oof)
        rows.append(bp)
        thrs.append(bp["thr"])
        print(f"rep{rep}: thr={bp['thr']:.2f} Sens {bp['sensitivity']:.3f} "
              f"Spec {bp['specificity']:.3f} gap {bp['gap']:.3f} MCC {bp['mcc']:.3f}")

    print(f"\n=== 균형점 threshold에서 ({N_REPS}회, group 5-fold) ===")
    for k in ("sensitivity", "specificity", "gap", "mcc"):
        v = np.array([r[k] for r in rows])
        print(f"  {k}: {v.mean():.3f} ± {v.std():.3f}")
    a = np.array(aucs)
    print(f"  roc_auc: {a.mean():.3f} ± {a.std():.3f}")
    t = np.array(thrs)
    print(f"  threshold: {t.mean():.3f} ± {t.std():.3f}")


if __name__ == "__main__":
    main()
