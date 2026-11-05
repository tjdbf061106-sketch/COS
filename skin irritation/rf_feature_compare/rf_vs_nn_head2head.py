# -*- coding: utf-8 -*-
"""RF vs 신경망(NN) 직접 맞대결 — 완전히 동일한 데이터·분할·feature로 비교.
목적: "같은 skin인데 왜 두 모델을 각자 쓰냐"는 지적에 답하기 위해, 둘 중 뭐가 진짜 더 나은지
공정하게(같은 ds, 같은 group 5-fold 분할, 같은 feature 변환) 결정한다."""
import sys
sys.path.insert(0, r"C:\Users\UserK\학술제\modeling\eval_b")
sys.path.insert(0, r"C:\Users\UserK\학술제\modeling\src")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix

import bharness as B
from splits import grouped_kfold_splits

ds = B.load_skin()
cfg = dict(B.BASE, lr=0.001, num_hidden_layers=1, dropout=0.0)
N_REPS = 10
KIND = "nz20"  # 현재 채택 중인 feature


def summarize(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                mcc=matthews_corrcoef(y, pred), roc_auc=roc_auc_score(y, proba))


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


nn_aucs, rf_aucs = [], []
nn_rows, rf_rows = [], []
for rep in range(N_REPS):
    r_nn = B.run_cv(ds, KIND, random_state=rep, cfg=cfg, group_aware_val=True, seed_offset=100 * rep)
    oof_nn = r_nn["oof"]
    oof_rf = run_rf(rep)

    m_nn = summarize(ds.y, oof_nn)
    m_rf = summarize(ds.y, oof_rf)
    nn_rows.append(m_nn); rf_rows.append(m_rf)
    nn_aucs.append(m_nn["roc_auc"]); rf_aucs.append(m_rf["roc_auc"])
    print(f"rep{rep}: NN  Sens {m_nn['sensitivity']:.3f} Spec {m_nn['specificity']:.3f} "
          f"격차 {m_nn['gap']:.3f} MCC {m_nn['mcc']:.3f} AUC {m_nn['roc_auc']:.3f}")
    print(f"      RF  Sens {m_rf['sensitivity']:.3f} Spec {m_rf['specificity']:.3f} "
          f"격차 {m_rf['gap']:.3f} MCC {m_rf['mcc']:.3f} AUC {m_rf['roc_auc']:.3f}")

print("\n=== 평균±표준편차 (nz20, 동일 분할 10회) ===")
for name, rows in [("NN", nn_rows), ("RF", rf_rows)]:
    for k in ["sensitivity", "specificity", "gap", "mcc", "roc_auc"]:
        vals = [r[k] for r in rows]
        print(f"{name} {k:<12} {np.mean(vals):.3f}±{np.std(vals):.3f}", end="   ")
    print()

print("\n=== 짝지은 비교 (같은 rep=같은 분할) — ROC-AUC ===")
diff = np.array(rf_aucs) - np.array(nn_aucs)
se = diff.std(ddof=1) / np.sqrt(len(diff))
t = diff.mean() / se if se > 0 else float("nan")
print(f"RF - NN: 평균차 {diff.mean():+.4f}  t={t:+.2f}  RF가 나은 회차 {int((diff>0).sum())}/{N_REPS}")

print("\n=== 짝지은 비교 — MCC ===")
mcc_nn = np.array([r["mcc"] for r in nn_rows])
mcc_rf = np.array([r["mcc"] for r in rf_rows])
diff_mcc = mcc_rf - mcc_nn
se2 = diff_mcc.std(ddof=1) / np.sqrt(len(diff_mcc))
t2 = diff_mcc.mean() / se2 if se2 > 0 else float("nan")
print(f"RF - NN: 평균차 {diff_mcc.mean():+.4f}  t={t2:+.2f}  RF가 나은 회차 {int((diff_mcc>0).sum())}/{N_REPS}")
