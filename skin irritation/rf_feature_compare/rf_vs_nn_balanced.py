# -*- coding: utf-8 -*-
"""RF vs 신경망(NN) 맞대결 — 각 모델의 '균형점 threshold'에서 비교.

rf_vs_nn_head2head.py의 수정판.

무엇이 달라졌나:
  기존: 두 모델에 똑같이 threshold 0.5를 적용 → Sens/Spec 비교
  문제: 두 모델은 확률을 뱉는 방식(보정)이 다르다. 양성이 5.5%뿐이라
        RF는 확률이 낮은 쪽에 몰리고, class-weighted CE를 쓰는 NN은 더 퍼진다.
        같은 0.5라도 두 모델에서 뜻이 다르므로, 0.5 고정 비교는
        모델의 실력이 아니라 보정 차이를 재게 된다.
  수정: 각 모델에서 threshold를 쓸어 |Sens-Spec|이 최소가 되는 지점을 찾고,
        그 지점끼리 비교한다. ROC-AUC는 threshold와 무관하므로 그대로 쓴다.

사용법:
    python rf_vs_nn_balanced.py        # 기본 10회
    python rf_vs_nn_balanced.py 20
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

N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
KIND = "nz20"
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)

ds = B.load_skin()
cfg = dict(B.BASE, lr=0.001, num_hidden_layers=1, dropout=0.0)


def at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(thr=thr, sensitivity=sens, specificity=spec,
                gap=abs(sens - spec), mcc=matthews_corrcoef(y, pred))


def balance_point(y, proba):
    """|Sens-Spec|이 가장 작은 threshold. 동점이면 MCC가 높은 쪽."""
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
    print(f"skin: {len(ds)}건, {ds.input_dim}차원, feature={KIND}, 반복={N_REPS}")
    print("비교 지점: 각 모델의 |Sens-Spec| 최소 threshold (모델마다 다름)\n")

    rows = {"NN": [], "RF": []}
    fixed = {"NN": [], "RF": []}     # 참고용: 0.5 고정
    aucs = {"NN": [], "RF": []}

    for rep in range(N_REPS):
        oof = {
            "NN": B.run_cv(ds, KIND, random_state=rep, cfg=cfg,
                           group_aware_val=True, seed_offset=100 * rep)["oof"],
            "RF": run_rf(rep),
        }
        for name in ("NN", "RF"):
            p = oof[name]
            aucs[name].append(roc_auc_score(ds.y, p))
            rows[name].append(balance_point(ds.y, p))
            fixed[name].append(at(ds.y, p, 0.5))
        print(f"rep{rep}: "
              f"NN thr={rows['NN'][-1]['thr']:.2f} Sens {rows['NN'][-1]['sensitivity']:.3f} "
              f"Spec {rows['NN'][-1]['specificity']:.3f} | "
              f"RF thr={rows['RF'][-1]['thr']:.2f} Sens {rows['RF'][-1]['sensitivity']:.3f} "
              f"Spec {rows['RF'][-1]['specificity']:.3f}")

    def block(title, data):
        print(f"\n=== {title} ===")
        for name in ("NN", "RF"):
            r = data[name]
            line = f"{name:<3}"
            for k in ("thr", "sensitivity", "specificity", "gap", "mcc"):
                if k not in r[0]:
                    continue
                v = np.array([x[k] for x in r])
                line += f"  {k} {v.mean():.3f}±{v.std():.3f}"
            a = np.array(aucs[name])
            line += f"  roc_auc {a.mean():.3f}±{a.std():.3f}"
            print(line)

    block(f"균형점 threshold에서 ({N_REPS}회, 동일 분할)", rows)
    block("[참고] threshold 0.5 고정 — 보정 차이가 섞인 비교", fixed)

    print("\n=== 짝지은 비교 (같은 rep=같은 분할) ===")
    for label, a, b in [
        ("ROC-AUC (threshold 무관)", np.array(aucs["RF"]), np.array(aucs["NN"])),
        ("Sensitivity @균형점", np.array([x["sensitivity"] for x in rows["RF"]]),
                                 np.array([x["sensitivity"] for x in rows["NN"]])),
        ("MCC @균형점", np.array([x["mcc"] for x in rows["RF"]]),
                        np.array([x["mcc"] for x in rows["NN"]])),
        ("격차 @균형점 (작을수록 좋음)", np.array([x["gap"] for x in rows["RF"]]),
                                        np.array([x["gap"] for x in rows["NN"]])),
    ]:
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        t = d.mean() / se if se > 0 else float("nan")
        print(f"{label:<28} RF-NN 평균차 {d.mean():+.4f}  t={t:+.2f}  "
              f"RF 우세 {int((d > 0).sum())}/{N_REPS}")

    print("\n주의: 균형점 threshold는 보고 대상과 같은 OOF에서 고른 값이다.")
    print("      모델 간 '균형을 맞출 수 있는 능력'의 비교로 읽을 것이며,")
    print("      운용 threshold로 쓰려면 사전에 선언된 고정값을 따로 정해야 한다.")


if __name__ == "__main__":
    main()
