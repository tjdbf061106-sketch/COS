# -*- coding: utf-8 -*-
"""농도 피처(resolved_pct) 추가 전/후 효과를, 수정된 comp_group_id 기준 반복 group 5-fold로 비교.

방법론은 skin_final_combined_eval.ipynb(팀 기존 "공식" 비교 방식)과 동일하게 맞춤:
  StratifiedGroupKFold(5-fold) x 5seed, RandomForest/ExtraTrees(n_estimators=500), threshold=0.5,
  MCC/Sensitivity/Specificity/ROC-AUC, overall/single/multi 하위집단.
차이점: comp_group_id는 이번에 고친 최종판(서명 6자리 + 벡터해시에 농도 포함)을 쓰고,
feature_cols은 production 기본(AromaticProportion+desc_*+fp_w_*) vs +resolved_pct 두 가지를 비교.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = r"C:\Users\UserK\학술제\skin irritation"
vec = pd.read_csv(rf"{BASE}\skin_vectorized.csv", low_memory=False)

BASE_FEATURE_COLS = ["AromaticProportion"] + [c for c in vec.columns if c.startswith(("desc_", "fp_w_"))]
CONC_FEATURE_COLS = BASE_FEATURE_COLS + ["resolved_pct"]

MODELS = {
    "ExtraTrees": lambda seed: ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
}

SEEDS = range(5)


def subgroup_metrics(y, prob, mask):
    yy, pp = y[mask], prob[mask]
    if len(yy) == 0 or len(set(yy)) < 2:
        return dict(n=int(mask.sum()), n_pos=int(yy.sum()) if len(yy) else 0,
                    mcc=np.nan, sensitivity=np.nan, specificity=np.nan, roc_auc=np.nan)
    pred05 = (pp >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred05, labels=[0, 1]).ravel()
    return dict(
        n=int(mask.sum()), n_pos=int(yy.sum()),
        mcc=matthews_corrcoef(yy, pred05),
        sensitivity=tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        specificity=tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        roc_auc=roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan,
    )


def grouped_cv(df, feature_cols, label_col, model_name):
    ok = df[feature_cols].notna().all(axis=1).values
    X = df.loc[ok, feature_cols].to_numpy(dtype=float)
    y = df.loc[ok, label_col].to_numpy(dtype=int)
    g = df.loc[ok, "comp_group_id"].astype(str).to_numpy()
    comp_count = df.loc[ok, "comp_count"].to_numpy()
    single_mask = comp_count == 1
    multi_mask = comp_count > 1

    rows = []
    for s in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            assert set(g[tr]).isdisjoint(g[te]), "그룹 누수 발생"
            m = MODELS[model_name](s).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        for sg_name, mask in [("overall", np.ones(len(y), dtype=bool)), ("single", single_mask), ("multi", multi_mask)]:
            row = subgroup_metrics(y, oof, mask)
            row.update(seed=s, subgroup=sg_name)
            rows.append(row)
    return pd.DataFrame(rows)


results = []
t0 = time.time()
for label_name, label_col in [("A", "binary_label_A"), ("B", "binary_label_B")]:
    for model_name in MODELS:
        for feat_name, feat_cols in [("baseline", BASE_FEATURE_COLS), ("+resolved_pct", CONC_FEATURE_COLS)]:
            r = grouped_cv(vec, feat_cols, label_col, model_name)
            for sg in ["overall", "single", "multi"]:
                sub = r[r.subgroup == sg]
                results.append({
                    "label": label_name, "model": model_name, "features": feat_name, "subgroup": sg,
                    "n": int(sub["n"].iloc[0]), "n_pos": int(sub["n_pos"].iloc[0]),
                    "mcc_mean": sub["mcc"].mean(), "mcc_std": sub["mcc"].std(),
                    "sens_mean": sub["sensitivity"].mean(), "spec_mean": sub["specificity"].mean(),
                    "roc_auc_mean": sub["roc_auc"].mean(),
                })
            print(f"완료 ({time.time()-t0:.0f}s): label={label_name} model={model_name} features={feat_name}")

table = pd.DataFrame(results)
out_path = rf"{BASE}\skin_concentration_feature_effect.csv"
table.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n저장:", out_path)

pd.set_option("display.width", 200)
print()
for sg in ["overall", "single", "multi"]:
    print(f"========== subgroup={sg} ==========")
    sub = table[table.subgroup == sg].copy()
    piv = sub.pivot_table(index=["label", "model"], columns="features", values="mcc_mean")
    piv["delta_mcc"] = piv["+resolved_pct"] - piv["baseline"]
    print(piv.round(4).to_string())
    print()

print(f"총 소요시간: {time.time()-t0:.0f}초")
