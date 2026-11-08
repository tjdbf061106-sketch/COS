# -*- coding: utf-8 -*-
"""comp_group_id 수정(벡터해시에 농도 포함, 서명 6자리) 자체가 성능에 미친 영향만 분리 측정.
feature는 baseline(desc/fp/AromaticProportion)으로 고정하고, 그룹만 구버전(vectoronly_v3, 농도 미포함
벡터해시로 과병합된 463그룹) vs 신버전(comp_group_id, 637그룹)으로 비교 — "그룹을 제대로 쪼갰더니
보고된 성능 중 일부가 누수로 부풀려져 있었는지"를 확인.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = r"C:\Users\UserK\학술제\skin irritation"
vec = pd.read_csv(rf"{BASE}\skin_vectorized.csv", low_memory=False)

FEATURE_COLS = ["AromaticProportion"] + [c for c in vec.columns if c.startswith(("desc_", "fp_w_"))]

MODELS = {
    "ExtraTrees": lambda seed: ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
}
SEEDS = range(5)


def metrics(y, prob):
    pred05 = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred05, labels=[0, 1]).ravel()
    return dict(
        mcc=matthews_corrcoef(y, pred05),
        sensitivity=tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        specificity=tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        roc_auc=roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan,
    )


def grouped_cv(df, group_col, label_col, model_name):
    ok = df[FEATURE_COLS].notna().all(axis=1).values
    X = df.loc[ok, FEATURE_COLS].to_numpy(dtype=float)
    y = df.loc[ok, label_col].to_numpy(dtype=int)
    g = df.loc[ok, group_col].astype(str).to_numpy()

    rows = []
    for s in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            assert set(g[tr]).isdisjoint(g[te]), "그룹 누수 발생"
            m = MODELS[model_name](s).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        row = metrics(y, oof)
        row["seed"] = s
        rows.append(row)
    return pd.DataFrame(rows)


results = []
t0 = time.time()
for label_name, label_col in [("A", "binary_label_A"), ("B", "binary_label_B")]:
    for model_name in MODELS:
        for grp_name, grp_col in [("old(vectoronly_v3, 463그룹)", "comp_group_id_vectoronly_v3"),
                                   ("new(comp_group_id, 637그룹)", "comp_group_id")]:
            r = grouped_cv(vec, grp_col, label_col, model_name)
            results.append({
                "label": label_name, "model": model_name, "grouping": grp_name,
                "n_groups": vec[grp_col].nunique(),
                "mcc_mean": r["mcc"].mean(), "mcc_std": r["mcc"].std(),
                "sens_mean": r["sensitivity"].mean(), "spec_mean": r["specificity"].mean(),
                "roc_auc_mean": r["roc_auc"].mean(),
            })
            print(f"완료 ({time.time()-t0:.0f}s): label={label_name} model={model_name} grouping={grp_name}")

table = pd.DataFrame(results)
out_path = rf"{BASE}\skin_grouping_fix_effect.csv"
table.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n저장:", out_path)

pd.set_option("display.width", 200)
piv = table.pivot_table(index=["label", "model"], columns="grouping", values="mcc_mean")
piv["delta_mcc(new-old)"] = piv.iloc[:, 1] - piv.iloc[:, 0]
print(piv.round(4).to_string())
print(f"\n총 소요시간: {time.time()-t0:.0f}초")
