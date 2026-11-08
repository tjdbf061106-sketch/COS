# -*- coding: utf-8 -*-
"""eye에도 skin과 동일한 방법론으로 농도 피처(known_pct) 효과를 검증 (eye팀 자체 검증 결과와
방향성이 같은지 교차 확인용). eye는 comp_group_id가 이미 조성해시 기반이라(PDF 확인됨) 그룹 재계산은
하지 않고, substance_type=='Mixture'만 사용(기존 eye_dataset() 필터와 동일), final_label로부터
binary_label_A(<=2)/binary_label_B(<=3)를 skin과 같은 방식으로 파생해 비교.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = r"C:\Users\UserK\학술제\eye irritation"
df = pd.read_csv(rf"{BASE}\eye_vectorized.csv", low_memory=False)
df = df[df["substance_type"] == "Mixture"].reset_index(drop=True)
df["binary_label_A"] = (df["final_label"] <= 2).astype(int)
df["binary_label_B"] = (df["final_label"] <= 3).astype(int)

BASE_FEATURE_COLS = [c for c in df.columns if c.startswith(("desc_", "fp_w_"))]
CONC_FEATURE_COLS = BASE_FEATURE_COLS + ["known_pct"]
print("feature dims:", len(BASE_FEATURE_COLS), "/", len(CONC_FEATURE_COLS))

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


def grouped_cv(df, feature_cols, label_col, model_name):
    ok = df[feature_cols].notna().all(axis=1).values
    X = df.loc[ok, feature_cols].to_numpy(dtype=float)
    y = df.loc[ok, label_col].to_numpy(dtype=int)
    g = df.loc[ok, "comp_group_id"].astype(str).to_numpy()

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
    return pd.DataFrame(rows), int(ok.sum())


results = []
t0 = time.time()
for label_name, label_col in [("A", "binary_label_A"), ("B", "binary_label_B")]:
    for model_name in MODELS:
        for feat_name, feat_cols in [("baseline", BASE_FEATURE_COLS), ("+known_pct", CONC_FEATURE_COLS)]:
            r, n = grouped_cv(df, feat_cols, label_col, model_name)
            results.append({
                "label": label_name, "model": model_name, "features": feat_name, "n": n,
                "mcc_mean": r["mcc"].mean(), "mcc_std": r["mcc"].std(),
                "sens_mean": r["sensitivity"].mean(), "spec_mean": r["specificity"].mean(),
                "roc_auc_mean": r["roc_auc"].mean(),
            })
            print(f"완료 ({time.time()-t0:.0f}s): label={label_name} model={model_name} features={feat_name}")

table = pd.DataFrame(results)
out_path = rf"{BASE}\eye_concentration_feature_effect.csv"
table.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n저장:", out_path)

pd.set_option("display.width", 200)
piv = table.pivot_table(index=["label", "model"], columns="features", values="mcc_mean")
piv["delta_mcc"] = piv["+known_pct"] - piv["baseline"]
print(piv.round(4).to_string())
print(f"\n총 소요시간: {time.time()-t0:.0f}초")
