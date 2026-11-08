# -*- coding: utf-8 -*-
"""팀원이 원래 요청한 (RF,ExtraTrees) x (wmean,top) 4조합 반복 group 5-fold.
작업1(농도 피처)/작업2(comp_group_id 수정) 이후에 돌리는 "작업3".
최신 권장사항(농도 피처 채택)을 반영해 resolved_pct 유무 축도 같이 비교(2x2x2x2=16칸),
threshold는 0.5 고정과 균형점(balanced) 둘 다 보고해 지난번 같은 threshold 아티팩트를 피한다.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = r"C:\Users\UserK\학술제\skin irritation"
SRC_B5 = r"C:\Users\UserK\OneDrive - 덕성여자대학교\문서\카카오톡 받은 파일\skin_irritation_최종후보.xlsx"
VEC_PATH = rf"{BASE}\skin_vectorized.csv"

vec = pd.read_csv(VEC_PATH, low_memory=False)
desc_full = pd.read_csv(rf"{BASE}\casrn_descriptors_full.csv").set_index("casrn")
PROD_FEATURE_COLS = ["AromaticProportion"] + [c for c in desc_full.columns if c.startswith(("desc_", "fp_w_"))]

# ------------------------------------------------------------------ 원자료 재구성 (fix_comp_sig_precision_and_add_concentration.py와 동일)
filt = pd.read_csv(rf"{BASE}\skin_filtered.csv")
known = filt.dropna(subset=["casrn"]).copy()
known = known.drop_duplicates(subset=["formulation_id", "casrn"], keep="first")
known["has_descriptor"] = known["casrn"].isin(desc_full.index)
known = known.rename(columns={"casrn": "key", "percent_active_ingredient": "pct"})
known_unified = known[["formulation_id", "key", "pct", "has_descriptor"]].copy()

aug_meta = vec.loc[vec["formulation_id"].astype(str).str.startswith("AUG_"), ["formulation_id"]].copy()
aug_meta["formulation_id"] = aug_meta["formulation_id"].str.replace("AUG_", "", regex=False)

NUMBER = r"\d+(?:\.\d+)?"
RANGE_RE = re.compile(rf"^({NUMBER})[-~]({NUMBER})%?$")
NUM_RE = re.compile(rf"^({NUMBER})%?$")
OCR_RE = re.compile(rf"\(OCR for\s*({NUMBER})\s*%?\)")
UNRELIABLE_MARKERS = ["inconsistency", "garbled", "international units", "per mg of product"]
SUFFIXES = [" percent by weight", " percent", "% by weight", "% w/w", " w/w", "% a.i.", " a.i.", "%"]


def clean_text(text):
    cleaned = str(text).strip().replace(" ", "").replace(",", "").strip("*'")
    for suf in SUFFIXES:
        suf_clean = suf.replace(" ", "")
        if cleaned.lower().endswith(suf_clean.lower()):
            cleaned = cleaned[: -len(suf_clean)]
            break
    return cleaned.lstrip(".")


def parse_bounds(text):
    if pd.isna(text):
        return None, None
    raw = str(text).strip()
    if any(m in raw.lower() for m in UNRELIABLE_MARKERS):
        return None, None
    m = OCR_RE.search(raw)
    if m:
        v = float(m.group(1))
        return v, v
    cleaned = clean_text(raw)
    if cleaned.count(".") > 1:
        return None, None
    try:
        m = RANGE_RE.match(cleaned)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = NUM_RE.match(cleaned if cleaned.endswith("%") else cleaned + "%")
        if m:
            v = float(m.group(1))
            return v, v
        if re.fullmatch(NUMBER, cleaned):
            v = float(cleaned)
            return v, v
    except ValueError:
        return None, None
    return None, None


def resolve_pct_mid(row):
    if pd.notna(row.get("정확값")):
        return float(row["정확값"])
    lo, hi = row.get("하한"), row.get("상한")
    if pd.notna(lo) and pd.notna(hi):
        return (float(lo) + float(hi)) / 2
    lo, hi = parse_bounds(row.get("함량_원문"))
    if lo is None:
        return np.nan
    return (lo + hi) / 2


B5_KEYS = ["v4:B5_003", "v4:B5_006", "v4:B5_009", "v4:B5_018", "v4:C011", "v4:C012", "v4:C016"]
all_aug_keys = aug_meta["formulation_id"].tolist()
v2v3_keys = [k for k in all_aug_keys if k.startswith("v2:") or k.startswith("v3:")]
v4_other_keys = [k for k in all_aug_keys if k.startswith("v4:") and k not in B5_KEYS]

comp1 = pd.read_excel(rf"{BASE}\skin_보강후보_통합.xlsx", sheet_name="성분함량")
comp1 = comp1[comp1["후보키"].isin(v2v3_keys)].copy()
comp1["key"] = comp1["CAS_원문"].fillna(comp1["성분명_원문"])
comp2 = pd.read_excel(rf"{BASE}\skin_보강후보_v4_신규.xlsx", sheet_name="성분함량")
comp2 = comp2[comp2["후보키"].isin(v4_other_keys)].copy()
comp2["key"] = comp2["CAS_원문"].fillna(comp2["성분명_원문"])

ing_rows = []
for comp in (comp1, comp2):
    for _, r in comp.iterrows():
        ing_rows.append({"formulation_id": r["후보키"], "key": r["key"], "pct": resolve_pct_mid(r)})

verif = pd.read_excel(SRC_B5, sheet_name="V4성분검증")
verif = verif[verif["후보키"].isin(B5_KEYS)].copy()
used = verif[verif["used"] == True].copy()  # noqa: E712
used["key"] = used["resolved_cas"].fillna(used["성분명_원문"])
for _, r in used.iterrows():
    ing_rows.append({"formulation_id": r["후보키"], "key": r["key"], "pct": float(r["parsed_exact_pct"])})

ing = pd.DataFrame(ing_rows)
ing["has_descriptor"] = ing["key"].isin(desc_full.index)
aug_unified = ing[["formulation_id", "key", "pct", "has_descriptor"]].copy()

raw_all = pd.concat([known_unified, aug_unified], ignore_index=True)
print("원자료 로드 완료:", len(raw_all), "행 /", raw_all["formulation_id"].nunique(), "제형")

# ------------------------------------------------------------------ top 방식 feature 재계산 (wmean은 production 그대로 사용)
top_rows = {}
for fid, grp in raw_all.groupby("formulation_id"):
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        continue
    pct = g_resolved["pct"].fillna(1.0).values
    idx = int(np.argmax(pct))
    key = g_resolved["key"].values[idx]
    top_rows[fid] = desc_full.loc[key, PROD_FEATURE_COLS].to_dict()

top_df = pd.DataFrame.from_dict(top_rows, orient="index")
top_df.index.name = "fid_raw"
top_df = top_df.reset_index()
is_aug = top_df["fid_raw"].isin(aug_meta["formulation_id"])
top_df["formulation_id"] = top_df["fid_raw"]
top_df.loc[is_aug, "formulation_id"] = "AUG_" + top_df.loc[is_aug, "fid_raw"]
top_df = top_df.drop(columns=["fid_raw"]).set_index("formulation_id")

print("top 방식 feature 계산 완료:", top_df.shape)

META_COLS = ["comp_group_id", "comp_count", "resolved_pct", "binary_label_A", "binary_label_B"]
meta = vec.set_index("formulation_id")[META_COLS]

wmean_full = meta.join(vec.set_index("formulation_id")[PROD_FEATURE_COLS])
top_full = meta.join(top_df)

DATASETS = {"wmean": wmean_full, "top": top_full}

MODELS = {
    "ExtraTrees": lambda seed: ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed
    ),
}
SEEDS = range(5)


def best_balanced(y, prob):
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    best = None
    for t in grid:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        gap = abs(sens - spec)
        mcc = matthews_corrcoef(y, pred)
        if best is None or gap < best[0] or (gap == best[0] and mcc > best[1]):
            best = (gap, mcc, t, sens, spec)
    return best


def grouped_cv(df, feature_cols, label_col, model_name):
    ok = (df[feature_cols].notna().all(axis=1) & df[label_col].notna()).to_numpy()
    X = df.loc[ok, feature_cols].to_numpy(dtype=float)
    y = df.loc[ok, label_col].to_numpy(dtype=int)
    g = df.loc[ok, "comp_group_id"].astype(str).to_numpy()
    pooled_y, pooled_oof = [], []
    for s in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            assert set(g[tr]).isdisjoint(g[te]), "그룹 누수 발생"
            m = MODELS[model_name](s).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        pooled_y.append(y)
        pooled_oof.append(oof)
    return np.concatenate(pooled_y), np.concatenate(pooled_oof), int(ok.sum())


results = []
t0 = time.time()
for label_name, label_col in [("A", "binary_label_A"), ("B", "binary_label_B")]:
    for model_name in MODELS:
        for method_name, df in DATASETS.items():
            for conc_name, extra in [("baseline", []), ("+resolved_pct", ["resolved_pct"])]:
                feat_cols = PROD_FEATURE_COLS + extra
                y, oof, n = grouped_cv(df, feat_cols, label_col, model_name)
                mcc05 = matthews_corrcoef(y, (oof >= 0.5).astype(int))
                auc = roc_auc_score(y, oof)
                gap, mcc_bal, thr, sens, spec = best_balanced(y, oof)
                results.append({
                    "label": label_name, "model": model_name, "method": method_name, "features": conc_name,
                    "n": n, "mcc_at_0.5": mcc05, "roc_auc": auc,
                    "balanced_thr": thr, "mcc_balanced": mcc_bal, "sens_balanced": sens, "spec_balanced": spec,
                })
                print(f"완료 ({time.time()-t0:.0f}s): {label_name}/{model_name}/{method_name}/{conc_name} "
                      f"n={n} mcc@0.5={mcc05:.3f} auc={auc:.3f} mcc_bal={mcc_bal:.3f}")

table = pd.DataFrame(results)
out_path = rf"{BASE}\skin_aggregation_method_comparison_v2.csv"
table.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n저장:", out_path)

pd.set_option("display.width", 220)
for conc_name in ["baseline", "+resolved_pct"]:
    print(f"\n========== features={conc_name} (mcc_balanced 기준) ==========")
    sub = table[table.features == conc_name]
    piv = sub.pivot_table(index=["label", "model"], columns="method", values="mcc_balanced")
    piv["delta(top-wmean)"] = piv["top"] - piv["wmean"]
    print(piv.round(4).to_string())

print(f"\n총 소요시간: {time.time()-t0:.0f}초")
