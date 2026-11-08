# -*- coding: utf-8 -*-
"""공식 재검증(rf_production_recheck.py와 동일 방법론: nz20 fold-internal filter, 균형점 threshold,
comp_group_id group 5-fold)을 (RF,ExtraTrees) x (wmean,top) x (농도유무) x (라벨A,B) 16그리드로 확장.

rf_production_recheck.py와 다른 점:
  - 모델 2종(RF/ExtraTrees), 둘 다 class_weight="balanced"로 통일(원본 RF 설정 그대로, ET도 동일 적용)
  - 집계방식 2종(wmean=production 그대로 / top=제형별 최우세성분 단독)
  - 농도피처(resolved_pct) 유무 축 추가
  - N_REPS=15(원본은 20 — 16칸 전체를 감당 가능한 시간 안에 끝내기 위한 절충, 보고서에 명시할 것)
  - 결과를 셀 단위로 즉시 CSV에 append해 중단되어도 이어서 돌릴 수 있음(--label A 또는 --label B로 분할 실행)

사용법:
    python rf_et_aggregation_concentration_grid.py A
    python rf_et_aggregation_concentration_grid.py B
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\UserK\학술제\skin irritation"
SRC_B5 = r"C:\Users\UserK\OneDrive - 덕성여자대학교\문서\카카오톡 받은 파일\skin_irritation_최종후보.xlsx"
N_REPS = 15
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)
OUT_PATH = Path(BASE) / "skin_official_grid_recheck.csv"

LABEL_ARG = sys.argv[1] if len(sys.argv) > 1 else "A"
assert LABEL_ARG in ("A", "B")

vec = pd.read_csv(rf"{BASE}\skin_vectorized.csv", low_memory=False)
desc_full = pd.read_csv(rf"{BASE}\casrn_descriptors_full.csv").set_index("casrn")
PROD_FEATURE_COLS = ["AromaticProportion"] + [c for c in desc_full.columns if c.startswith(("desc_", "fp_w_"))]

# ------------------------------------------------------------------ 원자료 재구성 (top 계산용, 이전 스크립트와 동일)
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

META_COLS = ["comp_group_id", "resolved_pct", "binary_label_A", "binary_label_B"]
meta = vec.set_index("formulation_id")[META_COLS]
wmean_full = meta.join(vec.set_index("formulation_id")[PROD_FEATURE_COLS])
top_full = meta.join(top_df)
DATASETS = {"wmean": wmean_full, "top": top_full}

print(f"원자료/top 재계산 완료. wmean n={len(wmean_full)}, top n={top_full[PROD_FEATURE_COLS[0]].notna().sum()}")


# ------------------------------------------------------------------ nz20 변환(bharness.py와 동일 로직, torch 의존성 없이 재구현)
class NonZeroFilter:
    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.keep_ = (X != 0).sum(axis=0) >= self.min_count
        if self.keep_.sum() == 0:
            self.keep_[:] = True
        return self

    def transform(self, X):
        return X[:, self.keep_]


def nz20_transform_fit(X):
    nz = NonZeroFilter(20).fit(X)
    sc = StandardScaler().fit(nz.transform(X))
    return nz, sc


MODELS = {
    "RandomForest": lambda seed: RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
    "ExtraTrees": lambda seed: ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
}


def balance_point(y, proba):
    best = None
    for t in GRID:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        gap = abs(sens - spec)
        mcc = matthews_corrcoef(y, pred)
        if best is None or (round(gap, 4), -mcc) < (round(best[0], 4), -best[1]):
            best = (gap, mcc, t, sens, spec)
    return best


def run_rep(X, y, g, model_name, rep):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, g):
        assert set(g[tr]).isdisjoint(g[te]), "그룹 누수 발생"
        nz, sc = nz20_transform_fit(X[tr])
        Xtr = sc.transform(nz.transform(X[tr]))
        Xte = sc.transform(nz.transform(X[te]))
        m = MODELS[model_name](rep).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def done_cells():
    if not OUT_PATH.exists():
        return set()
    done = pd.read_csv(OUT_PATH)
    return set(zip(done.label, done.model, done.method, done.features))


label_col = f"binary_label_{LABEL_ARG}"
t0 = time.time()
skip = done_cells()

for model_name in MODELS:
    for method_name, df in DATASETS.items():
        for conc_name, extra in [("baseline", []), ("+resolved_pct", ["resolved_pct"])]:
            key = (LABEL_ARG, model_name, method_name, conc_name)
            if key in skip:
                print(f"건너뜀(이미 완료): {key}")
                continue
            feat_cols = PROD_FEATURE_COLS + extra
            ok = (df[feat_cols].notna().all(axis=1) & df[label_col].notna()).to_numpy()
            X = df.loc[ok, feat_cols].to_numpy(dtype=float)
            y = df.loc[ok, label_col].to_numpy(dtype=int)
            g = df.loc[ok, "comp_group_id"].astype(str).to_numpy()

            pooled_y, pooled_oof = [], []
            rows = []
            for rep in range(N_REPS):
                oof = run_rep(X, y, g, model_name, rep)
                bp = balance_point(y, oof)
                rows.append(bp)
                pooled_y.append(y)
                pooled_oof.append(oof)
            gaps = np.array([r[0] for r in rows]); mccs = np.array([r[1] for r in rows])
            senss = np.array([r[3] for r in rows]); specs = np.array([r[4] for r in rows])
            thrs = np.array([r[2] for r in rows])
            yy, oo = np.concatenate(pooled_y), np.concatenate(pooled_oof)
            auc_pooled = roc_auc_score(yy, oo)

            result = {
                "label": LABEL_ARG, "model": model_name, "method": method_name, "features": conc_name,
                "n": int(ok.sum()), "n_reps": N_REPS,
                "mcc_mean": mccs.mean(), "mcc_std": mccs.std(),
                "sens_mean": senss.mean(), "spec_mean": specs.mean(), "gap_mean": gaps.mean(),
                "thr_mean": thrs.mean(), "roc_auc_pooled": auc_pooled,
            }
            header = not OUT_PATH.exists()
            pd.DataFrame([result]).to_csv(OUT_PATH, mode="a", header=header, index=False, encoding="utf-8-sig")
            print(f"완료 ({time.time()-t0:.0f}s): {key} n={result['n']} "
                  f"MCC={result['mcc_mean']:.4f}±{result['mcc_std']:.4f} AUC={auc_pooled:.4f}")

print(f"\n총 소요시간: {time.time()-t0:.0f}초. 결과: {OUT_PATH}")
