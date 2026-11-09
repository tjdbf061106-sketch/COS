# -*- coding: utf-8 -*-
"""wmean vs max 확정 비교 — outer 5-fold x inner 3-fold nested CV.

팀원 제안 프로토콜(2026-11-09, `세팅/wmean_vs_max_팀원교정반영_최종합의_20261109.pdf` §4) 그대로 구현:
  1. 동일 표본·동일 분할: wmean·max 둘 다 계산 가능한 동일 649건만 사용. 라벨B·RandomForest·
     +resolved_pct 고정(팀의 실제 목표 조건).
  2. 임계값 분리: outer 5-fold(최종 채점용, 완전히 봉인) x inner 3-fold(threshold+모델학습 전용).
     inner 안에서만 만든 out-of-fold 예측으로 balance_point threshold를 고르고, 그 threshold를
     outer test(그 threshold 선택에 한 번도 관여 안 한 데이터)에 적용해 채점.
     outer/inner 둘 다 comp_group_id 기준으로 그룹 분리(group leak 금지, assert로 검증).
  3. 전체 + 다성분 하위집단: pooled 결과를 '전체'와 '다성분 제형만'으로 나눠서 둘 다 보고.
  4. 사전 결정 기준(결과를 보기 전에 고정, 아래 DECISION_RULE 참고):
     - ΔMCC = mean(max) - mean(wmean) 를 20회 반복 각각의 pooled MCC로 쌍대응(paired) 비교.
     - "max 전환 검토": paired 평균 ΔMCC > 0.02 AND 양측 paired t-test p<0.05(Δ>0 방향) AND
       max의 Sens-Spec 격차(gap)가 wmean의 격차보다 0.05 이상 나빠지지 않음 AND
       전체표본과 다성분 하위집단에서 ΔMCC 방향이 서로 모순되지 않음(둘 다 >=0 또는 판단보류 아님).
     - 위 조건을 전부 만족하지 못하면: "wmean 유지".
  5. 20회 반복 = 같은 649건을 20번 다시 나눈 것일 뿐, 새 표본 20개가 아님 — 결과를 "영구 확정"으로
     표현하지 않음(팀원 지적 반영).

wmean과 max는 같은 rep(seed)에 대해 **동일한 outer/inner 분할**을 사용해 쌍대응(paired) 비교가
가능하게 함 — 분할 자체의 운(bad split)이 비교에 섞이지 않도록.

사용법:
    python nested_cv_wmean_vs_max.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\UserK\학술제\skin irritation"
SRC_B5 = r"C:\Users\UserK\OneDrive - 덕성여자대학교\문서\카카오톡 받은 파일\skin_irritation_최종후보.xlsx"
N_REPS = 20
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)
OUT_PATH = Path(BASE) / "skin_wmean_vs_max_nested_cv.csv"
LABEL_COL = "binary_label_B"  # 팀의 실제 목표 라벨

vec = pd.read_csv(rf"{BASE}\skin_vectorized.csv", low_memory=False)
desc_full = pd.read_csv(rf"{BASE}\casrn_descriptors_full.csv").set_index("casrn")
PROD_FEATURE_COLS = ["AromaticProportion"] + [c for c in desc_full.columns if c.startswith(("desc_", "fp_w_"))]

# ------------------------------------------------------------------ 원자료 재구성 (max/다성분 플래그 계산용, 기존 32그리드 스크립트와 동일)
filt = pd.read_csv(rf"{BASE}\skin_filtered.csv")
known = filt.dropna(subset=["casrn"]).copy()
known = known.drop_duplicates(subset=["formulation_id", "casrn"], keep="first")
known["has_descriptor"] = known["casrn"].isin(desc_full.index)
known = known.rename(columns={"casrn": "key", "percent_active_ingredient": "pct"})
known_unified = known[["formulation_id", "key", "pct", "has_descriptor"]].copy()

aug_meta = vec.loc[vec["formulation_id"].astype(str).str.startswith("AUG_"), ["formulation_id"]].copy()
aug_meta["formulation_id"] = aug_meta["formulation_id"].str.replace("AUG_", "", regex=False)

NUMBER = r"(?:\d+\.\d+|\.\d+|\d+)"
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
    return cleaned


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

max_rows = {}
n_resolved_components = {}
for fid, grp in raw_all.groupby("formulation_id"):
    g_resolved = grp[grp["has_descriptor"]]
    n_resolved_components[fid] = len(g_resolved)
    if len(g_resolved) == 0:
        continue
    sub = desc_full.loc[g_resolved["key"].values, PROD_FEATURE_COLS]
    max_rows[fid] = sub.max(axis=0).to_dict()


def _to_formulation_df(rows_dict):
    d = pd.DataFrame.from_dict(rows_dict, orient="index")
    d.index.name = "fid_raw"
    d = d.reset_index()
    is_aug_ = d["fid_raw"].isin(aug_meta["formulation_id"])
    d["formulation_id"] = d["fid_raw"]
    d.loc[is_aug_, "formulation_id"] = "AUG_" + d.loc[is_aug_, "fid_raw"]
    return d.drop(columns=["fid_raw"]).set_index("formulation_id")


max_df = _to_formulation_df(max_rows)

n_resolved_df = pd.Series(n_resolved_components, name="n_resolved").to_frame()
n_resolved_df.index.name = "fid_raw"
n_resolved_df = n_resolved_df.reset_index()
is_aug_r = n_resolved_df["fid_raw"].isin(aug_meta["formulation_id"])
n_resolved_df["formulation_id"] = n_resolved_df["fid_raw"]
n_resolved_df.loc[is_aug_r, "formulation_id"] = "AUG_" + n_resolved_df.loc[is_aug_r, "fid_raw"]
n_resolved_df = n_resolved_df.drop(columns=["fid_raw"]).set_index("formulation_id")

META_COLS = ["comp_group_id", "resolved_pct", LABEL_COL]
meta = vec.set_index("formulation_id")[META_COLS]
wmean_full = meta.join(vec.set_index("formulation_id")[PROD_FEATURE_COLS])
max_full = meta.join(max_df)

# ------------------------------------------------------------------ 동일 649건 매칭 (wmean과 max 둘 다 계산 가능한 formulation만)
feat_cols = PROD_FEATURE_COLS + ["resolved_pct"]
ok_wmean = wmean_full[feat_cols].notna().all(axis=1) & wmean_full[LABEL_COL].notna()
ok_max = max_full[feat_cols].notna().all(axis=1) & max_full[LABEL_COL].notna()
common_idx = wmean_full.index[ok_wmean & ok_max]
print(f"동일 매칭 표본: {len(common_idx)}건 (wmean 단독 가능 {ok_wmean.sum()}, max 단독 가능 {ok_max.sum()})")

multi_flag = n_resolved_df.reindex(common_idx)["n_resolved"].fillna(1) > 1
print(f"  다성분 하위집단: {int(multi_flag.sum())}건 / 단일성분: {int((~multi_flag).sum())}건")

Xw = wmean_full.loc[common_idx, feat_cols].to_numpy(dtype=float)
Xm = max_full.loc[common_idx, feat_cols].to_numpy(dtype=float)
y = wmean_full.loc[common_idx, LABEL_COL].to_numpy(dtype=int)
g = wmean_full.loc[common_idx, "comp_group_id"].astype(str).to_numpy()
multi_mask = multi_flag.to_numpy()
DATASETS = {"wmean": Xw, "max": Xm}


# ------------------------------------------------------------------ nz20 변환
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


def nz20_fit_transform(Xtr, Xte):
    nz = NonZeroFilter(20).fit(Xtr)
    sc = StandardScaler().fit(nz.transform(Xtr))
    return sc.transform(nz.transform(Xtr)), sc.transform(nz.transform(Xte))


def make_model(seed):
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)


def balance_point(y_true, proba):
    best = None
    for t in GRID:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        gap = abs(sens - spec)
        mcc = matthews_corrcoef(y_true, pred)
        if best is None or (round(gap, 4), -mcc) < (round(best[0], 4), -best[1]):
            best = (gap, mcc, t, sens, spec)
    return best[2]  # threshold만 반환 (성능은 outer test에서 따로 계산)


def run_nested_rep(X, y, g, rep):
    """outer 5-fold(봉인) x inner 3-fold(threshold 전용). outer별 (y_true, proba, threshold) 반환."""
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    outer_results = []
    for otr, ote in outer_cv.split(X, y, g):
        assert set(g[otr]).isdisjoint(g[ote]), "outer 그룹 누수"
        Xtr_outer, ytr_outer, gtr_outer = X[otr], y[otr], g[otr]

        # inner 3-fold: threshold 선택 전용 oof (outer test는 절대 관여 안 함)
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=rep * 100 + 1)
        inner_oof = np.zeros(len(ytr_outer))
        for itr, ite in inner_cv.split(Xtr_outer, ytr_outer, gtr_outer):
            assert set(gtr_outer[itr]).isdisjoint(gtr_outer[ite]), "inner 그룹 누수"
            Xitr, Xite = nz20_fit_transform(Xtr_outer[itr], Xtr_outer[ite])
            m_inner = make_model(rep).fit(Xitr, ytr_outer[itr])
            inner_oof[ite] = m_inner.predict_proba(Xite)[:, 1]
        thr = balance_point(ytr_outer, inner_oof)

        # outer test용 최종모델: train_outer 전체로 학습(= nz20/scaler도 train_outer 전체로 fit)
        Xtr_final, Xte_final = nz20_fit_transform(Xtr_outer, X[ote])
        m_final = make_model(rep).fit(Xtr_final, ytr_outer)
        proba_outer = m_final.predict_proba(Xte_final)[:, 1]

        outer_results.append((y[ote], proba_outer, thr, ote))
    return outer_results


def score_pooled(outer_results, mask=None):
    """outer_results의 (y_true,proba,thr,idx)를 모아 threshold별로 예측 적용 후 pooled 지표 계산.
    mask가 주어지면 전체 common_idx 기준 불린마스크로 하위집합만 필터링."""
    ys, preds, probas = [], [], []
    for y_true, proba, thr, idx in outer_results:
        sel = np.ones(len(idx), dtype=bool) if mask is None else mask[idx]
        if sel.sum() == 0:
            continue
        ys.append(y_true[sel])
        probas.append(proba[sel])
        preds.append((proba[sel] >= thr).astype(int))
    if not ys:
        return None
    yy, pp, prb = np.concatenate(ys), np.concatenate(preds), np.concatenate(probas)
    tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    mcc = matthews_corrcoef(yy, pp)
    try:
        auc = roc_auc_score(yy, prb)
    except ValueError:
        auc = np.nan
    return {"n": len(yy), "mcc": mcc, "sens": sens, "spec": spec, "gap": abs(sens - spec), "auc": auc}


# ------------------------------------------------------------------ 실행 (rep별 즉시 append, 중단 시 이어서 가능)
def done_reps():
    if not OUT_PATH.exists():
        return set()
    done = pd.read_csv(OUT_PATH)
    return set(zip(done.method, done.subset, done.rep))


t0 = time.time()
skip = done_reps()
for method_name, X in DATASETS.items():
    for rep in range(N_REPS):
        key_overall = (method_name, "overall", rep)
        if key_overall in skip:
            print(f"건너뜀(이미 완료): {method_name} rep={rep}")
            continue
        outer_results = run_nested_rep(X, y, g, rep)
        rows = []
        for subset_name, mask in [("overall", None), ("multi_component", multi_mask)]:
            sc = score_pooled(outer_results, mask)
            if sc is None:
                continue
            rows.append({"method": method_name, "subset": subset_name, "rep": rep, **sc})
        header = not OUT_PATH.exists()
        pd.DataFrame(rows).to_csv(OUT_PATH, mode="a", header=header, index=False, encoding="utf-8-sig")
        print(f"완료 ({time.time()-t0:.0f}s): {method_name} rep={rep} "
              f"overall_MCC={rows[0]['mcc']:.4f}")

print(f"\n총 소요시간: {time.time()-t0:.0f}초. 결과: {OUT_PATH}")
