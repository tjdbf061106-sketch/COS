# -*- coding: utf-8 -*-
"""
skin 자극성 — 입력(feature) 비교 + threshold 곡선 + 짝지은(paired) 유의성 검정

핵심: 평균끼리 비교하면 작은 차이가 변동폭에 묻힌다.
     같은 seed·같은 분할로 짝지어 비교해야 차이가 보인다.

사용법:
    python skin_compare.py                 # 기본 20 seed
    python skin_compare.py 30              # seed 30개

필요 파일 (BASE 폴더 안):
    skin_filtered.csv
    casrn_smiles_cache.csv
    casrn_descriptors_full.csv
    skin_vectorized.csv        (라벨·comp_group_id용)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix

# ─────────────────────────────────────────────
# 여기만 본인 경로로 바꾸세요
BASE = r"C:\Users\UserK\학술제\skin irritation"
# ─────────────────────────────────────────────

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SPLITS = 5
AGG = "wmean"          # wmean | top | max | min
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]


# ── 1. 데이터 준비 ────────────────────────────
def load():
    p = lambda name: os.path.join(BASE, name)
    f = pd.read_csv(p("skin_filtered.csv"), low_memory=False)
    cache = pd.read_csv(p("casrn_smiles_cache.csv"))
    vec = pd.read_csv(p("skin_vectorized.csv"), low_memory=False)
    desc = pd.read_csv(p("casrn_descriptors_full.csv")).set_index("casrn")

    sm = cache.dropna(subset=["smiles"]).set_index("casrn")["smiles"].to_dict()
    comp = (f.dropna(subset=["casrn"])
              .drop_duplicates(["formulation_id", "casrn"], keep="first")
              .assign(smiles=lambda d: d["casrn"].map(sm),
                      pct=lambda d: d["percent_active_ingredient"])
              [["formulation_id", "casrn", "pct", "smiles"]])

    labels = vec[["formulation_id", "binary_label", "comp_group_id", "comp_count"]]
    return comp, labels, desc


def aggregate(emb, comp, method=AGG):
    """성분별 벡터를 제형 단위로 접는다."""
    rows, ids = [], []
    for fid, g in comp.groupby("formulation_id"):
        g = g[g.casrn.isin(emb.index)]
        if len(g) == 0:
            continue
        d = emb.loc[g.casrn].values.astype(float)
        pct = g.pct.fillna(1.0).values.astype(float)
        if pct.sum() == 0:
            pct = np.ones_like(pct)
        if method == "wmean":
            v = (pct[:, None] * d).sum(0) / pct.sum()
        elif method == "top":
            v = d[int(np.argmax(pct))]
        elif method == "max":
            v = d.max(0)
        elif method == "min":
            v = d.min(0)
        else:
            raise ValueError(method)
        rows.append(v); ids.append(fid)
    out = pd.DataFrame(rows, index=ids)
    out.columns = [f"emb_{i}" for i in range(out.shape[1])]
    return out.reset_index().rename(columns={"index": "formulation_id"})


def clean(X):
    """결측 있는 열, 상수 열 제거."""
    X = X.loc[:, X.notna().all()]
    return X.loc[:, X.nunique() > 1]


# ── 2. 평가 ──────────────────────────────────
def oof_probs(feat, labels, seed):
    """seed 1개에 대한 out-of-fold 예측확률. 그룹 분할로 누수 차단."""
    df = feat.merge(labels[["formulation_id", "binary_label", "comp_group_id"]],
                    on="formulation_id")
    fc = [c for c in df.columns if c.startswith("emb_")]
    ok = df[fc].notna().all(axis=1).values
    X = df.loc[ok, fc].values
    y = df.loc[ok, "binary_label"].values.astype(int)
    g = df.loc[ok, "comp_group_id"].astype(str).values

    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, g):
        assert set(g[tr]).isdisjoint(g[te]), "그룹 누수"
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y, len(set(g))


def metrics_at(oof, y, thr):
    p = (oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                mcc=matthews_corrcoef(y, p),
                roc_auc=roc_auc_score(y, oof),
                pr_auc=average_precision_score(y, oof))


# ── 3. 실행 ──────────────────────────────────
def main():
    comp, labels, desc = load()
    fp   = [c for c in desc.columns if c.startswith("fp")]
    dsc  = [c for c in desc.columns if c.startswith("desc_")]
    base = [c for c in desc.columns if c not in fp + dsc]

    # nz20: 전체 데이터에서 nonzero<20인 열 제거 (기존 채택안 재현)
    allcols = desc[fp + dsc + base]
    nz = (allcols.fillna(0) != 0).sum()
    nz20 = allcols.loc[:, nz >= 20]

    sets = {
        "기술자+물성": desc[dsc + base],
        "핑거프린트":  desc[fp],
        "nz20":       nz20,
        "전체":        desc[fp + dsc + base],
    }

    # seed별 OOF를 먼저 전부 계산 (threshold는 나중에 쓸어보면 됨)
    store = {}
    for nm, X in sets.items():
        Xc = clean(X)
        feat = aggregate(Xc, comp)
        store[nm] = dict(dims=Xc.shape[1], oof=[])
        for s in range(N_SEEDS):
            oof, y, ng = oof_probs(feat, labels, s)
            store[nm]["oof"].append(oof)
        store[nm]["y"] = y
        store[nm]["n_groups"] = ng
        print(f"[완료] {nm}  {Xc.shape[1]}차원")

    y = store["전체"]["y"]
    print(f"\nn={len(y)}  양성={int(y.sum())}  그룹={store['전체']['n_groups']}  "
          f"seeds={N_SEEDS}  집계={AGG}  모델=RandomForest(500, balanced)")

    # (1) threshold 0.5 기준 요약
    print("\n=== (1) threshold 0.5 ===")
    rows = []
    for nm, d in store.items():
        M = pd.DataFrame([metrics_at(o, d["y"], 0.5) for o in d["oof"]])
        rows.append(dict(feature=nm, 차원=d["dims"],
                         **{k: f"{M[k].mean():.3f}±{M[k].std():.3f}"
                            for k in ["sensitivity", "specificity", "gap", "mcc", "roc_auc"]}))
    t1 = pd.DataFrame(rows)
    print(t1.to_string(index=False))

    # (2) 짝지은 비교 — 같은 seed끼리. 이게 핵심.
    print("\n=== (2) 짝지은 비교 (같은 seed·같은 분할) — 기준: 전체 ===")
    print("    ROC-AUC는 threshold와 무관하므로 입력 비교에 가장 적합")
    ref = np.array([roc_auc_score(y, o) for o in store["전체"]["oof"]])
    rows = []
    for nm, d in store.items():
        if nm == "전체":
            continue
        cur = np.array([roc_auc_score(y, o) for o in d["oof"]])
        diff = cur - ref
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(dict(feature=nm,
                         평균차=f"{diff.mean():+.4f}",
                         t=f"{diff.mean()/se:+.2f}" if se > 0 else "—",
                         우세=f"{int((diff > 0).sum())}/{len(diff)}"))
    print(pd.DataFrame(rows).to_string(index=False))

    # (3) threshold 곡선 — 가장 좋은 입력 기준
    best = max(store, key=lambda k: np.mean([roc_auc_score(y, o) for o in store[k]["oof"]]))
    print(f"\n=== (3) threshold 곡선 — {best} ({store[best]['dims']}차원) ===")
    auc = np.mean([roc_auc_score(y, o) for o in store[best]["oof"]])
    print(f"    ROC-AUC = {auc:.3f} (threshold 무관)")
    rows = []
    for t in THRESHOLDS:
        M = pd.DataFrame([metrics_at(o, y, t) for o in store[best]["oof"]])
        rows.append(dict(threshold=t,
                         **{k: f"{M[k].mean():.3f}±{M[k].std():.3f}"
                            for k in ["sensitivity", "specificity", "mcc"]},
                         격차=f"{M['gap'].mean():.3f}"))
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False))

    t1.to_csv("result_thr05.csv", index=False, encoding="utf-8-sig")
    t3.to_csv("result_threshold_curve.csv", index=False, encoding="utf-8-sig")
    print("\n저장: result_thr05.csv, result_threshold_curve.csv")


if __name__ == "__main__":
    main()
