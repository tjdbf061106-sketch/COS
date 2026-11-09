# -*- coding: utf-8 -*-
"""Lechuga 2025 외부검증 — 1단계(옵션 A).
frozen skin_rf_checkpoint.joblib로 추론만 수행(재학습 없음).
해결된 10종 성분으로 완전해결된 41행(single 38 + binary_mixture 3)을 production과 동일한
wmean(+resolved_pct) 벡터화로 만들어 예측 vs Zein 기반 binary label 비교.

주의: Zein Number(in vitro 대체시험, ECVAM cutoff 200) 기반 라벨은 skin_rf가 학습한
엔드포인트(Draize 동물실험 기반 EPA Classification)와 다름 — "모델 성능 검증"이 아니라
"구조 겹침 기반 참고 비교"로만 취급(flea_tick_shampoo와 동일 원칙).
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

BASE = r"C:\Users\UserK\학술제"
LECHUGA_CSV = rf"{BASE}\cosmetic_validation\lechuga2025_surfactant_mixture_zein.csv"
CACHE_CSV = rf"{BASE}\cosmetic_validation\lechuga_smiles_cache.csv"
CHECKPOINT = rf"{BASE}\modeling\skin_rf\skin_rf_checkpoint.joblib"
OUT_CSV = rf"{BASE}\cosmetic_validation\lechuga_external_validation_results.csv"

_FULL_DESC_LIST = Descriptors._descList
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def get_largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def compute_common8(mol):
    n_heavy = mol.GetNumHeavyAtoms()
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    arom_prop = n_aromatic / n_heavy if n_heavy > 0 else 0.0
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    esol_logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * arom_prop
    return {"MolWt": mw, "LogP": logp, "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol), "HBA": Descriptors.NumHAcceptors(mol),
            "RotatableBonds": rotb, "AromaticProportion": arom_prop, "ESOL_LogS": esol_logs}


def compute_full_descriptors(mol):
    out = {}
    for name, func in _FULL_DESC_LIST:
        try:
            v = func(mol)
            if not np.isfinite(v):
                v = np.nan
        except Exception:
            v = np.nan
        out[f"desc_{name}"] = v
    return out


def compute_morgan_fp(mol):
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return {f"fp_w_{i}": int(v) for i, v in enumerate(arr)}


# ---- 1) 해결된 10종의 descriptor/FP 계산 (skin_feature_expand_full.ipynb와 동일 로직) ----
cache = pd.read_csv(CACHE_CSV)
resolved_cache = cache[cache["status"] == "found"].copy()

comp_rows = []
for _, r in resolved_cache.iterrows():
    mol_orig = Chem.MolFromSmiles(r["smiles"])
    if mol_orig is None:
        continue
    mol_stripped = get_largest_fragment(mol_orig)
    row = {"inci_name": r["inci_name"]}
    row.update(compute_common8(mol_stripped))
    row.update(compute_full_descriptors(mol_stripped))
    row.update(compute_morgan_fp(mol_orig))
    comp_rows.append(row)

comp_desc = pd.DataFrame(comp_rows).set_index("inci_name")
DESC217_COLS = [c for c in comp_desc.columns if c.startswith("desc_")]
FP_COLS = [c for c in comp_desc.columns if c.startswith("fp_w_")]
print(f"성분 descriptor 계산 완료: {len(comp_desc)}종 x {1 + len(DESC217_COLS) + len(FP_COLS)}차원")

# ---- 2) 41행(완전해결) 제형 벡터화 ----
df = pd.read_csv(LECHUGA_CSV)
resolved_names = set(resolved_cache["inci_name"])


def row_components(r):
    """(name, share_pct) 리스트. proportion('1' 또는 '1-1','1-2' 등)으로 concentration_pct를 분배."""
    conc = r["concentration_pct"]
    if r["mixture_type"] == "single":
        return [(r["product1"], conc)]
    parts = [int(x) for x in str(r["proportion"]).split("-")]
    total = sum(parts)
    return [(r["product1"], conc * parts[0] / total), (r["product2"], conc * parts[1] / total)]


def row_fully_resolved(r):
    comps = row_components(r)
    return all(name in resolved_names for name, _ in comps)


df["all_resolved"] = df.apply(row_fully_resolved, axis=1)
usable = df[df["all_resolved"]].copy().reset_index(drop=True)
print(f"완전해결 행: {len(usable)} / {len(df)}")

ALL_COLS = ["AromaticProportion"] + DESC217_COLS + FP_COLS

vec_rows = []
for i, r in usable.iterrows():
    comps = row_components(r)
    pct = np.array([share for _, share in comps], dtype=float)
    names = [name for name, _ in comps]
    mat = comp_desc.loc[names, ALL_COLS].values.astype(float)
    w = pct.copy()
    if w.sum() == 0:
        w = np.ones_like(w)
    # desc_* 일부 NaN 가능성 대비 NaN-aware, AromaticProportion/FP는 NaN 없음(단일구조 확보 성분이라 보장)
    out = np.full(mat.shape[1], np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        wj = w[mask]
        out[j] = (wj * col[mask]).sum() / wj.sum()
    row = dict(zip(ALL_COLS, out))
    row["resolved_pct"] = r["concentration_pct"]  # 100% 구조해결된 활성성분이 차지하는 비율 = 시험농도 자체
    vec_rows.append(row)

X = pd.DataFrame(vec_rows)

# ---- 3) frozen 체크포인트로 추론 (재학습 없음) ----
ck = joblib.load(CHECKPOINT)
feature_cols = ck["feature_cols"]
X = X[feature_cols]  # 체크포인트가 기대하는 컬럼 순서로 정렬
Xv = X.values
Xv = np.nan_to_num(Xv, nan=0.0)  # production train.py도 NaN->0 동일 처리(desc_* 일부 무기물 결측 패턴과 동일)
Xv = Xv[:, ck["nz_filter_keep"]]

proba = ck["model"].predict_proba(Xv)[:, 1]
pred = (proba >= ck["threshold"]).astype(int)

usable["pred_proba"] = proba
usable["pred_binary"] = pred
usable["true_binary"] = usable["binary_label(ZN>=200)"]
usable["match"] = usable["pred_binary"] == usable["true_binary"]

usable.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print()
print(f"threshold(배포값) = {ck['threshold']:.3f}")
print(f"전체 일치율: {usable['match'].sum()}/{len(usable)} ({usable['match'].mean()*100:.1f}%)")
for mt in ["single", "binary_mixture"]:
    sub = usable[usable["mixture_type"] == mt]
    if len(sub) == 0:
        continue
    print(f"  {mt}: {sub['match'].sum()}/{len(sub)} ({sub['match'].mean()*100:.1f}%)")

y_true = usable["true_binary"].values
y_pred = usable["pred_binary"].values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
print()
print(f"혼동행렬: TN={tn} FP={fp} FN={fn} TP={tp}")
try:
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"MCC: {mcc:.3f}")
except Exception as e:
    print("MCC 계산 불가:", e)
if len(set(y_true)) > 1:
    auc = roc_auc_score(y_true, proba)
    print(f"ROC-AUC: {auc:.3f}")
print(f"실제 라벨 분포: 양성(ZN>=200) {y_true.sum()} / 음성 {len(y_true)-y_true.sum()}")
print(f"예측 라벨 분포: 양성 {y_pred.sum()} / 음성 {len(y_pred)-y_pred.sum()}")
