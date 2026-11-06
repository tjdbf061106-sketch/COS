# -*- coding: utf-8 -*-
"""1) N066(self-validated, 실제 관찰 아님) 제거
   2) skin_보강후보_v4_신규.xlsx의 추가후보 2건(MAQUAT 5.5-M, BEQ442-8) 집계해서 추가"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit import DataStructs

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pct_parser import parse_pct_text  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skin irritation/

# ---------------------------------------------------------------- 1. N066 제거
vec = pd.read_csv(f"{BASE}\\skin_vectorized.csv", low_memory=False)
before = len(vec)
vec = vec[vec["formulation_id"] != "AUG_v3:N066"].reset_index(drop=True)
print(f"N066 제거: {before} -> {len(vec)}행")

# ---------------------------------------------------------------- 2. v4 후보 집계
_FULL_DESC_LIST = Descriptors._descList
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def get_largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
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
            val = func(mol)
            if not np.isfinite(val):
                val = np.nan
        except Exception:
            val = np.nan
        out[f"desc_{name}"] = val
    return out


def compute_morgan_fp(mol):
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return {f"fp_w_{i}": int(v) for i, v in enumerate(arr)}


COMMON8_COLS = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "AromaticProportion", "ESOL_LogS"]
desc_full = pd.read_csv(f"{BASE}\\casrn_descriptors_full.csv").set_index("casrn")
DESC217_COLS = [c for c in desc_full.columns if c.startswith("desc_")]
FP_COLS = [c for c in desc_full.columns if c.startswith("fp_w_")]
ALL_FEATURE_COLS = COMMON8_COLS + DESC217_COLS + FP_COLS


def weighted_mean_nan_aware(pct, mat):
    result = np.full(mat.shape[1], np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = ~np.isnan(col)
        if not mask.any():
            continue
        w = pct[mask]
        if w.sum() == 0:
            w = np.ones_like(w)
        result[j] = (w * col[mask]).sum() / w.sum()
    return result


def weighted_mean_plain(pct, mat):
    w = pct.copy()
    if w.sum() == 0:
        w = np.ones_like(w)
    return (w[:, None] * mat).sum(axis=0) / w.sum()


def get_pct(row):
    if pd.notna(row.get("정확값")):
        return float(row["정확값"])
    if pd.notna(row.get("하한")) and pd.notna(row.get("상한")):
        return (float(row["하한"]) + float(row["상한"])) / 2
    parsed = parse_pct_text(row.get("함량_원문"))
    if parsed is not None:
        return parsed
    return np.nan


candidates = pd.read_excel(f"{BASE}\\skin_보강후보_v4_신규.xlsx", sheet_name="추가후보")
components = pd.read_excel(f"{BASE}\\skin_보강후보_v4_신규.xlsx", sheet_name="성분함량")
components["key"] = components["CAS_원문"].fillna(components["성분명_원문"])
components["has_structure"] = components["key"].isin(desc_full.index)

agg_rows = []
skipped = []
for _, cand in candidates.iterrows():
    key = cand["후보키"]
    comp = components[components["후보키"] == key].copy()
    g_resolved = comp[comp["has_structure"]]
    g_unresolved = comp[~comp["has_structure"]]
    if len(g_resolved) == 0:
        skipped.append(key)
        continue

    pct_raw = g_resolved.apply(get_pct, axis=1).values
    resolved_pct = np.nansum(pct_raw)
    unresolved_pct = g_unresolved.apply(get_pct, axis=1).sum() if len(g_unresolved) else 0.0
    pct = np.where(np.isnan(pct_raw), 1.0, pct_raw)
    if pct.sum() == 0:
        pct = np.ones_like(pct)

    common_mat = desc_full.loc[g_resolved["key"], COMMON8_COLS].values
    desc_mat = desc_full.loc[g_resolved["key"], DESC217_COLS].values
    fp_mat = desc_full.loc[g_resolved["key"], FP_COLS].values

    row = {"formulation_id": f"AUG_{key}"}
    for col, val in zip(COMMON8_COLS, weighted_mean_plain(pct, common_mat)):
        row[col] = val
    for col, val in zip(DESC217_COLS, weighted_mean_nan_aware(pct, desc_mat)):
        row[col] = val
    for col, val in zip(FP_COLS, weighted_mean_plain(pct, fp_mat)):
        row[col] = val

    row["comp_count"] = len(comp)
    row["resolved_pct"] = resolved_pct
    row["unresolved_structure_pct"] = unresolved_pct
    row["is_single_component"] = len(comp) == 1
    row["binary_label"] = 1 if cand["이진라벨_참고"] == 1 else 0
    comp_sig = "|".join(sorted(f"{c}:{p:.1f}" for c, p in zip(g_resolved["key"], pct_raw) if not np.isnan(p)))
    row["comp_group_id"] = "AUG_" + hashlib.md5(comp_sig.encode()).hexdigest()[:12]
    row["source"] = "skin_보강후보_v4_신규"
    agg_rows.append(row)

print(f"\nv4 후보 {len(candidates)}건 중 {len(agg_rows)}건 구조 해결, {len(skipped)}건 제외: {skipped}")

v4_df = pd.DataFrame(agg_rows)
if len(v4_df):
    print(v4_df[["formulation_id", "resolved_pct", "binary_label"]].to_string())

# ---------------------------------------------------------------- 3. 병합 저장
for c in vec.columns:
    if c not in v4_df.columns:
        v4_df[c] = pd.NA
v4_df = v4_df[vec.columns]
final = pd.concat([vec, v4_df], ignore_index=True)
print(f"\n최종: {len(vec)} - N066 + v4신규{len(v4_df)}건 = {len(final)}행")
print("binary_label 분포:", final["binary_label"].value_counts().to_dict())

final.to_csv(f"{BASE}\\skin_vectorized.csv", index=False, encoding="utf-8-sig")
print("저장: skin_vectorized.csv (최종 production)")
