# -*- coding: utf-8 -*-
"""skin_irritation_최종후보.xlsx(팀원 공유, Candidate_source=skin_보강후보_통합.xlsx 갱신판)에서
기존 production(652행)에 없던 신규 7건(v4:B5_003/006/009/018, v4:C011/012/016)을 집계해 추가.
구조(canonical_smiles)·정확 함량%는 V4성분검증 시트에 이미 해결되어 있어 재조회 없이 그대로 사용."""
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skin irritation/
SRC = r"C:\Users\UserK\OneDrive - 덕성여자대학교\문서\카카오톡 받은 파일\skin_irritation_최종후보.xlsx"

NEW_KEYS = ["v4:B5_003", "v4:B5_006", "v4:B5_009", "v4:B5_018", "v4:C011", "v4:C012", "v4:C016"]

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

verif = pd.read_excel(SRC, sheet_name="V4성분검증")
verif = verif[verif["후보키"].isin(NEW_KEYS)].copy()
used = verif[verif["used"] == True].copy()  # noqa: E712
used["key"] = used["resolved_cas"].fillna(used["성분명_원문"])

# 구조가 desc_full에 없는 신규 CAS는 canonical_smiles로 descriptor 새로 계산
missing_keys = used[~used["key"].isin(desc_full.index)][["key", "canonical_smiles"]].drop_duplicates("key")
print(f"신규 구조 계산 필요: {len(missing_keys)}건 -> {missing_keys['key'].tolist()}")

new_rows = []
for _, r in missing_keys.iterrows():
    mol_orig = Chem.MolFromSmiles(r["canonical_smiles"])
    if mol_orig is None:
        print(f"  [WARN] SMILES 파싱 실패: {r['key']} {r['canonical_smiles']}")
        continue
    mol_stripped = get_largest_fragment(mol_orig)
    row = {"casrn": r["key"]}
    row.update(compute_common8(mol_stripped))
    row.update(compute_full_descriptors(mol_stripped))
    row.update(compute_morgan_fp(mol_orig))
    new_rows.append(row)
new_desc = pd.DataFrame(new_rows).set_index("casrn") if new_rows else pd.DataFrame()

lookup = pd.concat([desc_full, new_desc]) if len(new_desc) else desc_full


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


cand_meta = pd.read_excel(SRC, sheet_name="후보목록")
cand_meta = cand_meta[cand_meta["후보키"].isin(NEW_KEYS)].set_index("후보키")

agg_rows = []
skipped = []
for key in NEW_KEYS:
    g_used = used[used["후보키"] == key]
    g_all = verif[verif["후보키"] == key]
    g_unused_resolved_missing = g_all[(g_all["used"] != True) & (~g_all["후보키"].isin([]))]  # noqa: E712
    if len(g_used) == 0:
        skipped.append(key)
        continue

    pct_raw = g_used["parsed_exact_pct"].astype(float).values
    resolved_pct = np.nansum(pct_raw)
    unresolved_pct = g_all.loc[g_all["used"] != True, "parsed_exact_pct"].astype(float).sum()  # noqa: E712
    pct = np.where(np.isnan(pct_raw), 1.0, pct_raw)
    if pct.sum() == 0:
        pct = np.ones_like(pct)

    common_mat = lookup.loc[g_used["key"], COMMON8_COLS].values
    desc_mat = lookup.loc[g_used["key"], DESC217_COLS].values
    fp_mat = lookup.loc[g_used["key"], FP_COLS].values

    row = {"formulation_id": f"AUG_{key}"}
    for col, val in zip(COMMON8_COLS, weighted_mean_plain(pct, common_mat)):
        row[col] = val
    for col, val in zip(DESC217_COLS, weighted_mean_nan_aware(pct, desc_mat)):
        row[col] = val
    for col, val in zip(FP_COLS, weighted_mean_plain(pct, fp_mat)):
        row[col] = val

    row["comp_count"] = len(g_all)
    row["resolved_pct"] = resolved_pct
    row["unresolved_structure_pct"] = unresolved_pct
    row["is_single_component"] = len(g_all) == 1
    row["binary_label"] = int(cand_meta.loc[key, "이진라벨_참고"])
    comp_sig = "|".join(sorted(f"{c}:{p:.2f}" for c, p in zip(g_used["key"], pct_raw) if not np.isnan(p)))
    row["comp_group_id"] = "AUG_" + hashlib.md5(comp_sig.encode()).hexdigest()[:12]
    row["source"] = "skin_irritation_최종후보_v4B5"
    agg_rows.append(row)

print(f"\nB5배치 {len(NEW_KEYS)}건 중 {len(agg_rows)}건 구조 해결, {len(skipped)}건 제외: {skipped}")
new_df = pd.DataFrame(agg_rows)
print(new_df[["formulation_id", "comp_count", "resolved_pct", "unresolved_structure_pct", "binary_label"]].to_string())

vec = pd.read_csv(f"{BASE}\\skin_vectorized.csv", low_memory=False)
backup_path = f"{BASE}\\skin_vectorized_pre_B5batch_{len(vec)}.csv"
vec.to_csv(backup_path, index=False, encoding="utf-8-sig")
print(f"\n백업 저장: {backup_path}")

for c in vec.columns:
    if c not in new_df.columns:
        new_df[c] = pd.NA
new_df = new_df[vec.columns]
final = pd.concat([vec, new_df], ignore_index=True)
print(f"\n최종: {len(vec)} + 신규{len(new_df)}건 = {len(final)}행")
print("binary_label 분포:", final["binary_label"].value_counts().to_dict())

final.to_csv(f"{BASE}\\skin_vectorized.csv", index=False, encoding="utf-8-sig")
print("저장: skin_vectorized.csv (최종 production)")

# 캐시 파일도 갱신(재사용 대비)
if len(new_desc):
    desc_full_updated = pd.concat([desc_full, new_desc])
    desc_full_updated.to_csv(f"{BASE}\\casrn_descriptors_full.csv", index_label="casrn")
    print(f"casrn_descriptors_full.csv 갱신: {len(desc_full)} -> {len(desc_full_updated)}")

    smiles_cache = pd.read_csv(f"{BASE}\\casrn_smiles_cache.csv")
    new_cache_rows = missing_keys.rename(columns={"key": "casrn", "canonical_smiles": "smiles"}).copy()
    new_cache_rows["cid"] = pd.NA
    new_cache_rows["status"] = "found"
    new_cache_rows = new_cache_rows[["casrn", "cid", "smiles", "status"]]
    smiles_cache_updated = pd.concat([smiles_cache, new_cache_rows], ignore_index=True)
    smiles_cache_updated.to_csv(f"{BASE}\\casrn_smiles_cache.csv", index=False)
    print(f"casrn_smiles_cache.csv 갱신: {len(smiles_cache)} -> {len(smiles_cache_updated)}")
