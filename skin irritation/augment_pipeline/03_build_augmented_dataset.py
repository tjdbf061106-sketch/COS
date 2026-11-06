# -*- coding: utf-8 -*-
"""skin_보강후보_통합.xlsx의 '추가후보' 125건 집계 (CASRN + 이름기반 구조해결 둘 다 반영,
함량_원문 텍스트 파서 포함). 01/02번 스크립트를 먼저 실행해 casrn_smiles_cache_new98.csv /
name_smiles_cache.csv를 만들어둔 뒤 실행할 것."""
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

new_cas = pd.read_csv(f"{BASE}\\casrn_smiles_cache_new98.csv")
new_cas_found = new_cas[new_cas["status"] == "found"][["casrn", "smiles"]].rename(columns={"casrn": "key"})
new_name = pd.read_csv(f"{BASE}\\name_smiles_cache.csv")
new_name_found = new_name[new_name["status"] == "found"][["name", "smiles"]].rename(columns={"name": "key"})

new_keys = pd.concat([new_cas_found, new_name_found], ignore_index=True).drop_duplicates(subset="key")
print(f"신규 구조 {len(new_keys)}개(CASRN {len(new_cas_found)} + 이름 {len(new_name_found)}) descriptor 계산 중...")

new_rows = []
for _, r in new_keys.iterrows():
    mol_orig = Chem.MolFromSmiles(r["smiles"])
    if mol_orig is None:
        continue
    mol_stripped = get_largest_fragment(mol_orig)
    row = {"key": r["key"]}
    row.update(compute_common8(mol_stripped))
    row.update(compute_full_descriptors(mol_stripped))
    row.update(compute_morgan_fp(mol_orig))
    new_rows.append(row)
new_desc = pd.DataFrame(new_rows).set_index("key")

# CASRN 기준 기존 desc_full과 새 구조(CASRN+이름)를 "key" 하나로 합친 조회 테이블
desc_full_keyed = desc_full.copy()
desc_full_keyed.index.name = "key"
lookup = pd.concat([desc_full_keyed, new_desc[~new_desc.index.isin(desc_full_keyed.index)]])
print(f"조회 테이블 크기: {len(lookup)} (기존 {len(desc_full_keyed)} + 신규 {len(new_desc)})")


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


candidates = pd.read_excel(f"{BASE}\\skin_보강후보_통합.xlsx", sheet_name="추가후보")
components = pd.read_excel(f"{BASE}\\skin_보강후보_통합.xlsx", sheet_name="성분함량")
# key: CAS_원문 있으면 그걸로, 없으면 성분명_원문으로 조회
components["key"] = components["CAS_원문"].fillna(components["성분명_원문"])
components["has_structure"] = components["key"].isin(lookup.index)

agg_rows = []
skipped_candidates = []
for _, cand in candidates.iterrows():
    key = cand["후보키"]
    comp = components[components["후보키"] == key].copy()
    g_resolved = comp[comp["has_structure"]]
    g_unresolved = comp[~comp["has_structure"]]
    if len(g_resolved) == 0:
        skipped_candidates.append(key)
        continue

    pct_raw = g_resolved.apply(get_pct, axis=1).values
    resolved_pct = np.nansum(pct_raw)
    unresolved_pct = g_unresolved.apply(get_pct, axis=1).sum() if len(g_unresolved) else 0.0
    pct = np.where(np.isnan(pct_raw), 1.0, pct_raw)
    if pct.sum() == 0:
        pct = np.ones_like(pct)

    common_mat = lookup.loc[g_resolved["key"], COMMON8_COLS].values
    desc_mat = lookup.loc[g_resolved["key"], DESC217_COLS].values
    fp_mat = lookup.loc[g_resolved["key"], FP_COLS].values

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
    row["source"] = "skin_보강후보_통합_v2v3"
    agg_rows.append(row)

aug = pd.DataFrame(agg_rows)[
    ["formulation_id", "comp_count", "is_single_component", "resolved_pct",
     "unresolved_structure_pct", "comp_group_id", "source", "binary_label"] + ALL_FEATURE_COLS
]
print(f"\n집계 완료: 후보 {len(candidates)}건 중 {len(aug)}건 성공, {len(skipped_candidates)}건 제외")
print(f"제외된 후보키: {skipped_candidates}")
print(f"binary_label 분포: {aug['binary_label'].value_counts().to_dict()}")

aug.to_csv(f"{BASE}\\skin_vectorized_augment_125.csv", index=False, encoding="utf-8-sig")
print(f"저장: {BASE}\\skin_vectorized_augment_125.csv")
