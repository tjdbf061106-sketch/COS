"""
eye_filtered.csv -> 벡터화 (제형 단위 집계)

결정사항
- 벡터화 단위 : 제형(formulation) 1개 = 1벡터
                 혼합물은 성분 벡터를 pct_active_ingredient로 가중평균
                 단일물질(Chemical)은 성분 1개짜리 제형으로 취급
- 라벨        : epa_class (Cat I~IV) + 이진 라벨(I/II = 자극성 1, III/IV = 0)
- 피처        : Morgan FP(ECFP4, r=2, 2048bit) + RDKit physchem descriptors + ESOL
- 결측/중복   : SMILES 결측·파싱실패 행은 제외(로그 기록), 동일 구조 중복은 유지

── 이번 개정 (RDKit 단일 경로 통합) ──────────────────────────────────────
① SMILES 백필   : backfill_smiles.py(PubChem)의 결과를 병합.
                   `smiles_source` 로 원본/PubChem 출처를 추적한다.
② 최대 프래그먼트: descriptor 는 최대 프래그먼트 기준으로 계산한다.
                   기존엔 전체 분자로 계산해서 NaCl 염의 Na, UVCB 를 2성분
                   한 분자로 적은 SMILES(Abamectin, Spinosad) 가 MW·LogP 를
                   합산값으로 부풀렸다. 제거된 질량은 aux_mw_removed 로 남긴다.
                   Morgan FP 는 염 정보도 의미가 있어 전체 분자 기준 유지.
③ ESOL          : Delaney(2004) 수용성 추정. RDKit 기본 217개에 없는 유일한
                   물성이라 직접 계산한다.
④ comp_group_id : 3주차 build_harmonized.py 와 동일한 조성 해시.
                   같은 조성 제형이 train/test 로 갈리면 성능이 부풀려지므로
                   StratifiedGroupKFold 의 group 키로 쓴다.
                   SMILES 유무와 무관하게 원본 전체 행에서 계산한다.
⑤ 조성완전성    : known_pct / unknown_pct / undeclared_pct.
                   FIFRA 신고서는 활성성분만 적어서 잔부가 크다 -> AD 신뢰도 축.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

SRC = Path(r"C:\Users\한세음\COS\1. filteration\eye_filtered.csv")
OUT = Path(__file__).resolve().parent
BACKFILL = OUT / "smiles_backfill.csv"
OUT.mkdir(parents=True, exist_ok=True)

FP_BITS = 2048
FP_RADIUS = 2

log = {}


# ---------------------------------------------------------------- 1. 로드
df = pd.read_csv(SRC)
log["n_rows_raw"] = len(df)

# 제형 키: 혼합물은 formulation_id, 단일물질은 record_id
df["form_key"] = df["formulation_id"].fillna("CHEM::" + df["record_id"].astype(str))
log["n_formulations_raw"] = df["form_key"].nunique()


# ------------------------------------------------- 1b. SMILES 백필 (PubChem)
df["smiles_source"] = np.where(df["smiles"].notna(), "original", "missing")

if BACKFILL.exists():
    bf = pd.read_csv(BACKFILL)
    bf = bf[bf["smiles_new"].notna()][["casrn", "chemical_name", "smiles_new", "cid"]]
    df = df.merge(bf, on=["casrn", "chemical_name"], how="left")
    filled = df["smiles"].isna() & df["smiles_new"].notna()
    df.loc[filled, "smiles"] = df.loc[filled, "smiles_new"]
    df.loc[filled, "smiles_source"] = "pubchem"
    df["pubchem_cid"] = np.where(filled, df["cid"], np.nan)
    df = df.drop(columns=["smiles_new", "cid"])
    log["n_smiles_backfilled"] = int(filled.sum())
else:
    df["pubchem_cid"] = np.nan
    log["n_smiles_backfilled"] = 0
    print("[warn] smiles_backfill.csv 없음 - backfill_smiles.py 를 먼저 실행")


# --------------------------------------- 1c. 조성 해시 / 조성완전성 (SMILES 무관)
# 3주차 규칙과 동일: ing_key = "CAS:%" 정렬 후 '|' 결합 -> md5[:10], 'C' 접두
#   - CAS 결측 성분은 '__UNK__' 로 두되 버리지 않는다 (unknown_pct 로 보존)
#   - % 결측은 원액(100%) 취급. 가중치 계산에는 이 규칙을 쓰지 않는다.
ck = df["casrn"].where(df["casrn"].notna() & df["casrn"].astype(str).ne(""), np.nan)
comp_key = pd.DataFrame(
    {
        "form_key": df["form_key"],
        "ck": ck.fillna("__UNK__"),
        "is_unknown_ing": ck.isna(),
        "pct": pd.to_numeric(df["pct_active_ingredient"], errors="coerce"),
    }
)
comp_key["pct_missing"] = comp_key["pct"].isna()
comp_key["pct"] = comp_key["pct"].fillna(100.0)
comp_key = comp_key.drop_duplicates(["form_key", "ck", "pct"])

comp_key["ing_key"] = comp_key["ck"] + ":" + comp_key["pct"].round(6).map(lambda x: f"{x:g}")
gk = comp_key.groupby("form_key")

grp = pd.DataFrame(
    {
        "composition_key": gk["ing_key"].apply(lambda s: "|".join(sorted(s))),
        "unknown_count": gk["is_unknown_ing"].sum().astype(int),
        "known_pct": gk.apply(
            lambda g: g.loc[~g["is_unknown_ing"], "pct"].sum(), include_groups=False
        ).round(6),
        "unknown_pct": gk.apply(
            lambda g: g.loc[g["is_unknown_ing"], "pct"].sum(), include_groups=False
        ).round(6),
        "has_missing_pct_raw": gk["pct_missing"].any(),
    }
)
grp["comp_group_id"] = grp["composition_key"].map(
    lambda k: "C" + hashlib.md5(k.encode()).hexdigest()[:10]
)
grp["undeclared_pct"] = (100.0 - grp["known_pct"] - grp["unknown_pct"]).clip(lower=0.0).round(6)
log["n_comp_groups"] = int(grp["comp_group_id"].nunique())
log["n_formulations_with_twin"] = int(
    grp["comp_group_id"].map(grp["comp_group_id"].value_counts()).gt(1).sum()
)
log["undeclared_pct_median"] = float(grp["undeclared_pct"].median())


# ------------------------------------------------- 2. SMILES 정제 (성분 행 단위)
df["smiles_missing"] = df["smiles"].isna()


def largest_fragment(m):
    """최대 프래그먼트(중원자 수 -> 분자량 순) 반환. descriptor 계산 기준."""
    # sanitizeFrags=True 필수. False 로 쪼개면 RingInfo 가 초기화되지 않아
    # TPSA / FpDensityMorgan / SPS / BCUT2D 등이 조용히 NaN 으로 떨어진다.
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return frags[0]
    return max(frags, key=lambda f: (f.GetNumHeavyAtoms(), Descriptors.MolWt(f)))


mols, mols_desc, canon, parse_fail = [], [], [], []
n_frags, mw_removed = [], []
for s in df["smiles"]:
    if pd.isna(s):
        mols.append(None)
        mols_desc.append(None)
        canon.append(np.nan)
        parse_fail.append(False)
        n_frags.append(np.nan)
        mw_removed.append(np.nan)
        continue
    m = Chem.MolFromSmiles(s)
    mols.append(m)
    canon.append(Chem.MolToSmiles(m) if m is not None else np.nan)
    parse_fail.append(m is None)
    if m is None:
        mols_desc.append(None)
        n_frags.append(np.nan)
        mw_removed.append(np.nan)
    else:
        lf = largest_fragment(m)
        mols_desc.append(lf)
        n_frags.append(len(Chem.GetMolFrags(m)))
        mw_removed.append(round(Descriptors.MolWt(m) - Descriptors.MolWt(lf), 4))

df["canonical_smiles"] = canon
df["parse_fail"] = parse_fail
df["n_fragments"] = n_frags
df["mw_removed"] = mw_removed

log["n_rows_smiles_missing"] = int(df["smiles_missing"].sum())
log["n_rows_parse_fail"] = int(df["parse_fail"].sum())
log["n_rows_multifragment"] = int((df["n_fragments"] > 1).sum())
log["n_rows_mw_removed_gt50"] = int((df["mw_removed"] > 50).sum())

dropped = df[df["smiles_missing"] | df["parse_fail"]].copy()
dropped["drop_reason"] = np.where(dropped["smiles_missing"], "smiles_missing", "rdkit_parse_fail")
dropped[
    ["record_id", "form_key", "chemical_name", "casrn", "smiles", "epa_class", "drop_reason"]
].to_csv(OUT / "dropped_rows.csv", index=False, encoding="utf-8-sig")

keep = ~(df["smiles_missing"] | df["parse_fail"])
comp = df[keep].reset_index(drop=True)
comp_mols = [m for m, k in zip(mols, keep) if k]
comp_mols_desc = [m for m, k in zip(mols_desc, keep) if k]
log["n_rows_vectorizable"] = len(comp)
log["n_unique_structures"] = int(comp["canonical_smiles"].nunique())


# ------------------------------------------------- 3. 성분 행 단위 피처 계산
# Morgan FP 는 전체 분자(염 포함), descriptor 는 최대 프래그먼트 기준
gen = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

fp = np.zeros((len(comp_mols), FP_BITS), dtype=np.float32)
for i, m in enumerate(comp_mols):
    fp[i] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)

desc_names = [n for n, _ in Descriptors._descList]
desc = np.full((len(comp_mols_desc), len(desc_names)), np.nan, dtype=np.float64)
for i, m in enumerate(comp_mols_desc):
    try:
        d = Descriptors.CalcMolDescriptors(m)
        desc[i] = [d.get(n, np.nan) for n in desc_names]
    except Exception:
        pass

desc[~np.isfinite(desc)] = np.nan  # inf -> nan (Ipc 등 폭발값 포함)


def esol_logS(m):
    """Delaney(2004) ESOL 수용성 추정, log10(mol/L).

        logS = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP
        AP = 방향족 중원자 / 전체 중원자
    """
    try:
        heavy = m.GetNumHeavyAtoms()
        if heavy == 0:
            return np.nan
        arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
        return (
            0.16
            - 0.63 * Descriptors.MolLogP(m)
            - 0.0062 * Descriptors.MolWt(m)
            + 0.066 * Descriptors.NumRotatableBonds(m)
            - 0.74 * (arom / heavy)
        )
    except Exception:
        return np.nan


logS = np.array([esol_logS(m) for m in comp_mols_desc], dtype=np.float64)
mw_arr = np.array(
    [Descriptors.MolWt(m) if m is not None else np.nan for m in comp_mols_desc], dtype=np.float64
)
with np.errstate(over="ignore", invalid="ignore"):
    logS_mgL = np.log10(np.clip(10.0**logS * mw_arr * 1000.0, 1e-30, None))

# 구조 보조 피처
aux = pd.DataFrame(
    {
        "n_fragments": comp["n_fragments"].to_numpy(dtype=np.float64),
        "has_charge": [
            int(any(a.GetFormalCharge() != 0 for a in m.GetAtoms())) for m in comp_mols
        ],
        "mw_removed": comp["mw_removed"].to_numpy(dtype=np.float64),
        "ESOL_logS": logS,
        "ESOL_logS_mgL": logS_mgL,
        "smiles_from_pubchem": comp["smiles_source"].eq("pubchem").astype(int).to_numpy(),
    }
)

log["n_descriptors"] = len(desc_names)
log["n_aux_features"] = aux.shape[1]
log["n_fp_bits"] = FP_BITS
log["esol_logS_describe"] = (
    pd.Series(logS).describe().round(3).to_dict()
)


# ------------------------------------------------- 4. 가중치 (성분 비율)
comp["w_raw"] = pd.to_numeric(comp["pct_active_ingredient"], errors="coerce")

weights = np.empty(len(comp), dtype=np.float64)
weight_mode = {}
for key, idx in comp.groupby("form_key").groups.items():
    idx = np.asarray(idx)
    w = comp.loc[idx, "w_raw"].to_numpy(dtype=np.float64)
    if np.isfinite(w).all() and np.nansum(w) > 0:
        w = w / w.sum()
        weight_mode[key] = "pct_active_ingredient"
    else:
        w = np.full(len(idx), 1.0 / len(idx))
        weight_mode[key] = "equal_fallback"
    weights[comp.index.get_indexer(idx)] = w

comp["weight"] = weights
log["weight_mode_counts"] = pd.Series(weight_mode).value_counts().to_dict()


# ------------------------------------------------- 5. 제형 단위 집계
form_keys = comp["form_key"].drop_duplicates().tolist()
pos = {k: i for i, k in enumerate(form_keys)}

FP_agg = np.zeros((len(form_keys), FP_BITS), dtype=np.float32)   # 가중평균
FP_any = np.zeros((len(form_keys), FP_BITS), dtype=np.uint8)     # 성분 중 1개라도 보유
D_agg = np.full((len(form_keys), len(desc_names)), np.nan)
AUX_agg = np.full((len(form_keys), aux.shape[1]), np.nan)
meta_rows = []

aux_np = aux.to_numpy(dtype=np.float64)

for key, g in comp.groupby("form_key", sort=False):
    r = pos[key]
    loc = comp.index.get_indexer(np.asarray(g.index))
    w = comp["weight"].to_numpy()[loc]

    FP_agg[r] = (fp[loc] * w[:, None]).sum(axis=0)
    FP_any[r] = (fp[loc] > 0).any(axis=0).astype(np.uint8)

    # descriptor: nan 무시 가중평균
    d = desc[loc]
    mask = np.isfinite(d)
    wm = mask * w[:, None]
    denom = wm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        D_agg[r] = np.where(denom > 0, np.nansum(np.where(mask, d, 0) * w[:, None], axis=0) / denom, np.nan)

    # aux 도 같은 규칙(nan 무시 가중평균)으로 집계 — mw_removed/ESOL 에 nan 가능
    a = aux_np[loc]
    amask = np.isfinite(a)
    adenom = (amask * w[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        AUX_agg[r] = np.where(
            adenom > 0, np.nansum(np.where(amask, a, 0) * w[:, None], axis=0) / adenom, np.nan
        )

    epa = g["epa_class"].dropna().unique()
    ghs = g["ghs_class"].dropna().unique()
    meta_rows.append(
        {
            "form_key": key,
            "source": g["source"].iloc[0],
            "substance_type": g["substance_type"].iloc[0],
            "formulation_name": g["formulation_name"].iloc[0],
            "n_components_used": len(g),
            "n_active_ingredients": g["n_active_ingredients"].iloc[0],
            "declared_composition_sum": g["declared_composition_sum"].iloc[0],
            "pct_sum_used": g["w_raw"].sum(min_count=1),
            "weight_mode": weight_mode[key],
            "n_smiles_from_pubchem": int(g["smiles_source"].eq("pubchem").sum()),
            # pandas 3 의 str dtype 은 astype(str) 로 NaN 이 문자열이 되지 않아 fillna 선행
            "component_names": " | ".join(g["chemical_name"].fillna("NA").astype(str)),
            "component_casrns": " | ".join(g["casrn"].fillna("NA").astype(str)),
            "epa_class": epa[0] if len(epa) == 1 else (np.nan if len(epa) == 0 else epa[0]),
            "epa_class_conflict": len(epa) > 1,
            "ghs_class": ghs[0] if len(ghs) == 1 else (np.nan if len(ghs) == 0 else ghs[0]),
            "ghs_class_conflict": len(ghs) > 1,
            "draize_score": g["draize_score"].dropna().mean() if g["draize_score"].notna().any() else np.nan,
        }
    )

meta = pd.DataFrame(meta_rows)
log["n_formulations_vectorized"] = len(meta)
log["n_epa_class_conflict"] = int(meta["epa_class_conflict"].sum())
log["n_ghs_class_conflict"] = int(meta["ghs_class_conflict"].sum())

# 조성 해시 / 조성완전성 병합 (원본 전체 행 기준으로 계산된 값)
meta = meta.merge(
    grp[
        [
            "comp_group_id",
            "composition_key",
            "known_pct",
            "unknown_pct",
            "undeclared_pct",
            "unknown_count",
        ]
    ],
    left_on="form_key",
    right_index=True,
    how="left",
)
# 성분이 일부 탈락한 제형은 구조벡터가 조성 전체를 대표하지 못한다 -> 커버리지로 남긴다
n_raw_comp = df.groupby("form_key").size()
meta["n_components_declared"] = meta["form_key"].map(n_raw_comp).astype(int)
meta["n_components_dropped"] = meta["n_components_declared"] - meta["n_components_used"]
meta["structure_coverage"] = (
    meta["n_components_used"] / meta["n_components_declared"]
).round(6)
meta["group_size"] = meta["comp_group_id"].map(meta["comp_group_id"].value_counts()).astype(int)
meta["is_composition_twin"] = meta["group_size"].gt(1)

log["n_formulations_partial_structure"] = int(meta["n_components_dropped"].gt(0).sum())

# 라벨
meta["y_epa"] = pd.to_numeric(meta["epa_class"], errors="coerce")
meta["y_binary"] = np.where(
    meta["y_epa"].isin([1.0, 2.0]), 1, np.where(meta["y_epa"].isin([3.0, 4.0]), 0, np.nan)
)
_conf = meta.groupby("comp_group_id")["y_epa"].nunique()
meta["group_label_conflict"] = meta["comp_group_id"].map(_conf > 1)
log["y_epa_counts"] = meta["y_epa"].value_counts(dropna=False).to_dict()
log["y_binary_counts"] = meta["y_binary"].value_counts(dropna=False).to_dict()
log["n_group_label_conflict"] = int(
    meta[meta["is_composition_twin"]]["group_label_conflict"].sum()
)


# ------------------------------------------------- 6. 행렬 정리 및 저장
desc_df = pd.DataFrame(D_agg, columns=[f"desc_{n}" for n in desc_names])
aux_df = pd.DataFrame(AUX_agg, columns=[f"aux_{c}" for c in aux.columns])
desc_df = pd.concat([desc_df, aux_df], axis=1)

# 전부 결측인 descriptor 열, 분산 0인 열 식별 (제거하지 않고 목록만 남김)
all_nan = desc_df.columns[desc_df.isna().all()].tolist()
zero_var = desc_df.columns[desc_df.nunique(dropna=True) <= 1].tolist()
log["desc_all_nan_cols"] = all_nan
log["desc_zero_variance_cols"] = zero_var

fp_agg_df = pd.DataFrame(FP_agg, columns=[f"fp_w_{i}" for i in range(FP_BITS)])
fp_any_df = pd.DataFrame(FP_any, columns=[f"fp_any_{i}" for i in range(FP_BITS)])

idx = meta[["form_key"]]
pd.concat([idx, meta.drop(columns=["form_key"])], axis=1).to_csv(
    OUT / "formulation_meta_labels.csv", index=False, encoding="utf-8-sig"
)
pd.concat([idx, desc_df], axis=1).to_parquet(OUT / "X_descriptors.parquet", index=False)
pd.concat([idx, fp_agg_df], axis=1).to_parquet(OUT / "X_morgan_weighted.parquet", index=False)
pd.concat([idx, fp_any_df], axis=1).to_parquet(OUT / "X_morgan_any.parquet", index=False)

# 모델 투입용 결합 행렬 (가중 FP + descriptor), 라벨 있는 행만
full = pd.concat([idx, fp_agg_df, desc_df], axis=1)
labeled = meta["y_epa"].notna().to_numpy()
full[labeled].to_parquet(OUT / "X_combined_labeled.parquet", index=False)
meta[labeled][
    ["form_key", "y_epa", "y_binary", "source", "substance_type", "comp_group_id"]
].to_parquet(OUT / "y_labeled.parquet", index=False)
log["n_labeled"] = int(labeled.sum())

# 성분 단위 중간 산출물 (추적용)
comp_out = comp[
    [
        "record_id", "form_key", "source", "substance_type", "chemical_name",
        "casrn", "canonical_smiles", "smiles_source", "pubchem_cid",
        "n_fragments", "mw_removed", "pct_active_ingredient", "weight", "epa_class",
    ]
].copy()
comp_out["ESOL_logS"] = logS.round(4)
comp_out.to_csv(OUT / "component_level_clean.csv", index=False, encoding="utf-8-sig")

(OUT / "vectorization_log.json").write_text(
    json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)

print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
