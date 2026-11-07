# -*- coding: utf-8 -*-
"""eye팀원 피드백 반영 — comp_group_id를 조성해시 단독이 아니라
"조성해시 그룹" UNION "벡터해시 그룹"(union-find)으로 재계산.

eye팀원이 지적한 구멍:
  (1) 구조 미해결(0개) 행이 comp_sig=="" 로 전부 한 그룹 -> 거대그룹 위험
      (확인 결과 내 기존 구현은 UNRESOLVED_{formulation_id}로 이미 안전 — 재확인만 함)
  (2) 미해결 성분을 서명에서 빼므로, 해결성분은 같고 미해결성분만 다른 제형이 과병합됨
      (누수방지 방향으로는 안전, 그냥 불필요하게 보수적인 과병합 — union-find로도 해결 안 되는
       잔여 한계, 벡터해시 쪽에서도 안 걸리면 그대로 남음. 보고서에 한계로만 명시)
  (3) 함량 소수1자리 반올림으로 40.04/40.0이 같은 서명이 됨 -> 실제로 다른 feature인데 과병합
      (조성해시만 쓰면 못 잡는 진짜 오류. 벡터해시를 OR 조건으로 union하면, "조성해시가 합쳤지만
       실제 벡터가 다른" 쌍을 분리하지는 못함 — union은 "더 묶기"만 가능하고 "쪼개기"는 못 함.
       하지만 핵심 안전조건("벡터가 같은데 다른 그룹"은 전부 해소)은 이걸로 완전히 충족됨.
       "벡터가 다른데 같은 그룹"(과병합, 홀 ②③)은 risk가 아니라 보수성(conservatism)이라
       5-fold 균형에 미치는 영향만 확인하면 됨.)

방법: 4가지 집계방식(wmean/top/max/min) 전부에 대해 "벡터해시(소수 6자리 반올림)" 그룹을 만들고,
조성해시 그룹과 함께 union-find로 합침 — 제형 i와 j가 (a) 조성해시가 같거나 (b) 4개 방식 중
하나라도 벡터해시가 같으면 같은 그룹으로 통합. 이러면 "어떤 집계방식을 쓰든 feature가 같은 쌍"은
반드시 같은 그룹이 되어, StratifiedGroupKFold가 절대 다른 fold로 쪼개지 않음.

production skin_vectorized.csv에 적용. 기존 두 버전은 보존:
  comp_group_id_manual_v1 (2026-08-29 수작업 15쌍)
  comp_group_id_compsig_v2 (오늘 낮에 적용한 조성해시 단독판 — 이번에 한 단계 더 발전시킴)
  comp_group_id (최종 = 조성해시 UNION 벡터해시)
"""
import hashlib
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

BASE = r"C:\Users\UserK\학술제\skin irritation"
SRC_B5 = r"C:\Users\UserK\OneDrive - 덕성여자대학교\문서\카카오톡 받은 파일\skin_irritation_최종후보.xlsx"
VEC_PATH = rf"{BASE}\skin_vectorized.csv"

# ------------------------------------------------------------------ 0) 백업
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = rf"{BASE}\skin_vectorized_backup_{ts}.csv"
shutil.copy2(VEC_PATH, backup_path)
print("백업 완료:", backup_path)

# ------------------------------------------------------------------ 1) 원자료 재구성 (직전 스크립트와 동일 로직)
filt = pd.read_csv(rf"{BASE}\skin_filtered.csv")
desc_full = pd.read_csv(rf"{BASE}\casrn_descriptors_full.csv").set_index("casrn")
FEATURE_COLS = list(desc_full.columns)

known = filt.dropna(subset=["casrn"]).copy()
known = known.drop_duplicates(subset=["formulation_id", "casrn"], keep="first")
known["has_descriptor"] = known["casrn"].isin(desc_full.index)
known = known.rename(columns={"casrn": "key", "percent_active_ingredient": "pct"})
known_unified = known[["formulation_id", "key", "pct", "has_descriptor"]].copy()

vec = pd.read_csv(VEC_PATH, low_memory=False)
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

# ------------------------------------------------------------------ 2) 조성해시 (기존, UNRESOLVED 안전처리 유지)
def comp_sig_key(fid, grp):
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        return f"UNRESOLVED_{fid}"
    sig = "|".join(sorted(f"{k}:{p:.1f}" for k, p in zip(g_resolved["key"], g_resolved["pct"]) if pd.notna(p)))
    if sig == "":
        return f"UNRESOLVED_{fid}"
    return sig  # 원시 서명 자체를 union-find 키로 씀 (해시는 맨 마지막에 한 번만)


# ------------------------------------------------------------------ 3) 4가지 집계방식 벡터해시
def aggregate_group(g_resolved, method):
    d = desc_full.loc[g_resolved["key"]].values
    pct = g_resolved["pct"].fillna(1.0).values
    if pct.sum() == 0:
        pct = np.ones_like(pct)
    if method == "wmean":
        return (pct[:, None] * d).sum(axis=0) / pct.sum()
    elif method == "top":
        idx = int(np.argmax(pct))
        return d[idx]
    elif method == "max":
        return d.max(axis=0)
    elif method == "min":
        return d.min(axis=0)
    raise ValueError(method)


# production skin_vectorized.csv 는 단 하나의 집계방식(wmean)로만 계산된 desc_*/fp_w_* 피처를 쓴다.
# top/max/min은 이 파일에 존재하지도 않는 다른 피처셋이라, 그 벡터해시를 같이 union하면 "wmean에서는
# 전혀 같지 않은 두 행"이 다른 방식의 우연한 일치를 매개로 전이적으로 체인되어 묶이는 오류가 나놀
# (실제로 MIX124/MIX177/MIX200/MIX202/MIX261이 한 그룹(36건)으로 묶인 것을 확인함 — 서로 다른
# 방식으로만 연결된 체인이었고 실제 공유 feature는 일부에만 있었음). 그래서 production은 **wmean 단일 방식만** union한다.
METHODS = ["wmean"]
fids = list(raw_all.groupby("formulation_id").groups.keys())
fid_idx = {fid: i for i, fid in enumerate(fids)}
n = len(fids)

comp_keys = [None] * n
vector_keys = {m: [None] * n for m in METHODS}

for fid, grp in raw_all.groupby("formulation_id"):
    i = fid_idx[fid]
    comp_keys[i] = comp_sig_key(fid, grp)
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        for m in METHODS:
            vector_keys[m][i] = f"UNRESOLVED_{fid}"  # 벡터 자체가 없으니(전부 NaN) 조성과 동일하게 고유 처리
        continue
    for m in METHODS:
        v = aggregate_group(g_resolved, m)
        vector_keys[m][i] = tuple(np.round(v, 6))

# ------------------------------------------------------------------ 4) union-find
parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[ry] = rx


def union_by_key(keys):
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    for idxs in buckets.values():
        if len(idxs) > 1:
            for j in idxs[1:]:
                union(idxs[0], j)


union_by_key(comp_keys)
for m in METHODS:
    union_by_key(vector_keys[m])

# ------------------------------------------------------------------ 5) 연결요소 -> 최종 group id (조성서명 해시, 해석가능성 유지)
components = {}
for i in range(n):
    components.setdefault(find(i), []).append(i)

new_group = {}
for comp_idxs in components.values():
    if len(comp_idxs) == 1:
        i = comp_idxs[0]
        label = comp_keys[i] if comp_keys[i].startswith("UNRESOLVED_") else (
            "GRP_" + hashlib.md5(comp_keys[i].encode()).hexdigest()[:12]
        )
    else:
        # 여러 조성서명이 벡터해시로 묶인 경우 전부 정렬해 합쳐서 해시(결정론적, 추적 가능)
        sigs = sorted(set(comp_keys[i] for i in comp_idxs))
        label = "GRP_" + hashlib.md5("||".join(sigs).encode()).hexdigest()[:12]
    for i in comp_idxs:
        new_group[fids[i]] = label

new_group_df = pd.Series(new_group, name="comp_group_id_v3").rename_axis("fid_raw").reset_index()
new_group_df["formulation_id_in_vec"] = new_group_df["fid_raw"]
is_aug_key = new_group_df["fid_raw"].isin(aug_meta["formulation_id"])
new_group_df.loc[is_aug_key, "formulation_id_in_vec"] = "AUG_" + new_group_df.loc[is_aug_key, "fid_raw"]

print("고유 그룹 수 (조성해시 UNION 벡터해시, 4방식):", new_group_df["comp_group_id_v3"].nunique(),
      "/ 제형 수:", len(new_group_df))

n_merged_by_vector_only = sum(1 for idxs in components.values()
                               if len(idxs) > 1 and len(set(comp_keys[i] for i in idxs)) > 1)
print("조성서명은 다른데 벡터해시 때문에 묶인 연결요소 수(순수 벡터기여분):", n_merged_by_vector_only)

# ------------------------------------------------------------------ 6) skin_vectorized.csv 갱신
merge_map = new_group_df.set_index("formulation_id_in_vec")["comp_group_id_v3"]

vec["comp_group_id_compsig_v2"] = vec["comp_group_id"]  # 오늘 낮 조성해시 단독판 보존
mapped = vec["formulation_id"].map(merge_map)
n_unmapped = mapped.isna().sum()
assert n_unmapped == 0, f"comp_group_id_v3 매핑 실패 {n_unmapped}건"
vec["comp_group_id"] = mapped

vec.to_csv(VEC_PATH, index=False)

print()
print("=== 적용 완료 ===")
print("comp_group_id 보존 이력: comp_group_id_manual_v1(수작업 15쌍) -> comp_group_id_compsig_v2(조성해시 단독) -> comp_group_id(최종, union)")
print("고유 그룹 수 비교 - manual_v1:", vec["comp_group_id_manual_v1"].nunique(),
      "/ compsig_v2:", vec["comp_group_id_compsig_v2"].nunique(),
      "/ 최종(union, wmean만):", vec["comp_group_id"].nunique())
print("백업 파일:", backup_path)
