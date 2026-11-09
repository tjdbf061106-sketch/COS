# -*- coding: utf-8 -*-
"""eye팀 작업정리 PDF(2026-11-08)가 지적한 skin comp_group_id 잔여 결함 수정 + 농도 피처 검증.

eye팀이 skin 수정안(apply_vector_hash_union_grouping.py)을 감사해서 찾은 것:
  ① 구조 미해결(0개) 행 -> comp_sig="" 전원 한 그룹      : 이미 안전(UNRESOLVED_{fid}) - 재확인만
  ② 미해결 성분 무시로 과병합                              : 보수적 방향이라 낮은 우선순위, 이번엔 유지
  ③ 함량 소수 1자리 반올림(`f"{p:.1f}"`)로 40.04/40.0 같은 서명 -> 과병합   <- 이번에 수정
또한 eye팀이 실험 11에서 검증한 "방안 1(농도 피처 추가)"을 skin에도 동일하게 적용:
  벡터해시(dedup 판정용)와 모델 feature 양쪽에 `resolved_pct`(구조 해결 성분 % 합)를 추가하면
  "성분 구성비는 같은데 농도 스케일만 다른" 제형이 더 이상 같은 벡터로 뭉개지지 않는지 확인한다.

이 스크립트는 skin_vectorized.csv를 갱신하지만 모든 이전 버전을 보존한다:
  comp_group_id_manual_v1      (2026-08-29 수작업 15쌍)
  comp_group_id_compsig_v2     (조성해시 단독, .1f 반올림 버그 포함)
  comp_group_id_vectoronly_v3  (조성해시 UNION 벡터해시(wmean, desc/fp만), 이번 수정 직전 최종본)
  comp_group_id                (최종 = 조성해시(6자리 수정) UNION 벡터해시(desc/fp + resolved_pct))
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

# ------------------------------------------------------------------ 1) 원자료 재구성 (기존 스크립트와 동일 로직)
filt = pd.read_csv(rf"{BASE}\skin_filtered.csv")
desc_full = pd.read_csv(rf"{BASE}\casrn_descriptors_full.csv").set_index("casrn")

known = filt.dropna(subset=["casrn"]).copy()
known = known.drop_duplicates(subset=["formulation_id", "casrn"], keep="first")
known["has_descriptor"] = known["casrn"].isin(desc_full.index)
known = known.rename(columns={"casrn": "key", "percent_active_ingredient": "pct"})
known_unified = known[["formulation_id", "key", "pct", "has_descriptor"]].copy()

vec = pd.read_csv(VEC_PATH, low_memory=False)
aug_meta = vec.loc[vec["formulation_id"].astype(str).str.startswith("AUG_"), ["formulation_id"]].copy()
aug_meta["formulation_id"] = aug_meta["formulation_id"].str.replace("AUG_", "", regex=False)

NUMBER = r"(?:\d+\.\d+|\.\d+|\d+)"  # 2026-11-09: 선행 소수점(".5")도 허용 — eye팀 감사에서 발견된 lstrip('.') 버그(".5%"가 "5%"로 읽히는 10배 오류) 수정의 짝
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
    return cleaned  # 2026-11-09: 선행 "." lstrip 제거 — ".5"를 "5"로 잃는 10배 오류를 낳음(NUMBER 정규식이 이제 선행점 허용하므로 더 이상 필요 없음)


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

# ------------------------------------------------------------------ 2) 조성해시 (버그 ③ 수정: .1f -> 6자리)
def comp_sig_key(fid, grp, decimals):
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        return f"UNRESOLVED_{fid}"
    fmt = f"%.{decimals}f"
    sig = "|".join(
        sorted(f"{k}:{fmt % p}" for k, p in zip(g_resolved["key"], g_resolved["pct"]) if pd.notna(p))
    )
    if sig == "":
        return f"UNRESOLVED_{fid}"
    return sig


# ------------------------------------------------------------------ 3) wmean 벡터해시 (desc/fp만 vs +resolved_pct)
def aggregate_wmean(g_resolved):
    d = desc_full.loc[g_resolved["key"]].values
    pct = g_resolved["pct"].fillna(1.0).values
    if pct.sum() == 0:
        pct = np.ones_like(pct)
    return (pct[:, None] * d).sum(axis=0) / pct.sum(), pct.sum()


fids = list(raw_all.groupby("formulation_id").groups.keys())
fid_idx = {fid: i for i, fid in enumerate(fids)}
n = len(fids)

comp_keys_old = [None] * n   # .1f (재현용, 버그 재확인)
comp_keys_new = [None] * n   # 6자리 (수정판)
vec_keys_plain = [None] * n       # desc/fp만 (기존 production과 동일 기준)
vec_keys_with_conc = [None] * n   # desc/fp + resolved_pct (eye 방안1 검증용)
resolved_pct_calc = [None] * n

for fid, grp in raw_all.groupby("formulation_id"):
    i = fid_idx[fid]
    comp_keys_old[i] = comp_sig_key(fid, grp, 1)
    comp_keys_new[i] = comp_sig_key(fid, grp, 6)
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        vec_keys_plain[i] = f"UNRESOLVED_{fid}"
        vec_keys_with_conc[i] = f"UNRESOLVED_{fid}"
        resolved_pct_calc[i] = 0.0
        continue
    v, pct_sum = aggregate_wmean(g_resolved)
    resolved_pct_calc[i] = float(g_resolved["pct"].fillna(0).sum())
    vec_keys_plain[i] = tuple(np.round(v, 6))
    vec_keys_with_conc[i] = tuple(np.round(v, 6)) + (round(resolved_pct_calc[i], 6),)

n_sig_changed = sum(1 for a, b in zip(comp_keys_old, comp_keys_new) if a != b)
print(f"조성서명 변경(.1f -> 6자리) 영향받은 제형 수: {n_sig_changed} / {n}")


# ------------------------------------------------------------------ 4) union-find 유틸
def run_union(comp_keys, vec_keys):
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
    union_by_key(vec_keys)

    components = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    group = {}
    for comp_idxs in components.values():
        if len(comp_idxs) == 1:
            i = comp_idxs[0]
            label = comp_keys[i] if comp_keys[i].startswith("UNRESOLVED_") else (
                "GRP_" + hashlib.md5(comp_keys[i].encode()).hexdigest()[:12]
            )
        else:
            sigs = sorted(set(comp_keys[i] for i in comp_idxs))
            label = "GRP_" + hashlib.md5("||".join(sigs).encode()).hexdigest()[:12]
        for i in comp_idxs:
            group[fids[i]] = label
    return group, components


# (a) 기존 production과 동일 재현(.1f + desc/fp만) - 대조군
group_repro, _ = run_union(comp_keys_old, vec_keys_plain)
# (b) 6자리만 수정(desc/fp만 벡터해시 유지)
group_fixed_sig, _ = run_union(comp_keys_new, vec_keys_plain)
# (c) 6자리 수정 + 벡터해시에 농도 포함 (최종)
group_final, components_final = run_union(comp_keys_new, vec_keys_with_conc)

print()
print("고유 그룹 수 비교:")
print("  (a) 재현판(.1f, desc/fp만)      :", len(set(group_repro.values())))
print("  (b) 서명만 6자리로 수정          :", len(set(group_fixed_sig.values())))
print("  (c) 6자리 + 벡터해시에 농도 포함 :", len(set(group_final.values())))

n_multi_final = sum(1 for idxs in components_final.values() if len(idxs) > 1)
print("  (c) 최종 기준 2개 이상 묶인 연결요소 수:", n_multi_final)

# ------------------------------------------------------------------ 5) skin_vectorized.csv 갱신
def to_vecformat(group_dict):
    s = pd.Series(group_dict, name="g").rename_axis("fid_raw").reset_index()
    s["formulation_id_in_vec"] = s["fid_raw"]
    is_aug = s["fid_raw"].isin(aug_meta["formulation_id"])
    s.loc[is_aug, "formulation_id_in_vec"] = "AUG_" + s.loc[is_aug, "fid_raw"]
    return s.set_index("formulation_id_in_vec")["g"]


map_final = to_vecformat(group_final)
map_fixed_sig = to_vecformat(group_fixed_sig)

vec["comp_group_id_vectoronly_v3"] = vec["comp_group_id"]  # 이번 수정 직전 최종본 보존

mapped_final = vec["formulation_id"].map(map_final)
mapped_fixedsig = vec["formulation_id"].map(map_fixed_sig)
assert mapped_final.isna().sum() == 0, "comp_group_id(최종) 매핑 실패 행 존재"
assert mapped_fixedsig.isna().sum() == 0, "comp_group_id(서명만 수정) 매핑 실패 행 존재"

vec["comp_group_id_sigfix_only"] = mapped_fixedsig  # 참고용(서명만 고친 중간판)
vec["comp_group_id"] = mapped_final

# resolved_pct 재계산값 교차검증(기존 컬럼과 일치하는지) - 모델 feature로 바로 쓸 수 있음을 확인
map_rp = to_vecformat(dict(zip(fids, resolved_pct_calc)))
rp_check = vec["formulation_id"].map(map_rp)
mismatch = (rp_check.round(6) != vec["resolved_pct"].round(6)).sum()
print(f"\nresolved_pct 재계산 vs 기존 컬럼 불일치 행 수: {mismatch} (0이어야 정상)")

vec.to_csv(VEC_PATH, index=False)

before_groups = pd.read_csv(backup_path, low_memory=False)["comp_group_id"].nunique()
print()
print("=== 적용 완료 ===")
print(f"comp_group_id 고유 그룹 수: 수정 전 {before_groups} -> 수정 후 {vec['comp_group_id'].nunique()}")
print("보존 이력: manual_v1 -> compsig_v2 -> vectoronly_v3(수정 직전 최종) -> comp_group_id(최종, 서명6자리+농도포함 벡터해시)")
print("참고 컬럼: comp_group_id_sigfix_only(서명만 6자리로 고치고 농도는 아직 안 넣은 중간판, 효과 분리용)")
print("백업 파일:", backup_path)
