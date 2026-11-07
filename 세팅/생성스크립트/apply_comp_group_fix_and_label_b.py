# -*- coding: utf-8 -*-
"""production skin_vectorized.csv에 두 가지 수정 반영:
1) binary_label을 라벨 B(final_label in 1,2,3 -> 1) 기준으로 교체 (기존 A 값은 binary_label_A로 보존)
2) comp_group_id를 조성해시(해결된 성분 CASRN:함량 서명 MD5) 기준으로 재계산해서 교체
   (기존 값은 comp_group_id_manual_v1으로 보존 — 2026-08-29 팀원B가 수작업으로 15쌍만 식별한 값,
   `세팅/진행상황.md` 참조. 버그가 아니라 범위가 좁은 수작업 식별이었음 — 같은 활성성분이 다른
   제품명으로 등록된 경우까지는 못 잡음)

실행 전 skin_vectorized.csv를 타임스탬프 백업한다.
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

# ------------------------------------------------------------------ 1) 원자료 재구성 (comp_group_id_fixed 계산용)
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


def comp_group_from_raw(fid, grp):
    g_resolved = grp[grp["has_descriptor"]]
    if len(g_resolved) == 0:
        return f"UNRESOLVED_{fid}"
    sig = "|".join(sorted(f"{k}:{p:.1f}" for k, p in zip(g_resolved["key"], g_resolved["pct"]) if pd.notna(p)))
    if sig == "":
        return f"UNRESOLVED_{fid}"
    return "GRP_" + hashlib.md5(sig.encode()).hexdigest()[:12]


new_group = {fid: comp_group_from_raw(fid, grp) for fid, grp in raw_all.groupby("formulation_id")}
new_group_df = pd.Series(new_group, name="comp_group_id_fixed").rename_axis("fid_raw").reset_index()

# 원본 544개는 formulation_id가 그대로 키, 증강 118개는 skin_vectorized.csv에서 "AUG_" 접두사가 붙어있음
new_group_df["formulation_id_in_vec"] = new_group_df["fid_raw"]
is_aug_key = new_group_df["fid_raw"].isin(aug_meta["formulation_id"])
new_group_df.loc[is_aug_key, "formulation_id_in_vec"] = "AUG_" + new_group_df.loc[is_aug_key, "fid_raw"]

print("고유 조성그룹 수:", new_group_df["comp_group_id_fixed"].nunique(), "/ 제형 수:", len(new_group_df))

# ------------------------------------------------------------------ 2) skin_vectorized.csv 수정
merge_map = new_group_df.set_index("formulation_id_in_vec")["comp_group_id_fixed"]

n_before_missing = vec["comp_group_id"].isna().sum()
vec["comp_group_id_manual_v1"] = vec["comp_group_id"]  # 기존 값 보존(2026-08-29 수작업 식별본, 30건/15쌍)
mapped = vec["formulation_id"].map(merge_map)
n_unmapped = mapped.isna().sum()
assert n_unmapped == 0, f"comp_group_id_fixed 매핑 실패 {n_unmapped}건 — formulation_id 불일치 확인 필요"
vec["comp_group_id"] = mapped

vec["binary_label_A"] = vec["final_label"].map(lambda x: 1 if x in (1, 2) else 0)
vec["binary_label_B"] = vec["final_label"].map(lambda x: 1 if x in (1, 2, 3) else 0)
vec["binary_label"] = vec["binary_label_B"]  # 기본 binary_label을 라벨 B 기준으로 교체

vec.to_csv(VEC_PATH, index=False)

print()
print("=== 적용 완료 ===")
print("comp_group_id: 기존(comp_group_id_manual_v1, 수작업 15쌍 식별본) 보존, 새 값은 조성해시 기준으로 교체")
print(f"  고유 그룹 수 — 수정 전(manual_v1): {vec['comp_group_id_manual_v1'].nunique()} / 수정 후(조성해시): {vec['comp_group_id'].nunique()}")
print("binary_label: A(기존)->binary_label_A로 보존, B 기준으로 binary_label 교체")
print(f"  binary_label_A 양성: {vec['binary_label_A'].sum()}/{len(vec)}")
print(f"  binary_label_B(=새 binary_label) 양성: {vec['binary_label_B'].sum()}/{len(vec)}")
print(f"  binary_label(현재) 양성: {vec['binary_label'].sum()}/{len(vec)}")
print()
print("백업 파일:", backup_path)
