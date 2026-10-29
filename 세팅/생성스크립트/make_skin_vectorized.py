"""
skin_vectorized.csv 생성 스크립트 (3주차 최종 벡터화)

입력:
  - skin_filtered.csv (1,799행, record 단위, casrn/percent_active_ingredient/unknown_pct 등)
  - skin_formulation_level_with_comp_group.csv (544행, formulation 단위, final_label/comp_group_id 등)

처리 순서:
  1. CASRN 있는 행만으로 pivot (formulation_id x casrn -> percent_active_ingredient)
     - 성분이 원래 없는 조합은 0으로 채움
     - 성분은 있지만 percent가 결측인 8개 formulation(단일성분)만 NaN으로 남김 + has_missing_pct 플래그
  2. 성분개수(comp_count) 계산 -> 단일/다성분 플래그
  3. formulation 단위 라벨(final_label, comp_group_id, is_composition_twin, internal_conflict_resolved) 조인
  4. binary_label 파생 (Category 1~2 -> 1, 3~4 -> 0)
  5. unknown_pct / is_unknown_ingredient formulation 단위로 조인
  6. baseline(전체 544) + 다성분 서브셋(comp_count>1) 두 버전 저장
"""
import pandas as pd

BASE = r"C:\Users\UserK\Desktop\학술제\skin irritation"

filtered = pd.read_csv(rf"{BASE}\skin_filtered.csv")
labels = pd.read_csv(rf"{BASE}\skin_formulation_level_with_comp_group.csv")

# --- 1. pivot ---
known = filtered.dropna(subset=["casrn"])

present = known.groupby(["formulation_id", "casrn"]).size().unstack(fill_value=0) > 0
values = known.pivot_table(index="formulation_id", columns="casrn",
                            values="percent_active_ingredient", aggfunc="first")
values = values.reindex(index=present.index, columns=present.columns)
pivot = values.where(present, 0.0)
pivot.columns = [f"casrn_{c}" for c in pivot.columns]

has_missing_pct = pivot.isna().any(axis=1)
n_missing = int(has_missing_pct.sum())
assert n_missing == 8, f"결측 percent formulation 개수가 예상(8)과 다름: {n_missing}"

# --- 2. 성분개수 ---
comp_count = filtered.dropna(subset=["casrn"]).groupby("formulation_id")["casrn"].nunique()
comp_count = comp_count.reindex(pivot.index)

# --- 3. formulation 단위 메타 (unknown_pct, is_unknown_ingredient) ---
meta = filtered.groupby("formulation_id").agg(
    unknown_pct=("unknown_pct", "first"),
    has_unknown_ingredient=("is_unknown_ingredient", "any"),
).reindex(pivot.index)

# --- 4. 조립 ---
final = pivot.copy()
final.insert(0, "comp_count", comp_count)
final.insert(1, "is_single_component", comp_count == 1)
final.insert(2, "has_missing_pct", has_missing_pct)
final.insert(3, "unknown_pct", meta["unknown_pct"])
final.insert(4, "has_unknown_ingredient", meta["has_unknown_ingredient"])

final = final.reset_index().rename(columns={"index": "formulation_id"})

# --- 5. 라벨/그룹 조인 ---
final = final.merge(
    labels[["formulation_id", "final_label", "comp_group_id",
            "is_composition_twin", "internal_conflict_resolved"]],
    on="formulation_id", how="left", validate="one_to_one",
)
assert final["final_label"].isna().sum() == 0, "라벨 조인 후 결측 발생"

# --- 6. binary_label ---
final["binary_label"] = final["final_label"].map(lambda x: 1 if x in (1, 2) else 0)

# 컬럼 순서: 메타 -> 라벨 -> casrn 벡터
meta_cols = ["formulation_id", "comp_count", "is_single_component", "has_missing_pct",
             "unknown_pct", "has_unknown_ingredient", "comp_group_id",
             "is_composition_twin", "internal_conflict_resolved",
             "final_label", "binary_label"]
casrn_cols = [c for c in final.columns if c.startswith("casrn_")]
final = final[meta_cols + casrn_cols]

# --- 7. 저장 (baseline 전체) ---
out_path = rf"{BASE}\skin_vectorized.csv"
final.to_csv(out_path, index=False)

# --- 8. 다성분 서브셋 ---
multi = final[final["comp_count"] > 1].copy()
multi_path = rf"{BASE}\skin_vectorized_multionly.csv"
multi.to_csv(multi_path, index=False)

print("=== 완료 ===")
print("baseline:", final.shape, "->", out_path)
print("multi-only:", multi.shape, "->", multi_path)
print()
print("comp_count 분포:")
print(final["comp_count"].value_counts().sort_index())
print()
print("final_label 분포:")
print(final["final_label"].value_counts().sort_index())
print()
print("binary_label 분포:")
print(final["binary_label"].value_counts())
print()
print("has_missing_pct True 개수:", final["has_missing_pct"].sum())
print("comp_group_id 결측:", final["comp_group_id"].isna().sum())
