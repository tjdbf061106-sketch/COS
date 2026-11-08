"""Compare cached experiment 03 editions without retraining or changing inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def compare(first, second):
    keys = ["grouping", "label", "model", "seed"]
    if first.duplicated(keys).any() or second.duplicated(keys).any():
        raise ValueError("Duplicate experiment cells")
    expected = 2 * 2 * 6 * 3
    if len(first) != expected or len(second) != expected:
        raise ValueError("Incomplete experiment: expected 72 cells per edition")
    matched = first.merge(
        second, on=keys, validate="one_to_one", suffixes=("_v1", "_v2")
    )
    if (
        len(matched) != expected
        or not np.isfinite(matched[["MCC_v1", "MCC_v2"]]).all().all()
    ):
        raise ValueError("Missing or nonfinite experiment metrics")
    baseline = matched[matched.grouping.eq("기존 그룹화")]
    if not np.allclose(baseline.MCC_v1, baseline.MCC_v2, atol=1e-12, rtol=0):
        raise ValueError(
            "Original grouping baseline changed; isolate this before interpreting delta"
        )
    rows = []
    for label in ["ALL", "A", "B"]:
        for edition, frame in [("v1", first), ("v2", second)]:
            subset = frame if label == "ALL" else frame[frame.label.eq(label)]
            paired = subset.pivot(
                index=["label", "model", "seed"], columns="grouping", values="MCC"
            )
            delta = paired["완전일치벡터 병합"] - paired["기존 그룹화"]
            rows.append(
                {
                    "edition": edition,
                    "label": label,
                    "original_MCC": float(paired["기존 그룹화"].mean()),
                    "merged_MCC": float(paired["완전일치벡터 병합"].mean()),
                    "delta_MCC": float(delta.mean()),
                    "paired_cells": len(delta),
                }
            )
    table = pd.DataFrame(rows)
    delta = float(
        table.loc[(table.edition == "v2") & (table.label == "ALL"), "delta_MCC"].iloc[0]
    )
    if abs(delta) < 0.01:
        decision = "그룹화 효과 기준으로 전체 재실행 불필요"
    elif abs(delta) > 0.03:
        decision = "그룹화 효과 기준으로 전체 재실행 필요"
    else:
        decision = "중간 구간: 제공된 기준으로 확정 불가"
    return (
        matched,
        table,
        {
            "delta_MCC": delta,
            "seed_variation_reference": 0.024,
            "decision": decision,
            "evaluation_rows": 673,
            "existing_evaluation_pool": 606,
            "augmentation_rows_in_cv": 67,
            "limitation": "1판과 동일 조건의 그룹화 민감도 비교임. 증강 67행이 CV 평가에 포함되므로 production train-only 검증을 대체하지 못함.",
            "cache_warning": "00~02 및 04~11 기존 JSON은 1판 그룹화 결과이며 2판으로 재계산되지 않았음.",
        },
    )


def main():
    first = pd.read_json(RESULTS / "03_grouping_fix_v1.json")
    second = pd.read_json(RESULTS / "03_grouping_fix.json")
    matched, table, verdict = compare(first, second)
    matched.to_csv(
        RESULTS / "03_grouping_fix_paired_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table.to_csv(
        RESULTS / "03_grouping_fix_edition_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (RESULTS / "03_grouping_fix_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
