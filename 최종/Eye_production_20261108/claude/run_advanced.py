"""Run the second round of eye experiments (07-10) and cache every result.

Each stage writes its own file, so a partial run still leaves usable results,
and a failing stage does not stop the ones after it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advanced_utils as A  # noqa: E402


# 09는 사용자 요청으로 중단/제외 (606행 한정이라 다른 단계와 비교 불가, 2시간 소요)
SKIP = {"09_fingerprint_families"}


def stage(name):
    def wrap(fn):
        if name in SKIP:
            print(f"[제외] {name} (요청으로 생략)", flush=True)
            return
        path = A.RESULTS / f"{name}.json"
        if path.exists():
            print(f"[건너뜀] {name} (이미 있음)", flush=True)
            return
        print(f"[실행] {name}", flush=True)
        t0 = time.time()
        try:
            value = fn()
        except Exception as exc:
            import traceback
            print(f"[실패] {name} {time.time() - t0:.0f}초: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
            return
        payload = value.to_dict("records") if isinstance(value, pd.DataFrame) else value
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=A.u._plain),
            encoding="utf-8")
        print(f"[완료] {name} {time.time() - t0:.0f}초", flush=True)

    return wrap


def main():
    df, M, cols = A.E.load_eye()
    grades = df.final_label.to_numpy()
    labels = A.E.labels_of(df)
    groups = A.E.merged_groups(df, M)
    A.register_matrix("base", M)

    # ------------------------------------------------ 07 표현 결합
    @stage("07_combined_representation")
    def _():
        combined, covered, multi = A.combined_representation(df, M)
        A.register_matrix("combined", combined)
        (A.RESULTS / "07_coverage.json").write_text(json.dumps({
            "n_features": int(combined.shape[1]),
            "rows_with_component_provenance": int(covered.sum()),
            "multi_component_rows": int(multi.sum()),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        jobs = []
        for label, y in labels.items():
            for seed in A.SEEDS[:2]:
                jobs += A.build_jobs(
                    {"표현": "결합(농도가중+최악성분)", "label": label, "seed": seed},
                    "combined", y, groups, grades, A.MODELS, seed)
        print(f"   작업 {len(jobs)}개", flush=True)
        return A.collect(A.execute(jobs, progress=30), len(df))

    # ------------------------------------------------ 08 확률 보정
    @stage("08_calibration")
    def _():
        jobs = []
        for kind, name in ((None, "보정 없음"), ("platt", "Platt"), ("isotonic", "Isotonic")):
            for label, y in labels.items():
                for seed in A.SEEDS:
                    jobs += A.build_jobs(
                        {"보정": name, "label": label, "seed": seed},
                        "base", y, groups, grades, ["svm", "lr", "lgbm"], seed,
                        calibrate=kind)
        print(f"   작업 {len(jobs)}개", flush=True)
        return A.collect(A.execute(jobs, progress=45), len(df))

    # ------------------------------------- 09 다른 fingerprint 계열
    @stage("09_fingerprint_families")
    def _():
        blocks, covered = A.fingerprint_families(df)
        rows = np.where(covered)[0]
        desc = np.asarray(M[:, 2048:])
        sub_groups = groups[rows]
        sub_grades = grades[rows]
        sub_labels = {k: v[rows] for k, v in labels.items()}
        meta = {"rows_used": int(len(rows)), "rows_dropped": int((~covered).sum()),
                "families": {}}
        names = []
        for family, block in blocks.items():
            matrix = np.hstack([block[rows], desc[rows]])
            tag = f"fam_{family}"
            A.register_matrix(tag, matrix)
            names.append((family, tag, int(matrix.shape[1])))
            meta["families"][family] = int(block.shape[1])
        (A.RESULTS / "09_coverage.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        jobs = []
        for family, tag, width in names:
            for label, y in sub_labels.items():
                for seed in A.SEEDS[:2]:
                    jobs += A.build_jobs(
                        {"계열": family, "label": label, "seed": seed, "n_features": width},
                        tag, y, sub_groups, sub_grades, ["lr", "lgbm", "xgb"], seed)
        print(f"   작업 {len(jobs)}개 (평가 {len(rows)}행)", flush=True)
        return A.collect(A.execute(jobs, progress=60), len(rows))

    # -------------------------- 10 OOF 수집 + 접근법 교차 앙상블
    @stage("10_cross_ensemble")
    def _():
        binary_models = ["lr", "rf", "lgbm", "xgb"]
        jobs = []
        for label, y in labels.items():
            for seed in A.SEEDS:
                jobs += A.build_jobs({"label": label, "seed": seed}, "base", y,
                                     groups, grades, binary_models, seed)
        print(f"   이진 작업 {len(jobs)}개", flush=True)
        binary = A.collect(A.execute(jobs, progress=30), len(df), keep_oof=True)

        ojobs = []
        for seed in A.SEEDS:
            ojobs += A.build_ordinal_jobs({"seed": seed}, "base", grades, groups,
                                          ["lr", "lgbm"], seed)
        print(f"   순서형 작업 {len(ojobs)}개", flush=True)
        ordinal = A.collect(A.execute_ordinal(ojobs, progress=10), len(df), keep_oof=True)

        table = pd.concat([binary, ordinal], ignore_index=True)
        table.to_json(A.RESULTS / "10_oof_pool.json", orient="records",
                      force_ascii=False)

        recipes = {
            "이진 4개": binary_models,
            "순서형 2개": ["lr-순서형", "lgbm-순서형"],
            "이진 트리 3개": ["rf", "lgbm", "xgb"],
            "순서형 LR + 이진 트리 3개": ["lr-순서형", "rf", "lgbm", "xgb"],
            "순서형 2개 + 이진 트리 3개": ["lr-순서형", "lgbm-순서형", "rf", "lgbm", "xgb"],
            "전체 6개": binary_models + ["lr-순서형", "lgbm-순서형"],
        }
        rows = []
        for label in labels:
            for seed in A.SEEDS:
                for name, members in recipes.items():
                    m = A.blend(table, label, seed, members)
                    if m:
                        rows.append({"구성": name, "label": label, "seed": seed,
                                     **{k: m[k] for k in A.CORE},
                                     "PR-AUC": m["PR-AUC"]})
                for _, r in table[(table.label == label) & (table.seed == seed)].iterrows():
                    rows.append({"구성": f"단일 {r['model']}", "label": label,
                                 "seed": seed, **{k: r[k] for k in A.CORE},
                                 "PR-AUC": r["PR-AUC"]})
        return pd.DataFrame(rows)

    print("\n모든 단계 완료")


if __name__ == "__main__":
    main()
