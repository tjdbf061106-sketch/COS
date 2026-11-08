"""Run the eye model-strengthening experiments and cache every result.

Heavy work lives here (Windows spawn needs a guarded entry point); the
notebook only loads the cached tables. Each stage writes its own file, so a
partial run still leaves usable results. Nothing here touches the published
production runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_utils as X  # noqa: E402


def stage(name):
    def wrap(fn):
        path = X.RESULTS / f"{name}.json"
        if path.exists():
            print(f"[건너뜀] {name} (이미 있음)")
            return
        print(f"[실행] {name}", flush=True)
        t0 = time.time()
        try:
            value = fn()
        except Exception as exc:          # 한 단계 실패가 나머지를 막지 않게 한다
            import traceback
            print(f"[실패] {name} {time.time() - t0:.0f}초: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
            return
        payload = (
            value.to_dict("records") if isinstance(value, pd.DataFrame) else value
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=X.u._plain),
            encoding="utf-8",
        )
        print(f"[완료] {name} {time.time() - t0:.0f}초", flush=True)

    return wrap


def main():
    df, M, cols = X.load_eye()
    grades = df.final_label.to_numpy()
    labels = X.labels_of(df)
    groups = {"기존 그룹화": X.original_groups(df), "완전일치벡터 병합": X.merged_groups(df, M)}
    X.register_matrix("base", M)

    # ---------------------------------------------------------------- 진단
    @stage("00_diagnostics")
    def _():
        report = X.duplicate_report(df, M)
        report["evaluation_role"] = (
            df.evaluation_role.value_counts(dropna=False).to_dict()
            if "evaluation_role" in df.columns
            else {}
        )
        report["n_components_used"] = (
            df.n_components_used.value_counts(dropna=False).sort_index().to_dict()
        )
        report["weight_mode"] = df.weight_mode.value_counts(dropna=False).to_dict()
        report["epa_class"] = df.final_label.value_counts().sort_index().to_dict()
        return report

    @stage("01_ceiling")
    def _():
        return [X.ceiling_estimate(df, M, key) for key in ("A", "B")]

    @stage("02_ensemble_saved_oof")
    def _():
        out = []
        for label in ("A", "B"):
            table = X.ensemble_from_saved(label)
            table.insert(0, "label", label)
            out.append(table)
        return pd.concat(out, ignore_index=True)

    # ------------------------------------------- 요청 실험: 분할 그룹화 수정
    @stage("03_grouping_fix")
    def _():
        jobs = []
        for name, g in groups.items():
            for label, y in labels.items():
                for seed in X.SEEDS:
                    jobs += X.build_jobs(
                        {"grouping": name, "label": label, "seed": seed},
                        "base", y, g, grades, X.MODELS, seed,
                    )
        print(f"   작업 {len(jobs)}개", flush=True)
        out = X.collect(X.execute(jobs, progress=30), len(df))
        return out.drop(columns=["oof_scores", "oof_y"])

    # ------------------------------------------------------ 4-class 순서형
    @stage("04_ordinal")
    def _():
        jobs = []
        for seed in X.SEEDS:
            jobs += X.build_ordinal_jobs(
                "base", grades, groups["완전일치벡터 병합"], X.MULTICLASS_MODELS, seed
            )
        print(f"   작업 {len(jobs)}개", flush=True)
        out = X.collect(X.execute_ordinal(jobs, progress=15), len(df))
        return out.drop(columns=["oof_scores", "oof_y"])

    # --------------------------------------------------- 최악 성분 표현
    @stage("05_worst_actor")
    def _():
        Xw, covered, multi = X.worst_actor_features(df, M)
        X.register_matrix("worst", Xw)
        (X.RESULTS / "05_worst_actor_coverage.json").write_text(
            json.dumps(
                {
                    "rows_with_component_provenance": int(covered.sum()),
                    "multi_component_rows_changed": int(multi.sum()),
                    "rows_unchanged": int((~covered).sum()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # 기준(농도가중) arm은 03이 같은 행렬, 같은 병합 그룹화, 같은 seed로
        # 이미 계산해 두었으므로 여기서는 최악 성분 표현만 돌린다.
        jobs = []
        for label, y in labels.items():
            for seed in X.SEEDS[:2]:
                jobs += X.build_jobs(
                    {"표현": "최악 성분 pooling", "label": label, "seed": seed},
                    "worst", y, groups["완전일치벡터 병합"], grades, X.MODELS, seed,
                )
        print(f"   작업 {len(jobs)}개", flush=True)
        out = X.collect(X.execute(jobs, progress=30), len(df))
        return out.drop(columns=["oof_scores", "oof_y"])

    # --------------------------------------------------------- 특징 집합
    @stage("06_representations")
    def _():
        sets = X.representation_sets(M, cols)
        for name, matrix in sets.items():
            X.register_matrix(f"rep_{abs(hash(name)) % 10**8}", matrix)
        models = ["lr", "lgbm", "xgb"]
        jobs = []
        for name, matrix in sets.items():
            tag = f"rep_{abs(hash(name)) % 10**8}"
            for label, y in labels.items():
                for seed in X.SEEDS[:2]:
                    jobs += X.build_jobs(
                        {"특징집합": name, "label": label, "seed": seed,
                         "n_features": int(matrix.shape[1])},
                        tag, y, groups["완전일치벡터 병합"], grades, models, seed,
                    )
        print(f"   작업 {len(jobs)}개", flush=True)
        out = X.collect(X.execute(jobs, progress=60), len(df))
        return out.drop(columns=["oof_scores", "oof_y"])

    print("\n모든 단계 완료")


if __name__ == "__main__":
    main()
