"""11. 농도 복원 실험 - 방안 1(농도 피처)과 방안 2(희석 스케일링)를 실제로 측정한다.

문제: 단일성분 제형에서 농도 가중 평균이 약분되어(p*v/p = v) 농도가 사라진다.
평가 606행 중 217행이 "같은 입력, 다른 정답" 상태가 된다.

방안 1  농도를 피처 4개로 추가한다 (벡터는 그대로).
방안 2  정규화를 빼서 희석을 벡터에 반영한다.
        sum(p_i/100 * v_i) = (sum(p_i)/100) * 현재벡터
        따라서 행마다 pct_sum_used/100을 곱하면 된다. 성분 테이블이 필요 없다.
방안 1+2  둘 다.

기준 팔은 2부 실험 03의 "완전일치벡터 병합" 결과를 그대로 쓴다(같은 분할, 같은 격자).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from concentration_features import concentration_block, fit_concentration_median

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advanced_utils as A  # noqa: E402

MODELS = ["lr", "svm", "rf", "lgbm", "xgb"]  # et는 비용 19로 가장 비싸고 이긴 적이 없다
SEEDS = A.SEEDS[:2]


def dilution_scale(df):
    """방안 2: 행마다 곱할 희석 계수 sum(p_i)/100."""
    s = df.pct_sum_used.to_numpy(dtype=np.float64)
    s = np.where(np.isnan(s), df.known_pct.to_numpy(dtype=np.float64), s)
    s = np.where(np.isnan(s), 100.0, s)
    return (s / 100.0)[:, None]


def main():
    df, M, cols = A.E.load_eye()
    M = np.asarray(M, dtype=np.float64)
    grades = df.final_label.to_numpy()
    labels = A.E.labels_of(df)
    groups = A.E.merged_groups(df, M)

    # Historical exploratory protocol only. Production validation fits medians
    # within each training fold in run_concentration_production.py.
    conc, conc_names, pct = concentration_block(df, fit_concentration_median(df))
    scale = dilution_scale(df)

    arms = {
        "② 농도 피처 추가": np.hstack([M, conc]),
        "③ 희석 스케일링": M * scale,
        "④ 농도 피처 + 희석": np.hstack([M * scale, conc]),
    }

    meta = {
        "n_rows": int(len(df)),
        "base_features": int(M.shape[1]),
        "concentration_features": conc_names,
        "pct_range": [float(pct.min()), float(pct.max())],
        "scale_range": [float(scale.min()), float(scale.max())],
        "arms": {k: int(v.shape[1]) for k, v in arms.items()},
        "models": MODELS,
        "seeds": [int(s) for s in SEEDS],
    }

    # 희석 스케일링이 실제로 중복을 깨는지 먼저 확인한다. 안 깨지면 돌릴 이유가 없다.
    import hashlib

    def dup_conflict(X):
        keys = [
            hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in X
        ]
        s = pd.Series(keys)
        counts = s.value_counts()
        dup = counts[counts > 1].index
        frame = pd.DataFrame({"k": keys, "y": grades})
        conflict = [k for k in dup if frame[frame.k == k].y.nunique() > 1]
        return int(counts[counts > 1].sum()), int(frame.k.isin(conflict).sum())

    meta["duplicate_check"] = {
        "① 기준": dup_conflict(M),
        **{k: dup_conflict(v) for k, v in arms.items()},
    }
    (A.RESULTS / "11_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("중복행 / 라벨충돌행:", flush=True)
    for k, v in meta["duplicate_check"].items():
        print(f"   {k:20} 중복 {v[0]:3}행, 충돌 {v[1]:3}행", flush=True)

    path = A.RESULTS / "11_concentration.json"
    if path.exists():
        print("[건너뜀] 11_concentration (이미 있음)", flush=True)
        return

    jobs = []
    for i, (tag, matrix) in enumerate(arms.items()):
        name = f"c11_arm{i}"
        (A.MATS / f"{name}.npy").unlink(missing_ok=True)  # 과거 실행의 찌꺼기 방지
        A.register_matrix(name, matrix)
        for label, y in labels.items():
            for seed in SEEDS:
                jobs += A.build_jobs(
                    {
                        "표현": tag,
                        "label": label,
                        "seed": seed,
                        "n_features": int(matrix.shape[1]),
                    },
                    name,
                    y,
                    groups,
                    grades,
                    MODELS,
                    seed,
                )
    print(f"[실행] 11_concentration / 작업 {len(jobs)}개", flush=True)
    t0 = time.time()
    table = A.collect(A.execute(jobs, progress=30), len(df))
    path.write_text(
        json.dumps(
            table.to_dict("records"), ensure_ascii=False, indent=2, default=A.u._plain
        ),
        encoding="utf-8",
    )
    print(f"[완료] 11_concentration {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
