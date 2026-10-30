"""B4: feature 선별/차원축소 비교. 모든 변환은 fold 내부 train에서만 fit(= leakage 없음).
비교는 반복 group 5-fold(10회) 평균±표준편차로 — 단일 분할은 변동폭이 커서 순위가 뒤집힌다(B6 결과)."""
import os
import sys
HERE2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import bharness as B

ds = B.load_skin()
N = 10

VARIANTS = [
    ("all",          "전체 2,266차원 (현재)"),
    ("nz5",          "train에서 nonzero<5 컬럼 제거"),
    ("nz20",         "train에서 nonzero<20 컬럼 제거"),
    ("desc_only",    "descriptor만 (225)"),
    ("fp_only",      "Morgan FP만 (2,048)"),
    ("kbest200",     "f_classif 상위 200"),
    ("kbest100",     "f_classif 상위 100"),
    ("kbest50",      "f_classif 상위 50"),
    ("desc_kbest50", "descriptor 중 상위 50"),
    ("pca64",        "PCA 64성분"),
    ("pca32",        "PCA 32성분"),
    ("pca16",        "PCA 16성분"),
]

print(f"B4: feature 표현별 성능 — 반복 group 5-fold {N}회, threshold 0.5, A 채택 설정(층1/do0/lr0.01)\n")
out = {}
for kind, desc in VARIANTS:
    rows = B.repeat_cv(ds, kind, n_repeats=N)
    out[kind] = rows
    a = B.agg(rows)
    dim = int(np.mean([r["dim"] for r in rows]))
    ep = np.mean([r["stop_epoch_mean"] for r in rows])
    print(B.fmt_agg(a, f"{kind}"), f" dim={dim:<5} 종료ep={ep:.1f}   {desc}")
    sys.stdout.flush()

with open(os.path.join(HERE2, "b4_results.json"), "w") as f:
    json.dump(out, f)

print("\n" + "=" * 110)
print("MCC 기준 정렬:")
for kind, rows in sorted(out.items(), key=lambda kv: -np.mean([r["mcc"] for r in kv[1]])):
    m = [r["mcc"] for r in rows]
    auc = [r["roc_auc"] for r in rows]
    print(f"  {kind:<14} MCC {np.mean(m):.3f}±{np.std(m):.3f}   AUC {np.mean(auc):.3f}±{np.std(auc):.3f}"
          f"   dim={int(np.mean([r['dim'] for r in rows]))}")
