"""현재 설정 vs B4 후보들을 '같은 분할·같은 초기화 시드'로 짝지어 비교(paired) — 20회 반복."""
import os
import sys
HERE2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bharness as B

ds = B.load_skin()
N = 20
KINDS = ["all", "nz20", "nz5", "pca16"]

store = {k: [] for k in KINDS}
for rep in range(N):
    for k in KINDS:
        r = B.run_cv(ds, k, random_state=rep, seed_offset=100 * rep)
        res = B.summarize(ds.y, r["oof"], 0.5)
        res["dim"] = int(np.mean(r["dims"]))
        res["stop_epoch_mean"] = float(np.mean(r["stop_epochs"]))
        store[k].append(res)
    sys.stdout.write(f"\rrep {rep+1}/{N}")
    sys.stdout.flush()
print()

import json
with open(os.path.join(HERE2, "final_results.json"), "w") as f:
    json.dump(store, f)

print(f"\n같은 분할/시드를 공유한 {N}회 반복, threshold 0.5\n")
for k in KINDS:
    a = B.agg(store[k])
    dim = int(np.mean([r["dim"] for r in store[k]]))
    print(B.fmt_agg(a, k), f"  dim={dim}")

print("\n현재(all) 대비 짝지은 차이 — 회차별 차이의 평균/표준편차와 개선 회차 수")
for k in KINDS[1:]:
    print(f"\n  {k} - all")
    for key, name in [("sensitivity", "Sens"), ("specificity", "Spec"),
                      ("gap", "격차"), ("mcc", "MCC"), ("roc_auc", "AUC")]:
        d = np.array([store[k][i][key] for i in range(N)]) - np.array([store["all"][i][key] for i in range(N)])
        better = int((d < 0).sum()) if key == "gap" else int((d > 0).sum())
        # 평균차 / 표준오차 비율로 신호 세기 가늠
        t = d.mean() / (d.std(ddof=1) / np.sqrt(N)) if d.std(ddof=1) > 0 else 0.0
        print(f"    {name:<5} 차이 {d.mean():+.3f} (sd {d.std(ddof=1):.3f}, t={t:+.2f}) "
              f"유리한 회차 {better}/{N}")
