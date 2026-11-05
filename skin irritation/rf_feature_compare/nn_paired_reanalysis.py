# -*- coding: utf-8 -*-
"""내 NN 파이프라인에서도 짝지은(paired) 비교로 feature 종류를 재검증.
repeat_cv(random_state=rep)가 kind별로 동일 seed를 쓰므로 이미 암묵적으로 paired 상태 —
이번엔 그 사실을 이용해 rep별 ROC-AUC 차이를 명시적으로 계산하고 t값을 낸다."""
import sys
sys.path.insert(0, r"C:\Users\UserK\학술제\modeling\eval_b")
sys.path.insert(0, r"C:\Users\UserK\학술제\modeling\src")

import numpy as np
import bharness as B

ds = B.load_skin()
cfg = dict(B.BASE, lr=0.001, num_hidden_layers=1, dropout=0.0)

KINDS = ["desc_only", "fp_only", "nz20", "all"]
N = 10

store = {}
for kind in KINDS:
    aucs = []
    for rep in range(N):
        r = B.run_cv(ds, kind, random_state=rep, cfg=cfg, group_aware_val=True, seed_offset=100 * rep)
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(ds.y, r["oof"]))
    store[kind] = np.array(aucs)
    print(f"{kind:<12} ROC-AUC rep별: {[round(a,3) for a in aucs]}  평균 {aucs.mean() if hasattr(aucs,'mean') else np.mean(aucs):.3f}")

print("\n=== 짝지은 비교 (같은 rep=같은 분할+초기화 시드) — 기준: all ===")
ref = store["all"]
for kind in KINDS:
    if kind == "all":
        continue
    diff = store[kind] - ref
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t = diff.mean() / se if se > 0 else float("nan")
    print(f"{kind:<12} 평균차 {diff.mean():+.4f}  t={t:+.2f}  우세 {int((diff>0).sum())}/{N}")
