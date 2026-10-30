import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bharness as B

ds = B.load_skin()
print(f"skin: {len(ds)}건, {ds.input_dim}차원, 양성 {int(ds.y.sum())}건")
r = B.run_cv(ds, "all", random_state=0)
res = B.summarize(ds.y, r["oof"])
print(B.fmt(res, "재현 all(2266) rs=0"))
print(f"  혼동행렬 TN={res['tn']} FP={res['fp']} FN={res['fn']} TP={res['tp']}  종료epoch {r['stop_epochs']}")
print("A 보고치               Sens 0.667 / Spec 0.812 / 격차 0.145 / MCC 0.267 / ROC-AUC 0.733")
