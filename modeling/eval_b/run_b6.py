import os
"""B6: 반복 group 5-fold로 A의 1~3단계 개선이 변동폭 안에서 구분되는지 검증."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bharness as B

ds = B.load_skin()
N = 10

CONFIGS = [
    ("1단계 전 (층2/do0.2/lr0.001)", dict(B.BASE, num_hidden_layers=2, dropout=0.2, lr=0.001)),
    ("1단계 후 = 2단계 전",          dict(B.BASE, num_hidden_layers=2, dropout=0.2, lr=0.001)),
    ("2단계 후 (lr 0.01)",           dict(B.BASE, num_hidden_layers=2, dropout=0.2, lr=0.01)),
    ("3단계 후 (층1/do0.0) = 채택",  dict(B.BASE, num_hidden_layers=1, dropout=0.0, lr=0.01)),
]

print(f"반복 group 5-fold OOF, {N}회 반복 (분할 시드 + 초기화 시드 동시 변경), threshold 0.5\n")
store = {}
for label, cfg in CONFIGS[1:]:          # 1단계 전/후는 같은 설정이라 중복 제거
    rows = B.repeat_cv(ds, "all", n_repeats=N, cfg=cfg)
    store[label] = rows
    a = B.agg(rows)
    print(B.fmt_agg(a, label))
    print(f"{'':24} MCC 회차별: {[round(r['mcc'], 3) for r in rows]}")
    print(f"{'':24} 평균 종료 epoch: {np.mean([r['stop_epoch_mean'] for r in rows]):.1f}\n")

print("=" * 100)
print("A가 보고한 단계별 차이 (단일 분할 1회):")
print("  2단계: MCC 0.249 -> 0.238 (-0.011),  3단계: MCC 0.238 -> 0.267 (+0.029)")
print()
keys = list(store)
for k in keys:
    mccs = [r["mcc"] for r in store[k]]
    print(f"  {k:<28} MCC {np.mean(mccs):.3f} ± {np.std(mccs):.3f}  (최소 {min(mccs):.3f} ~ 최대 {max(mccs):.3f})")
print()
a, b = store[keys[-2]], store[keys[-1]]
d = np.array([x["mcc"] for x in b]) - np.array([x["mcc"] for x in a])
print(f"  2단계 후 -> 3단계 후 회차별 MCC 차이: 평균 {d.mean():+.3f}, 표준편차 {d.std():.3f}, "
      f"개선된 회차 {int((d > 0).sum())}/{N}")
