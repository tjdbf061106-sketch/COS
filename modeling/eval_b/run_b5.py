import os
"""B5: 분류 threshold 조정.

두 가지를 나눠서 본다.
 (A) OOF 확률 위에서 threshold를 쓸어보기 — 트레이드오프 곡선 확인용.
     단, 여기서 고른 threshold를 그대로 성능으로 보고하면 같은 데이터에 맞춘 낙관적 수치가 된다.
 (B) nested: threshold를 fold 내부 검증셋에서만 고르고 그 fold의 test에 적용 — 편향 없는 추정.
A와 B의 차이가 곧 "threshold 튜닝으로 생기는 낙관 편향"의 크기다.
"""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bharness as B

ds = B.load_skin()
N = 10
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
KINDS = ["all", "nz20"]


def pick(proba, y, criterion):
    """검증셋에서 threshold를 고른다."""
    best, best_t = -9e9, 0.5
    for t in GRID:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if criterion == "youden":
            score = sens + spec - 1
        elif criterion == "mcc":
            from sklearn.metrics import matthews_corrcoef
            score = matthews_corrcoef(y, pred)
        elif criterion == "sens80":
            # Sens 0.8 이상을 만족하는 것 중 Spec 최대, 없으면 Sens 최대
            score = (spec if sens >= 0.80 else -1 + sens)
        if score > best:
            best, best_t = score, t
    return float(best_t)


print(f"B5-(A) OOF threshold sweep — A 채택 설정, 전체 2,266차원, 반복 {N}회 평균\n")
oofs = []
for rep in range(N):
    r = B.run_cv(ds, "all", random_state=rep, seed_offset=100 * rep)
    oofs.append(r["oof"])

print(" thr   Sens          Spec          격차          MCC           F1")
for t in GRID:
    rows = [B.summarize(ds.y, o, t) for o in oofs]
    g = lambda k: (np.mean([x[k] for x in rows]), np.std([x[k] for x in rows]))
    s, sp, gp, mc, f1 = g("sensitivity"), g("specificity"), g("gap"), g("mcc"), g("f1")
    mark = "  <- 현재" if abs(t - 0.5) < 1e-9 else ""
    print(f"{t:5.2f}  {s[0]:.3f}±{s[1]:.3f}  {sp[0]:.3f}±{sp[1]:.3f}  {gp[0]:.3f}±{gp[1]:.3f}  "
          f"{mc[0]:.3f}±{mc[1]:.3f}  {f1[0]:.3f}±{f1[1]:.3f}{mark}")

for crit in ("youden", "mcc"):
    best_per_rep = [GRID[np.argmax([
        (B.summarize(ds.y, o, t)["sensitivity"] + B.summarize(ds.y, o, t)["specificity"] - 1)
        if crit == "youden" else B.summarize(ds.y, o, t)["mcc"] for t in GRID])] for o in oofs]
    print(f"\nOOF에서 {crit} 최대가 되는 threshold: 회차별 {list(np.round(best_per_rep,2))} "
          f"-> 평균 {np.mean(best_per_rep):.2f}, 표준편차 {np.std(best_per_rep):.2f}")

print("\n" + "=" * 100)
print(f"B5-(B) nested threshold (fold 내부 검증셋에서 고르고 그 fold test에 적용), 반복 {N}회\n")

for kind in KINDS:
    for crit in ("fixed0.5", "youden", "mcc", "sens80"):
        reps = []
        thr_used = []
        for rep in range(N):
            r = B.run_cv(ds, kind, random_state=rep, seed_offset=100 * rep, want_val_proba=True)
            y_all, p_all = [], []
            for vp, vy, tp_, ty in r["val_packs"]:
                t = 0.5 if crit == "fixed0.5" else pick(vp, vy, crit)
                thr_used.append(t)
                y_all.append(ty)
                p_all.append((tp_ >= t).astype(int))
            y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
            tp = int(((p_all == 1) & (y_all == 1)).sum()); fn = int(((p_all == 0) & (y_all == 1)).sum())
            tn = int(((p_all == 0) & (y_all == 0)).sum()); fp = int(((p_all == 1) & (y_all == 0)).sum())
            sens, spec = tp / (tp + fn), tn / (tn + fp)
            from sklearn.metrics import matthews_corrcoef
            reps.append(dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                             mcc=matthews_corrcoef(y_all, p_all), roc_auc=np.nan))
        m = lambda k: (np.mean([x[k] for x in reps]), np.std([x[k] for x in reps]))
        s, sp, gp, mc = m("sensitivity"), m("specificity"), m("gap"), m("mcc")
        print(f"{kind:<6} {crit:<9} Sens {s[0]:.3f}±{s[1]:.3f}  Spec {sp[0]:.3f}±{sp[1]:.3f}  "
              f"격차 {gp[0]:.3f}±{gp[1]:.3f}  MCC {mc[0]:.3f}±{mc[1]:.3f}  "
              f"고른 thr 평균 {np.mean(thr_used):.2f}(sd {np.std(thr_used):.2f})")
        sys.stdout.flush()
