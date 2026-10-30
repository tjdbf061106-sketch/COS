"""그림에 필요한 수치를 계산해 figdata.json에 캐싱."""
import os
import sys
HERE2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import bharness as B

ds = B.load_skin()
N = 10
out = {}

# ---- (1) B6: 단계별 반복 CV MCC 분포 + A가 보고한 단일분할 수치
CONFIGS = [
    ("1·2단계 전\n(층2 / do0.2 / lr0.001)", dict(B.BASE, num_hidden_layers=2, dropout=0.2, lr=0.001), 0.249),
    ("2단계 후\n(lr 0.01)", dict(B.BASE, num_hidden_layers=2, dropout=0.2, lr=0.01), 0.238),
    ("3단계 후 = 채택\n(층1 / do0.0)", dict(B.BASE, num_hidden_layers=1, dropout=0.0, lr=0.01), 0.267),
]
stages = []
for label, cfg, a_reported in CONFIGS:
    rows = B.repeat_cv(ds, "all", n_repeats=N, cfg=cfg)
    stages.append(dict(label=label, a_reported=a_reported,
                       mcc=[r["mcc"] for r in rows],
                       sens=[r["sensitivity"] for r in rows],
                       spec=[r["specificity"] for r in rows],
                       gap=[r["gap"] for r in rows]))
    print("stage done:", label.replace("\n", " "))
    sys.stdout.flush()
out["stages"] = stages

# ---- (2) threshold sweep (OOF, 반복 N회)
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
oofs = [B.run_cv(ds, "all", random_state=rep, seed_offset=100 * rep)["oof"] for rep in range(N)]
sweep = {"grid": GRID}
for key in ("sensitivity", "specificity", "mcc"):
    mean, sd = [], []
    for t in GRID:
        vals = [B.summarize(ds.y, o, t)[key] for o in oofs]
        mean.append(float(np.mean(vals))); sd.append(float(np.std(vals)))
    sweep[key] = {"mean": mean, "sd": sd}
out["sweep"] = sweep
print("sweep done"); sys.stdout.flush()

# ---- (3) 조기종료 진단: epoch별 train/val loss, test AUC
folds = B.grouped_kfold_splits(ds, n_splits=5, random_state=0)
tr, te = folds[0]
scaler = StandardScaler().fit(ds.X[tr])
Xtr_all, Xte = scaler.transform(ds.X[tr]), scaler.transform(ds.X[te])
ytr_all, yte = ds.y[tr], ds.y[te]
idx = np.random.default_rng(0).permutation(len(Xtr_all))
n_val = max(1, int(len(Xtr_all) * 0.2))
val_pos, train_pos = idx[:n_val], idx[n_val:]

torch.manual_seed(0)
model = B.SkinOnlyModel(Xtr_all.shape[1], hidden_dim=128, num_hidden_layers=1, dropout=0.0)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss(weight=B.class_weights(ytr_all[train_pos]))
Xt = torch.from_numpy(Xtr_all[train_pos].astype(np.float32))
yt = torch.from_numpy(ytr_all[train_pos].astype(np.int64))
Xv = torch.from_numpy(Xtr_all[val_pos].astype(np.float32))
yv = torch.from_numpy(ytr_all[val_pos].astype(np.int64))
Xs = torch.from_numpy(Xte.astype(np.float32))

curve = {"epoch": [], "train_loss": [], "val_loss": [], "test_auc": []}
for epoch in range(40):
    model.train()
    perm = torch.randperm(len(Xt)); tot = 0.0; nb = 0
    for i in range(0, len(Xt), 32):
        b = perm[i:i + 32]
        opt.zero_grad()
        loss = crit(model(Xt[b]), yt[b]); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    model.eval()
    with torch.no_grad():
        vl = crit(model(Xv), yv).item()
        p = torch.softmax(model(Xs), dim=1)[:, 1].numpy()
    curve["epoch"].append(epoch + 1)
    curve["train_loss"].append(tot / nb)
    curve["val_loss"].append(vl)
    curve["test_auc"].append(float(roc_auc_score(yte, p)))
out["curve"] = curve
print("curve done")

# ---- (4) B4 짝지은 차이 (final_results.json 재사용)
with open(os.path.join(HERE2, "final_results.json")) as f:
    store = json.load(f)
Nf = len(store["all"])
paired = {}
for k in ("nz20", "nz5", "pca16"):
    paired[k] = {}
    for key in ("sensitivity", "specificity", "gap", "mcc", "roc_auc"):
        d = np.array([store[k][i][key] for i in range(Nf)]) - np.array([store["all"][i][key] for i in range(Nf)])
        paired[k][key] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                              favor=int((d < 0).sum() if key == "gap" else (d > 0).sum()), n=Nf)
paired["dims"] = {k: int(np.mean([r["dim"] for r in store[k]])) for k in store}
out["paired"] = paired

with open(os.path.join(HERE2, "figdata.json"), "w") as f:
    json.dump(out, f)
print("saved figdata.json")
