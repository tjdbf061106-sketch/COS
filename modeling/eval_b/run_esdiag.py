import os
"""조기종료가 epoch 1~3에서 멈추는 이유 진단: fold 1개의 val loss / test MCC 곡선을 epoch별로 본다."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import bharness as B

ds = B.load_skin()
folds = B.grouped_kfold_splits(ds, n_splits=5, random_state=0)
tr, te = folds[0]

scaler = StandardScaler().fit(ds.X[tr])
Xtr_all, Xte = scaler.transform(ds.X[tr]), scaler.transform(ds.X[te])
ytr_all, yte = ds.y[tr], ds.y[te]

idx = np.random.default_rng(0).permutation(len(Xtr_all))
n_val = max(1, int(len(Xtr_all) * 0.2))
val_pos, train_pos = idx[:n_val], idx[n_val:]

for lr in (0.01, 0.001):
    torch.manual_seed(0)
    model = B.SkinOnlyModel(Xtr_all.shape[1], hidden_dim=128, num_hidden_layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=B.class_weights(ytr_all[train_pos]))

    Xt = torch.from_numpy(Xtr_all[train_pos].astype(np.float32))
    yt = torch.from_numpy(ytr_all[train_pos].astype(np.int64))
    Xv = torch.from_numpy(Xtr_all[val_pos].astype(np.float32))
    yv = torch.from_numpy(ytr_all[val_pos].astype(np.int64))
    Xs = torch.from_numpy(Xte.astype(np.float32))

    print(f"\n=== lr={lr} | val 양성 {int(ytr_all[val_pos].sum())}/{len(val_pos)}건, "
          f"test 양성 {int(yte.sum())}/{len(yte)}건 ===")
    print("epoch  train_loss  val_loss   test_MCC  test_AUC")
    for epoch in range(40):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
            p = torch.softmax(model(Xs), dim=1)[:, 1].numpy()
        mcc = matthews_corrcoef(yte, (p >= 0.5).astype(int))
        try:
            auc = roc_auc_score(yte, p)
        except ValueError:
            auc = float("nan")
        if epoch < 12 or epoch % 5 == 0:
            print(f"{epoch+1:5d}  {tot/max(1,len(Xt)//32):9.4f}  {vl:8.4f}  {mcc:9.3f}  {auc:8.3f}")
