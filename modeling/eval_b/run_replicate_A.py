import os
"""A의 notebook 함수를 한 글자도 바꾸지 않고 복제해, 환경/시드만 다를 때 수치가 얼마나 흔들리는지 본다."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef

from datasets import skin_dataset
from splits import grouped_kfold_splits
from train import class_weights, EarlyStopping

skin_ds = skin_dataset(bharness.SKIN_CSV)


class SkinOnlyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, num_hidden_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_early_stopping(X_train, y_train, X_test, seed, hidden_dim=128, lr=0.01,
                       num_hidden_layers=2, dropout=0.2,
                       max_epochs=200, patience=15, batch_size=32, val_size=0.2):
    rng_split = np.random.default_rng(seed)
    n = len(X_train)
    idx = rng_split.permutation(n)
    n_val = max(1, int(n * val_size))
    val_pos, train_pos = idx[:n_val], idx[n_val:]

    torch.manual_seed(seed)
    model = SkinOnlyModel(X_train.shape[1], hidden_dim=hidden_dim,
                          num_hidden_layers=num_hidden_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train[train_pos]))

    Xt = torch.from_numpy(X_train[train_pos].astype(np.float32))
    yt = torch.from_numpy(y_train[train_pos].astype(np.int64))
    Xv = torch.from_numpy(X_train[val_pos].astype(np.float32))
    yv = torch.from_numpy(y_train[val_pos].astype(np.int64))

    stopper = EarlyStopping(patience=patience)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[b]), yt[b])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        if stopper.step(val_loss, epoch, model):
            break
    model.load_state_dict(stopper.best_state)

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.from_numpy(X_test.astype(np.float32))), dim=1)[:, 1].numpy()
    return proba, stopper.best_epoch + 1


def run_cv(num_hidden_layers, dropout, lr=0.01, hidden_dim=128, n_splits=5, random_state=0):
    folds = grouped_kfold_splits(skin_ds, n_splits=n_splits, random_state=random_state)
    oof = np.zeros(len(skin_ds.y))
    stop_epochs = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        scaler = StandardScaler().fit(skin_ds.X[train_idx])
        X_train = scaler.transform(skin_ds.X[train_idx])
        X_test = scaler.transform(skin_ds.X[test_idx])
        proba, ep = fit_early_stopping(X_train, skin_ds.y[train_idx], X_test, seed=fold_i,
                                       hidden_dim=hidden_dim, lr=lr,
                                       num_hidden_layers=num_hidden_layers, dropout=dropout)
        oof[test_idx] = proba
        stop_epochs.append(ep)
    return oof, stop_epochs


def summarize(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                mcc=matthews_corrcoef(y_true, pred), roc_auc=roc_auc_score(y_true, proba))


print("A의 코드 그대로, A가 보고한 네 단계 설정을 각각 재현 (rs=0)\n")
settings = [
    ("1단계 전(층2/do0.2/lr0.001)", dict(num_hidden_layers=2, dropout=0.2, lr=0.001)),
    ("2단계 후(층2/do0.2/lr0.01)", dict(num_hidden_layers=2, dropout=0.2, lr=0.01)),
    ("3단계 후(층1/do0.0/lr0.01)", dict(num_hidden_layers=1, dropout=0.0, lr=0.01)),
]
for label, kw in settings:
    oof, eps = run_cv(**kw)
    r = summarize(skin_ds.y, oof)
    print(f"{label:<28} Sens {r['sensitivity']:.3f} / Spec {r['specificity']:.3f} / "
          f"격차 {r['gap']:.3f} / MCC {r['mcc']:.3f} / ROC-AUC {r['roc_auc']:.3f} / 종료ep {eps}")

print("\nA 보고치 참고")
print("  1단계 후(early stopping)  Sens 0.567 / Spec 0.847 / 격차 0.281 / MCC 0.249 / ROC-AUC 0.762")
print("  2단계 후(lr 0.01)         Sens 0.633 / Spec 0.800 / 격차 0.167 / MCC 0.238 / ROC-AUC 0.778")
print("  3단계 후(층1/do0)         Sens 0.667 / Spec 0.812 / 격차 0.145 / MCC 0.267 / ROC-AUC 0.733")
