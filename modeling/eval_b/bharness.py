"""
B 파트(입력 feature / 평가 개선) 실험 하네스.

A가 쓴 평가 프로토콜을 그대로 따른다:
  - skin-only MLP, group 5-fold(StratifiedGroupKFold) OOF
  - fold별 StandardScaler(train fit), class-weighted CE, early stopping(patience=15, max 200)
  - A 채택 설정: hidden_dim=128, lr=0.01, 은닉층 1개, dropout 0.0
차이점: feature 변환(차원축소/선별)을 fold 내부 train에서만 fit하고,
threshold도 fold 내부 검증셋에서 고르는 nested 옵션을 제공한다.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.model_selection import GroupShuffleSplit

import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SKIN_CSV = os.path.join(REPO, "skin irritation", "skin_vectorized.csv")
sys.path.insert(0, SRC)
from datasets import skin_dataset          # noqa: E402
from splits import grouped_kfold_splits    # noqa: E402
from train import class_weights, EarlyStopping  # noqa: E402


# A가 1~3단계에서 채택한 설정
BASE = dict(hidden_dim=128, lr=0.01, num_hidden_layers=1, dropout=0.0,
            max_epochs=200, patience=15, batch_size=32, val_size=0.2)


def load_skin():
    return skin_dataset(SKIN_CSV)


class SkinOnlyModel(nn.Module):
    """A의 skin_architecture_demo.ipynb와 동일한 구조(dropout은 첫 은닉층 뒤에만)."""

    def __init__(self, input_dim, hidden_dim=128, num_classes=2, num_hidden_layers=1, dropout=0.0):
        super().__init__()
        layers, in_dim = [], input_dim
        for i in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- feature 변환
class NonZeroFilter:
    """train에서 nonzero 샘플 수가 min_count 미만인 컬럼을 버린다(희소 FP 비트 정리)."""

    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.keep_ = (X != 0).sum(axis=0) >= self.min_count
        if self.keep_.sum() == 0:
            self.keep_[:] = True
        return self

    def transform(self, X):
        return X[:, self.keep_]


class Chain:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for s in self.steps:
            X = s.fit(X, y).transform(X)
        return self

    def transform(self, X):
        for s in self.steps:
            X = s.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class ColumnSubset:
    def __init__(self, mask):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.mask]


def make_transform(kind, feature_cols):
    """kind 이름 -> 변환기. 모두 fold 내부 train에서만 fit된다.
    StandardScaler는 항상 마지막에 둬서 모델 입력 스케일을 A와 동일하게 맞춘다."""
    desc_mask = np.array([c.startswith("desc_") or not c.startswith("fp_w_") for c in feature_cols])
    fp_mask = np.array([c.startswith("fp_w_") for c in feature_cols])

    if kind == "all":
        return Chain([StandardScaler()])
    if kind.startswith("nz"):                     # nz5, nz10, nz20
        return Chain([NonZeroFilter(int(kind[2:])), StandardScaler()])
    if kind == "desc_only":
        return Chain([ColumnSubset(desc_mask), StandardScaler()])
    if kind == "fp_only":
        return Chain([ColumnSubset(fp_mask), StandardScaler()])
    if kind.startswith("kbest"):                  # kbest50, kbest100 ...
        k = int(kind[5:])
        return Chain([NonZeroFilter(5), StandardScaler(), SelectKBest(f_classif, k=k)])
    if kind.startswith("pca"):                    # pca16, pca32 ...
        k = int(kind[3:])
        return Chain([NonZeroFilter(5), StandardScaler(), PCA(n_components=k, random_state=0),
                      StandardScaler()])
    if kind.startswith("desc_kbest"):             # desc_kbest50
        k = int(kind[len("desc_kbest"):])
        return Chain([ColumnSubset(desc_mask), StandardScaler(), SelectKBest(f_classif, k=k)])
    raise ValueError(f"알 수 없는 변환: {kind}")


# ---------------------------------------------------------------- 학습 1회
def fit_one(X_train, y_train, X_eval_list, seed, groups_train=None,
            group_aware_val=False, cfg=BASE):
    """X_eval_list의 각 배열에 대한 양성 확률을 반환. groups_train을 주고
    group_aware_val=True면 early stopping 검증셋도 group 단위로 분리한다."""
    n = len(X_train)
    if group_aware_val and groups_train is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["val_size"], random_state=seed)
        train_pos, val_pos = next(gss.split(np.arange(n), groups=groups_train))
    else:
        idx = np.random.default_rng(seed).permutation(n)
        n_val = max(1, int(n * cfg["val_size"]))
        val_pos, train_pos = idx[:n_val], idx[n_val:]

    # 검증셋에 양성이 하나도 없으면 val loss가 조기종료 신호로 쓸모없어짐 -> 재분할
    if y_train[val_pos].sum() == 0 or y_train[train_pos].sum() == 0:
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        rng = np.random.default_rng(seed + 1000)
        n_val_pos = max(1, int(len(pos) * cfg["val_size"]))
        vp = rng.choice(pos, n_val_pos, replace=False)
        vn = rng.choice(neg, max(1, int(len(neg) * cfg["val_size"])), replace=False)
        val_pos = np.concatenate([vp, vn])
        train_pos = np.setdiff1d(np.arange(n), val_pos)

    torch.manual_seed(seed)
    model = SkinOnlyModel(X_train.shape[1], hidden_dim=cfg["hidden_dim"],
                          num_hidden_layers=cfg["num_hidden_layers"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss(weight=class_weights(y_train[train_pos]))

    Xt = torch.from_numpy(X_train[train_pos].astype(np.float32))
    yt = torch.from_numpy(y_train[train_pos].astype(np.int64))
    Xv = torch.from_numpy(X_train[val_pos].astype(np.float32))
    yv = torch.from_numpy(y_train[val_pos].astype(np.int64))

    stopper = EarlyStopping(patience=cfg["patience"])
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            if stopper.step(crit(model(Xv), yv).item(), epoch, model):
                break
    model.load_state_dict(stopper.best_state)

    model.eval()
    out = []
    with torch.no_grad():
        for Xe in X_eval_list:
            p = torch.softmax(model(torch.from_numpy(Xe.astype(np.float32))), dim=1)[:, 1].numpy()
            out.append(p)
    # 조기종료 시점 + 검증셋 인덱스(threshold를 fold 내부에서 고를 때 필요)
    return out, stopper.best_epoch + 1, np.asarray(val_pos)


# ---------------------------------------------------------------- CV
def run_cv(ds, kind="all", random_state=0, n_splits=5, cfg=BASE,
           group_aware_val=False, want_val_proba=False, seed_offset=0):
    """group 5-fold OOF 확률을 모은다. seed_offset으로 가중치 초기화 시드까지 바꿀 수 있다.
    want_val_proba면 fold별 (train 전체에 대한 확률, train 라벨, 검증셋 위치)도 반환."""
    folds = grouped_kfold_splits(ds, n_splits=n_splits, random_state=random_state)
    groups = np.asarray(ds.group_id)
    oof = np.zeros(len(ds.y))
    stop_epochs, dims, val_packs = [], [], []

    for fold_i, (tr, te) in enumerate(folds):
        tf = make_transform(kind, ds.feature_cols)
        Xtr = tf.fit_transform(ds.X[tr], ds.y[tr])
        Xte = tf.transform(ds.X[te])
        dims.append(Xtr.shape[1])

        # 검증셋 확률까지 받으려면 train 자체도 eval 대상에 넣는다
        evals = [Xte, Xtr] if want_val_proba else [Xte]
        probs, ep, val_idx = fit_one(Xtr, ds.y[tr], evals, seed=fold_i + seed_offset,
                                     groups_train=groups[tr], group_aware_val=group_aware_val, cfg=cfg)
        oof[te] = probs[0]
        stop_epochs.append(ep)
        if want_val_proba:
            # (fold 내부 검증셋 확률, 그 라벨, 이 fold의 test 확률, test 라벨)
            val_packs.append((probs[1][val_idx], ds.y[tr][val_idx], probs[0], ds.y[te]))

    return dict(oof=oof, stop_epochs=stop_epochs, dims=dims, val_packs=val_packs)


def repeat_cv(ds, kind="all", n_repeats=10, cfg=BASE, group_aware_val=False, threshold=0.5):
    """random_state(분할)와 초기화 시드를 함께 바꿔 n_repeats회 반복. 각 회차 지표와 평균±표준편차."""
    rows = []
    for rep in range(n_repeats):
        r = run_cv(ds, kind, random_state=rep, cfg=cfg,
                   group_aware_val=group_aware_val, seed_offset=100 * rep)
        res = summarize(ds.y, r["oof"], threshold)
        res["stop_epoch_mean"] = float(np.mean(r["stop_epochs"]))
        res["dim"] = int(np.mean(r["dims"]))
        rows.append(res)
    return rows


def agg(rows, keys=("sensitivity", "specificity", "gap", "mcc", "roc_auc")):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}


def fmt_agg(a, label="", n=None):
    s = f"{label:<24} " + " ".join(
        f"{name} {a[k][0]:.3f}±{a[k][1]:.3f}"
        for k, name in [("sensitivity", "Sens"), ("specificity", "Spec"),
                        ("gap", "격차"), ("mcc", "MCC"), ("roc_auc", "AUC")])
    return s + (f"  (n={n})" if n else "")


def summarize(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(sensitivity=sens, specificity=spec, gap=abs(sens - spec),
                mcc=matthews_corrcoef(y_true, pred), f1=f1_score(y_true, pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, proba), threshold=threshold,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def fmt(res, label=""):
    return (f"{label:<22} Sens {res['sensitivity']:.3f} / Spec {res['specificity']:.3f} / "
            f"격차 {res['gap']:.3f} / MCC {res['mcc']:.3f} / ROC-AUC {res['roc_auc']:.3f}")
