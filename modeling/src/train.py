"""
Multi-task skin/eye 독성(0/1) 분류 학습.

사용법 (어느 디렉터리에서 실행해도 기본 CSV 경로는 레포 루트 기준으로 잡힌다):
  py train.py --epochs 50

클래스 불균형 대응 순서(진행상황.md 2026-09-18 합의):
  1순위: class-weighted CrossEntropyLoss (기본값, --balance weighted)
  최후수단: 1:1 언더샘플링 (--balance 1to1) — skin처럼 양성이 29건뿐이면 데이터 손실이 커서 권장 안 함
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from datasets import EYE_CSV, SKIN_CSV, skin_dataset, eye_dataset
from model import MultiTaskIrritationModel
from splits import group_train_test_split, group_train_test_split_from_subset


def standardize_inplace(dataset, train_idx):
    """train split 기준으로 평균0/분산1 정규화 — Linear 입력이 원시 descriptor 스케일(MolWt 수백~수천 등)
    그대로 들어가면 logit이 발산하는 문제(실행해보고 처음 발견, 2026-10-29)를 막기 위함."""
    scaler = StandardScaler().fit(dataset.X[train_idx])
    dataset.X = scaler.transform(dataset.X).astype(np.float32)
    return scaler


class NonZeroFilter:
    """train 샘플 기준 nonzero 개수가 min_count 미만인(거의 다 0인) feature 컬럼 제거.
    2026-10-31 B트랙(장다예) 검증 결과 채택: skin 2,266->444차원(min_count=20).
    MCC는 그대로지만 Sens/Spec 격차·ROC-AUC가 20회 짝지은 비교에서 꾸준히 개선됨
    (modeling/eval_b/final_results.json). 반드시 train 인덱스에서만 fit — 전체 데이터에 fit하면
    test 정보가 샘(장다예 지적)."""

    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X):
        self.keep_ = (X != 0).sum(axis=0) >= self.min_count
        if self.keep_.sum() == 0:
            self.keep_[:] = True
        return self

    def transform(self, X):
        return X[:, self.keep_]


def filter_inplace(dataset, train_idx, min_count=20):
    """NonZeroFilter를 train 인덱스에서 fit하고 데이터셋 전체(train+val+test)에 적용(컬럼만 추리는
    변환이라 test 쪽 라벨/분포 정보가 새어 들어가지 않음 — standardize_inplace와 동일한 안전 패턴).
    StandardScaler보다 먼저 적용해야 함(스케일링 후엔 0이던 값도 0이 아니게 되어 nonzero 판정이 깨짐)."""
    filt = NonZeroFilter(min_count).fit(dataset.X[train_idx])
    dataset.X = filt.transform(dataset.X).astype(np.float32)
    return filt


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def class_weights(labels, num_classes=2):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """검증 손실(val loss)이 patience 에폭 연속 개선 안 되면 멈추라는 신호를 주는 헬퍼.
    가장 좋았던 에폭의 모델 가중치를 따로 기억해뒀다가 돌려준다(마지막 에폭이 아니라 최적 에폭 채택)."""

    def __init__(self, patience=15, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.best_epoch = None
        self.best_state = None

    def step(self, value, epoch, model):
        improved = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience  # True면 멈출 시점


def undersample_1to1(indices, labels, rng):
    indices = np.array(indices)
    labels = labels[indices]
    pos = indices[labels == 1]
    neg = indices[labels == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError("1:1 언더샘플링 불가 — 한쪽 클래스가 0건")
    pos_sample = rng.choice(pos, size=n, replace=False)
    neg_sample = rng.choice(neg, size=n, replace=False)
    return np.concatenate([pos_sample, neg_sample])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin-csv", default=SKIN_CSV)
    parser.add_argument("--eye-csv", default=EYE_CSV)
    parser.add_argument("--epochs", type=int, default=200, help="최대 에폭(상한) — early stopping이 보통 더 일찍 멈춤")
    parser.add_argument("--patience", type=int, default=15, help="검증 손실이 이만큼 연속 개선 안 되면 조기 종료")
    parser.add_argument("--val-size", type=float, default=0.2, help="train 중 검증용으로 떼어낼 비율(조기종료 모니터링용)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="2026-10-30(1차) lr=1e-2로 바꿨다가, 2026-10-31 반복 group 5-fold(10회) 재검증 결과 "
                              "철회(원복). 단일 분할 1회 비교에선 좋아 보였지만 반복해보니 노이즈였고, 오히려 "
                              "lr=0.01이 구조개선(층1/dropout0)의 격차개선 효과를 깎아먹는 것으로 확인됨 "
                              "(skin_stage23_repeatedcv_revalidation.ipynb 참조, B트랙 장다예가 문제 제기)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--skin-num-layers", type=int, default=1,
                         help="2026-10-30 구조 탐색 결과 채택(기존 2): skin은 데이터가 적어(541건, 양성 29건) 얕은 구조가 더 나음")
    parser.add_argument("--skin-dropout", type=float, default=0.0,
                         help="2026-10-30 구조 탐색 결과 채택(기존 0.2): 은닉층을 줄이니 dropout 없이도 격차/MCC 개선")
    parser.add_argument("--eye-num-layers", type=int, default=2, help="eye는 팀원 담당 영역, 기존 구조 유지")
    parser.add_argument("--eye-dropout", type=float, default=0.2, help="eye는 팀원 담당 영역, 기존 구조 유지")
    parser.add_argument("--skin-nz-min-count", type=int, default=20,
                         help="2026-10-31 B트랙(장다예) 검증 결과 채택: nonzero 개수가 이 값 미만인 skin "
                              "feature 컬럼 제거(2,266->444차원). eye는 적용 안 함(팀원 담당 영역)")
    parser.add_argument("--balance", choices=["weighted", "1to1"], default="weighted")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="checkpoint.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    skin_ds = skin_dataset(args.skin_csv)
    eye_ds = eye_dataset(args.eye_csv)

    skin_trainval_idx, skin_test_idx = group_train_test_split(skin_ds, args.test_size, args.seed)
    eye_trainval_idx, eye_test_idx = group_train_test_split(eye_ds, args.test_size, args.seed)

    skin_filter = filter_inplace(skin_ds, skin_trainval_idx, min_count=args.skin_nz_min_count)
    skin_scaler = standardize_inplace(skin_ds, skin_trainval_idx)
    eye_scaler = standardize_inplace(eye_ds, eye_trainval_idx)

    # early stopping 모니터링용 검증셋을 train에서 추가로 떼어냄(group-aware, test와는 완전 분리된 채로).
    skin_train_idx, skin_val_idx = group_train_test_split_from_subset(skin_ds, skin_trainval_idx, args.val_size, args.seed)
    eye_train_idx, eye_val_idx = group_train_test_split_from_subset(eye_ds, eye_trainval_idx, args.val_size, args.seed)

    if args.balance == "1to1":
        skin_train_idx = undersample_1to1(skin_train_idx, skin_ds.y, rng)
        eye_train_idx = undersample_1to1(eye_train_idx, eye_ds.y, rng)

    skin_train = Subset(skin_ds, skin_train_idx)
    eye_train = Subset(eye_ds, eye_train_idx)

    skin_loader = DataLoader(skin_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eye_loader = DataLoader(eye_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = MultiTaskIrritationModel(
        skin_ds.X.shape[1], eye_ds.input_dim, hidden_dim=args.hidden_dim,
        skin_num_layers=args.skin_num_layers, skin_dropout=args.skin_dropout,
        eye_num_layers=args.eye_num_layers, eye_dropout=args.eye_dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.balance == "weighted":
        skin_criterion = nn.CrossEntropyLoss(weight=class_weights(skin_ds.y[skin_train_idx]))
        eye_criterion = nn.CrossEntropyLoss(weight=class_weights(eye_ds.y[eye_train_idx]))
    else:
        skin_criterion = nn.CrossEntropyLoss()
        eye_criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(skin_loader), len(eye_loader))
    skin_iter = cycle(skin_loader)
    eye_iter = cycle(eye_loader)

    skin_X_val = torch.from_numpy(skin_ds.X[skin_val_idx])
    skin_y_val = torch.from_numpy(skin_ds.y[skin_val_idx])
    eye_X_val = torch.from_numpy(eye_ds.X[eye_val_idx])
    eye_y_val = torch.from_numpy(eye_ds.y[eye_val_idx])

    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            sx, sy = next(skin_iter)
            ex, ey = next(eye_iter)

            skin_logits = model.forward_skin(sx)
            eye_logits = model.forward_eye(ex)

            skin_loss = skin_criterion(skin_logits, sy)
            eye_loss = eye_criterion(eye_logits, ey)
            loss = model.alpha * skin_loss + model.beta * eye_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_skin_loss = skin_criterion(model.forward_skin(skin_X_val), skin_y_val)
            val_eye_loss = eye_criterion(model.forward_eye(eye_X_val), eye_y_val)
            val_loss = (model.alpha * val_skin_loss + model.beta * val_eye_loss).item()

        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={total_loss / steps_per_epoch:.4f} "
            f"val_loss={val_loss:.4f} alpha={model.alpha.item():.3f} beta={model.beta.item():.3f}"
        )

        should_stop = stopper.step(val_loss, epoch, model)
        if should_stop:
            print(f"조기 종료: epoch {epoch + 1}에서 중단 (최적은 epoch {stopper.best_epoch + 1}, val_loss={stopper.best:.4f})")
            break

    model.load_state_dict(stopper.best_state)
    print(f"최종 채택: epoch {stopper.best_epoch + 1}의 가중치 (val_loss={stopper.best:.4f})")

    torch.save(
        {
            "model_state": model.state_dict(),
            "skin_input_dim": skin_ds.X.shape[1],
            "eye_input_dim": eye_ds.input_dim,
            "skin_nz_filter_keep": skin_filter.keep_,
            "skin_nz_min_count": args.skin_nz_min_count,
            "hidden_dim": args.hidden_dim,
            "skin_num_layers": args.skin_num_layers,
            "skin_dropout": args.skin_dropout,
            "eye_num_layers": args.eye_num_layers,
            "eye_dropout": args.eye_dropout,
            "skin_test_idx": skin_test_idx,
            "eye_test_idx": eye_test_idx,
            "skin_scaler": skin_scaler,
            "eye_scaler": eye_scaler,
            "best_epoch": stopper.best_epoch,
        },
        args.save_path,
    )
    print(f"저장 완료: {args.save_path} (evaluate.py에서 로드해 테스트셋 평가)")


if __name__ == "__main__":
    main()
