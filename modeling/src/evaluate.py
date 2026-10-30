"""
train.py가 저장한 checkpoint로 skin/eye 테스트셋(group split로 분리된 held-out) 평가.
지표: F1-score, ROC-AUC, 혼동행렬 (연구계획서 이분류 평가지표).

사용법: py evaluate.py --checkpoint checkpoint.pt
"""
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from datasets import EYE_CSV, SKIN_CSV, skin_dataset, eye_dataset
from model import MultiTaskIrritationModel


def evaluate_task(model, forward_fn, dataset, test_idx, label):
    X = torch.from_numpy(dataset.X[test_idx])
    y_true = dataset.y[test_idx]

    model.eval()
    with torch.no_grad():
        logits = forward_fn(X)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        y_pred = logits.argmax(dim=1).numpy()

    print(f"\n=== {label} 테스트셋 (n={len(y_true)}) ===")
    print(f"F1-score: {f1_score(y_true, y_pred):.3f}")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, probs):.3f}")
    except ValueError:
        print("ROC-AUC: 계산 불가(테스트셋에 한 클래스만 존재)")
    print("혼동행렬 [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["non-toxic(0)", "toxic(1)"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--skin-csv", default=SKIN_CSV)
    parser.add_argument("--eye-csv", default=EYE_CSV)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    skin_ds = skin_dataset(args.skin_csv)
    eye_ds = eye_dataset(args.eye_csv)
    skin_ds.X = ckpt["skin_scaler"].transform(skin_ds.X).astype("float32")
    eye_ds.X = ckpt["eye_scaler"].transform(eye_ds.X).astype("float32")

    model = MultiTaskIrritationModel(
        ckpt["skin_input_dim"], ckpt["eye_input_dim"], hidden_dim=ckpt["hidden_dim"],
        skin_num_layers=ckpt.get("skin_num_layers", 2), skin_dropout=ckpt.get("skin_dropout", 0.2),
        eye_num_layers=ckpt.get("eye_num_layers", 2), eye_dropout=ckpt.get("eye_dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])

    evaluate_task(model, model.forward_skin, skin_ds, ckpt["skin_test_idx"], "Skin")
    evaluate_task(model, model.forward_eye, eye_ds, ckpt["eye_test_idx"], "Eye")


if __name__ == "__main__":
    main()
