"""
skin/eye 벡터화 CSV를 PyTorch Dataset으로 로드.
라벨: EPA Classification Category 1~4 -> binary(0=비독성(Cat3~4), 1=독성(Cat1~2)).

2026-10-06 결정: skin은 CASRN one-hot(casrn_* prefix) 방식을 폐기하고 RDKit SMILES descriptor
방식으로 교체 — 새 화합물에 일반화 안 되는 근본적 한계 때문.
2026-10-06(2차): 팀 논의로 skin도 eye 수준(전체 RDKit descriptor 217개 + Morgan FP 2048개)까지
확장 — 기존 공통 8개 descriptor는 유지한 채 desc_*/fp_w_* 컬럼 추가(skin_feature_expand_full.ipynb).
eye도 같은 시점에 동일 네이밍(desc_*/fp_w_*)으로 재구성 완료 — 두 트랙 feature 스키마가 호환됨.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CATEGORY_TO_BINARY = {1: 1, 2: 1, 3: 0, 4: 0}

# skin_vectorized.csv의 공통 8개 descriptor (desc_*/fp_w_* 와 별도로 이름이 고정된 컬럼)
SKIN_COMMON8_COLS = ["MolWt", "LogP", "TPSA", "HBD", "HBA",
                     "RotatableBonds", "AromaticProportion", "ESOL_LogS"]


class IrritationDataset(Dataset):
    def __init__(self, csv_path, label_col, feature_prefix=None, feature_cols=None, substance_type_filter=None):
        df = pd.read_csv(csv_path)

        if substance_type_filter is not None and "substance_type" in df.columns:
            df = df[df["substance_type"] == substance_type_filter]

        df = df.dropna(subset=[label_col])

        if feature_cols is not None:
            cols = feature_cols
            # one-hot과 달리 결측(descriptor 전부 NaN = 구조 완전 미해결, 예: MIX306)은 0으로 채우면 안 되므로 제외
            df = df.dropna(subset=cols)
        elif feature_prefix is not None:
            cols = [c for c in df.columns if c.startswith(feature_prefix)]
            if not cols:
                raise ValueError(f"'{feature_prefix}'로 시작하는 feature 컬럼을 찾지 못함")
        else:
            raise ValueError("feature_prefix 또는 feature_cols 중 하나는 지정해야 함")

        df = df.reset_index(drop=True)
        self.feature_cols = cols

        if label_col == "binary_label":
            y = df[label_col].astype(int)
        else:
            y = df[label_col].astype(int).map(CATEGORY_TO_BINARY)
            if y.isna().any():
                raise ValueError(f"{label_col}에 1~4 외 매핑 안 되는 값 존재: {df[label_col].unique()}")

        self.X = df[cols].fillna(0).to_numpy(dtype=np.float32)
        self.y = y.to_numpy(dtype=np.int64)
        self.formulation_id = df["formulation_id"].to_numpy()
        self.group_id = df["comp_group_id"] if "comp_group_id" in df.columns else df.get("group_id")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    @property
    def input_dim(self):
        return len(self.feature_cols)


def skin_dataset(csv_path="../skin irritation/skin_vectorized.csv"):
    # 공통 8개 + desc_*(전체 RDKit 217) + fp_w_*(Morgan FP 2048) = 2,273차원
    # 2026-10-29: unknown_pct/undeclared_pct를 feature로 추가하는 실험을 해봤으나(2,275차원),
    # Sensitivity/Specificity 균형·MCC 기준으로 skin 성능이 오히려 악화되어(Sens 0.367->0.333,
    # 격차 0.520->0.545, MCC 0.174->0.143) 되돌림. skin_addons.ipynb의 단순 모델 테스트에서는
    # 개선된 것처럼 보였으나(RF 계열), 실제 Multi-task 신경망에서는 재현 안 됨 — 진행상황.md 참조.
    header = pd.read_csv(csv_path, nrows=0).columns
    expanded_cols = [c for c in header if c.startswith(("desc_", "fp_w_"))]
    feature_cols = SKIN_COMMON8_COLS + expanded_cols
    return IrritationDataset(csv_path, label_col="binary_label", feature_cols=feature_cols)


def eye_dataset(csv_path="../eye irritation/eye_vectorized.csv"):
    # substance_type == 'Mixture'만 사용(순물질 Chemical 제외) — 진행상황.md 2026-09-18 결정사항
    # 2026-10-06 재구성 이후 eye도 desc_*/fp_w_* 스키마 + binary_label 보유(GHS만 있는 47건은 NaN -> 자동 제외)
    return IrritationDataset(
        csv_path, label_col="binary_label", feature_prefix=("desc_", "fp_w_"),
        substance_type_filter="Mixture",
    )
