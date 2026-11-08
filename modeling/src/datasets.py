"""
skin/eye 벡터화 CSV를 PyTorch Dataset으로 로드.
라벨: EPA Classification Category 1~4 -> binary(0=비독성(Cat3~4), 1=독성(Cat1~2)).

2026-10-06 결정: skin은 CASRN one-hot(casrn_* prefix) 방식을 폐기하고 RDKit SMILES descriptor
방식으로 교체 — 새 화합물에 일반화 안 되는 근본적 한계 때문.
2026-10-06(2차): 팀 논의로 skin도 eye 수준(전체 RDKit descriptor 217개 + Morgan FP 2048개)까지
확장 — 기존 공통 8개 descriptor는 유지한 채 desc_*/fp_w_* 컬럼 추가(skin_feature_expand_full.ipynb).
eye도 같은 시점에 동일 네이밍(desc_*/fp_w_*)으로 재구성 완료 — 두 트랙 feature 스키마가 호환됨.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 기본 CSV 경로는 이 파일 위치에서 레포 루트를 거슬러 올라가 고정한다.
# cwd 기준 상대경로("../skin irritation/...")로 두면 modeling/src에서 실행할 때
# modeling/skin irritation/ 을 찾아 실패하고, 레포 루트에서 실행하면 또 다른 곳을 찾는다.
# datasets.py -> src -> modeling -> 레포 루트
REPO_ROOT = Path(__file__).resolve().parents[2]
SKIN_CSV = REPO_ROOT / "skin irritation" / "skin_vectorized.csv"
EYE_CSV = REPO_ROOT / "eye irritation" / "eye_vectorized.csv"

CATEGORY_TO_BINARY = {1: 1, 2: 1, 3: 0, 4: 0}

# skin_vectorized.csv의 공통 descriptor (desc_*/fp_w_* 와 별도로 이름이 고정된 컬럼)
# 2026-10-29: ESOL_LogS 제외(원래 8개->7개). ESOL_LogS = 0.16-0.63*LogP-0.0062*MolWt+0.066*RotB-0.74*AromaticProportion
# 로 LogP/MolWt/RotatableBonds/AromaticProportion의 선형조합이라 100% 중복 정보. 3가지 처리방식(ESOL_LogS 제거/
# LogP 제거/PCA 1성분)을 group 5-fold로 비교한 결과 Sensitivity/Specificity/MCC는 동률, ROC-AUC는 ESOL_LogS
# 제거가 최고(0.700 vs LogP제거 0.697 vs 유지 0.686)라 채택. PCA는 격차가 더 벌어져(0.567->0.575) 기각.
# 2026-10-29(2차): 나머지 6개 중 MolWt/LogP/TPSA/HBD/HBA/RotatableBonds가 desc_217 안의
# desc_MolWt/desc_MolLogP/desc_TPSA/desc_NumHDonors/desc_NumHAcceptors/desc_NumRotatableBonds와
# 값이 바이트 단위로 100% 동일(완전 중복)함을 발견 — 둘 다 염 제거된 동일 분자에 같은 RDKit 함수를 적용해서 생김.
# AromaticProportion만 217개 안에 동일 계산이 없는 유일한 독립 컬럼이라 이것만 남김(7개->1개).
SKIN_UNIQUE_COMMON_COLS = ["AromaticProportion"]


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


def skin_dataset(csv_path=SKIN_CSV, include_concentration=False):
    # AromaticProportion(유일 독립 공통컬럼, 위 주석 참조) + desc_*(전체 RDKit 217) + fp_w_*(Morgan FP 2048) = 2,266차원
    # 2026-10-29: unknown_pct/undeclared_pct를 feature로 추가하는 실험을 해봤으나(+2차원),
    # Sensitivity/Specificity 균형·MCC 기준으로 skin 성능이 오히려 악화되어(Sens 0.367->0.333,
    # 격차 0.520->0.545, MCC 0.174->0.143) 되돌림. skin_addons.ipynb의 단순 모델 테스트에서는
    # 개선된 것처럼 보였으나(RF 계열), 실제 Multi-task 신경망에서는 재현 안 됨 — 진행상황.md 참조.
    #
    # 2026-11-08: eye팀이 wmean 집계식(pct*d/pct.sum())이 단일성분 제형에서 농도를 약분해 버린다는
    # 근본 결함을 발견(같은 feature 벡터에 다른 라벨이 붙는 사례 다수). skin도 동일 결함 확인됨
    # (comp_group_id를 서명 6자리로 정밀화 + 벡터해시에 resolved_pct를 포함해 재계산하니 고유 그룹
    # 463 -> 637로 증가 = 농도만 다른 제형들이 더 이상 같은 벡터로 뭉개지지 않음).
    # `include_concentration=True`로 `resolved_pct`(구조 해결 성분 % 합, 이미 CSV에 존재하던 컬럼)를
    # feature에 추가하는 효과를 반복 group 5-fold(5seed)로 측정한 결과(skin_concentration_feature_effect.csv):
    #   overall ΔMCC - 라벨A ExtraTrees +0.044 / RandomForest -0.020, 라벨B ExtraTrees +0.016 / RandomForest -0.001
    # ExtraTrees는 일관되게 개선, 현재 production 모델인 RandomForest는 효과가 들쭉날쭉(0 근방) —
    # 기본값은 False로 유지(기존 checkpoint 차원 호환성 보존). 세팅/생성스크립트/eval_concentration_feature_effect.py 참조.
    header = pd.read_csv(csv_path, nrows=0).columns
    expanded_cols = [c for c in header if c.startswith(("desc_", "fp_w_"))]
    feature_cols = SKIN_UNIQUE_COMMON_COLS + expanded_cols
    if include_concentration:
        feature_cols = feature_cols + ["resolved_pct"]
    return IrritationDataset(csv_path, label_col="binary_label", feature_cols=feature_cols)


def eye_dataset(csv_path=EYE_CSV, include_concentration=False):
    # substance_type == 'Mixture'만 사용(순물질 Chemical 제외) — 진행상황.md 2026-09-18 결정사항
    # 2026-10-06 재구성 이후 eye도 desc_*/fp_w_* 스키마 + binary_label 보유(GHS만 있는 47건은 NaN -> 자동 제외)
    #
    # 2026-11-08: eye팀 자체 검증(실험11, `세팅/eye_11월07-08_작업정리.pdf`)에서 `known_pct`(구조 해결
    # 성분 % 합) 피처 추가가 라벨B에서 seed변동(±0.024)을 넘는 개선(+0.030, 10칸 중 9칸)을 보였고,
    # 라벨A는 무해함을 확인(방안1 채택, production 반영은 보류). 이 저장소의 `eye_concentration_feature_effect.csv`로
    # 같은 방향성(ΔMCC 라벨A +0.007~+0.007, 라벨B +0.013~+0.027, 전부 양수)을 독립 재확인함.
    # `include_concentration=True`로 `known_pct`를 feature에 추가할 수 있으나, 기본값은 False로 유지
    # (이 harness를 쓰는 기존 코드/체크포인트의 차원 호환성 보존 — 채택 여부는 팀 결정 사항).
    header = pd.read_csv(csv_path, nrows=0).columns
    feature_cols = [c for c in header if c.startswith(("desc_", "fp_w_"))]
    if include_concentration:
        feature_cols = feature_cols + ["known_pct"]
    return IrritationDataset(
        csv_path, label_col="binary_label", feature_cols=feature_cols,
        substance_type_filter="Mixture",
    )
