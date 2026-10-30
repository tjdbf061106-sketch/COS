"""
comp_group_id(skin) / group_id(eye) 기준으로 group-aware 분할.
동일 조성 쌍(composition twin)이 train/test에 나뉘어 들어가는 data leakage 방지.

2026-10-06: group_train_test_split() 단일분할은 skin처럼 양성이 29건뿐인 경우
테스트셋 양성이 2~9건으로 들쭉날쭉해 ROC-AUC가 시드에 따라 0.30~0.95까지 흔들림을 확인함
(skin_harness_cv_upgrade.ipynb 참조). 성능 "보고/비교"는 grouped_kfold_splits()를 쓸 것 —
단일분할은 최종 배포 모델을 한 번 학습시킬 때만 사용.
"""
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def group_train_test_split(dataset, test_size=0.2, random_state=42):
    """최종 모델을 한 번 학습시킬 때 쓰는 단순 분할. 성능 보고용으로는 쓰지 말 것(변동폭 큼)."""
    groups = dataset.group_id
    if groups is None:
        raise ValueError("dataset에 comp_group_id/group_id가 없어 group split 불가")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dataset.X, dataset.y, groups=groups))
    return train_idx, test_idx


def group_train_test_split_from_subset(dataset, subset_idx, val_size=0.2, random_state=42):
    """subset_idx(예: train+val을 합친 인덱스)를 group-aware하게 다시 train/val로 쪼갠다.
    early stopping 모니터링용 검증셋을 만들 때 사용 — 이미 떼어둔 test셋과는 완전히 분리된 상태에서
    train 내부만 추가로 나누므로 test leakage는 없음."""
    import numpy as np

    subset_idx = np.asarray(subset_idx)
    groups = np.asarray(dataset.group_id)[subset_idx]

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_pos, val_pos = next(splitter.split(subset_idx, groups=groups))
    return subset_idx[train_pos], subset_idx[val_pos]


def grouped_kfold_splits(dataset, n_splits=5, random_state=0):
    """StratifiedGroupKFold -> [(train_idx, test_idx), ...]. 성능 보고/비교는 이 함수로 OOF 예측을 모아서 할 것."""
    groups = dataset.group_id
    if groups is None:
        raise ValueError("dataset에 comp_group_id/group_id가 없어 group split 불가")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(dataset.X, dataset.y, groups=groups))
