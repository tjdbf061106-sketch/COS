"""
Multi-task 모델: skin/eye는 입력 차원이 달라(270 vs 454) task-specific encoder로 각각 받고,
shared_trunk에서 공통 표현을 학습한 뒤 task별 head(skin_head/eye_head)로 분기.

alpha/beta는 "벡터를 더하는 용도"가 아니라 skin_loss/eye_loss를 합칠 때 쓰는 가중치.
(skin·eye가 서로 다른 Formulation_ID라 샘플 단위로 벡터를 더할 짝이 없음 — 진행상황.md 참조)
"""
import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    """num_layers=은닉층 개수. dropout은 기존 구조와 동일하게 첫 번째 은닉층 뒤에만 적용
    (2026-10-30 구조 탐색 노트북(skin_architecture_demo.ipynb)에서 이 형태로 일반화해 비교함)."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiTaskIrritationModel(nn.Module):
    """skin/eye encoder 구조를 독립적으로 지정 가능.
    2026-10-30: skin은 1단계-3번 구조 탐색 결과 num_layers=1/dropout=0.0 채택
    (은닉층 2개/dropout 0.2 대비 Sens 0.633->0.667, Spec 0.800->0.812, 격차 0.167->0.145, MCC 0.238->0.267).
    eye는 담당 팀원 영역이라 기존 구조(num_layers=2, dropout=0.2) 유지."""

    def __init__(self, skin_input_dim, eye_input_dim, hidden_dim=128, num_classes=2,
                 skin_num_layers=1, skin_dropout=0.0, eye_num_layers=2, eye_dropout=0.2):
        super().__init__()
        self.skin_encoder = TaskEncoder(skin_input_dim, hidden_dim, dropout=skin_dropout, num_layers=skin_num_layers)
        self.eye_encoder = TaskEncoder(eye_input_dim, hidden_dim, dropout=eye_dropout, num_layers=eye_num_layers)
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.skin_head = nn.Linear(hidden_dim, num_classes)
        self.eye_head = nn.Linear(hidden_dim, num_classes)

        # task loss 가중치(학습 중 최적화). sigmoid로 0~1 범위 보장, alpha+beta=1 제약.
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return 1.0 - self.alpha

    def forward_skin(self, x):
        z = self.shared_trunk(self.skin_encoder(x))
        return self.skin_head(z)

    def forward_eye(self, x):
        z = self.shared_trunk(self.eye_encoder(x))
        return self.eye_head(z)
