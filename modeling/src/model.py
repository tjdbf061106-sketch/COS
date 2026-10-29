"""
Multi-task 모델: skin/eye는 입력 차원이 달라(270 vs 454) task-specific encoder로 각각 받고,
shared_trunk에서 공통 표현을 학습한 뒤 task별 head(skin_head/eye_head)로 분기.

alpha/beta는 "벡터를 더하는 용도"가 아니라 skin_loss/eye_loss를 합칠 때 쓰는 가중치.
(skin·eye가 서로 다른 Formulation_ID라 샘플 단위로 벡터를 더할 짝이 없음 — 진행상황.md 참조)
"""
import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskIrritationModel(nn.Module):
    def __init__(self, skin_input_dim, eye_input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.skin_encoder = TaskEncoder(skin_input_dim, hidden_dim)
        self.eye_encoder = TaskEncoder(eye_input_dim, hidden_dim)
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
