"""Shared PyTorch model definition for Football tabular prediction."""

from __future__ import annotations

import torch
from torch import nn


class FootballTorchModel(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        hidden_mid = max(128, hidden_width // 2)
        hidden_low = max(64, hidden_width // 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_width),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_mid),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_mid),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, hidden_low),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_low, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
