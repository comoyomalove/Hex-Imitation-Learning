"""Small imitation policy model for Hex smoke tests."""

from __future__ import annotations
import torch
from torch import nn


class TinyHexPolicyNet(nn.Module):
    def __init__(
        self, board_size: int, input_channels: int = 4, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.board_size = board_size
        input_size = input_channels * board_size * board_size
        output_size = board_size * board_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
