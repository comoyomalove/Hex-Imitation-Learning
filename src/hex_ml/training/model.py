"""Trained policy network: the "brain" of the delivered imitation-learning agent.

TinyHexPolicyNet takes the 4-channel encoded board (see
src/hex_ml/encoding/board_encoder.py) and outputs one score per board cell.
The highest-scoring legal cell (after masking in
src/hex_ml/agents/trained_agent.py) is the agent's chosen move. This is a
small fully-connected network by design - simple enough to train quickly from
imitation data, while still learning a real board -> move mapping instead of
following hand-written rules.
"""

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
