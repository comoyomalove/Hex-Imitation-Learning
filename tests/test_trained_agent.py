import numpy as np
import pytest
import torch
from hex_ml.agents.trained_agent import TrainedHexAgent, mask_illegal_moves
from hex_ml.core.board import PLAYER_1, is_legal_move


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.fixed_logits.unsqueeze(0).repeat(x.shape[0], 1)


def test_mask_illegal_moves_keeps_only_legal_cells():
    logits = torch.arange(9, dtype=torch.float32)
    masked = mask_illegal_moves(logits, legal_moves=[(0, 0), (2, 2)], board_size=3)
    assert masked[0].item() == 0
    assert masked[8].item() == 8
    assert masked[1].item() < -1e8


def test_trained_agent_never_selects_occupied_highest_logit_cell():
    board = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    model = FixedLogitModel([0, 1, 2, 3, 100, 5, 6, 7, 8])
    agent = TrainedHexAgent(model)
    move = agent.choose_move(board, PLAYER_1)
    assert move == (2, 2)
    assert is_legal_move(board, move)


def test_trained_agent_raises_when_no_legal_moves():
    board = np.ones((2, 2), dtype=np.int8)
    agent = TrainedHexAgent(FixedLogitModel([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="No legal moves"):
        agent.choose_move(board, PLAYER_1)
