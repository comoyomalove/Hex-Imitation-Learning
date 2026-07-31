import numpy as np

from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.core.board import PLAYER_1, create_empty_board, is_legal_move


def test_expert_wrapper_returns_legal_move():
    board = create_empty_board(3)
    wrapper = ExpertAgentWrapper()

    result = wrapper.choose_move(board, PLAYER_1)

    assert is_legal_move(board, result.move)
    assert result.expert_name == "hackathon_winner_agent"


def test_expert_wrapper_does_not_mutate_board():
    board = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int8)
    before = board.copy()
    wrapper = ExpertAgentWrapper()

    wrapper.choose_move(board, PLAYER_1)

    assert np.array_equal(board, before)
