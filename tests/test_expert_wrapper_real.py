import numpy as np
from gamelib.hex.gamestate import GameState
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.gamelib_wrapper import make_gamelib_state
from hex_ml.core.board import PLAYER_1, create_empty_board, is_legal_move


def testmake_gamelib_state_uses_real_aica_gamelib_state():
    state = make_gamelib_state(create_empty_board(3), PLAYER_1)
    assert isinstance(state, GameState)
    assert state.board_size == 3
    assert state.turn == 0
    assert state.board == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_real_expert_wrapper_returns_legal_move_without_mutating_board():
    board = create_empty_board(3)
    wrapper = ExpertAgentWrapper()
    result = wrapper.choose_move(board, PLAYER_1)
    assert wrapper.available is True
    assert result.source == "winner_agent"
    assert is_legal_move(board, result.move)
    assert np.array_equal(board, create_empty_board(3))
