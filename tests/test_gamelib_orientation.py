import numpy as np

from gamelib.hex.engine import Engine

from hex_ml.agents.gamelib_wrapper import make_gamelib_state
from hex_ml.core.board import PLAYER_1, PLAYER_2
from hex_ml.core.game import check_winner


def gamelib_status_for_local_board(board, player_to_move=PLAYER_1):
    state = make_gamelib_state(board, player_to_move)
    return Engine().get_status(state)


def test_local_player_1_matches_gamelib_player_0_left_to_right():
    board = np.array([[1, 1, 1], [0, 2, 0], [0, 0, 2]], dtype=np.int8)
    assert check_winner(board) == PLAYER_1
    assert gamelib_status_for_local_board(board) == 0


def test_local_player_2_matches_gamelib_player_1_top_to_bottom():
    board = np.array([[2, 1, 0], [2, 1, 0], [2, 0, 1]], dtype=np.int8)
    assert check_winner(board) == PLAYER_2
    assert gamelib_status_for_local_board(board) == 1
