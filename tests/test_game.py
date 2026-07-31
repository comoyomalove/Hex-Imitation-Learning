import numpy as np
import pytest

from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board
from hex_ml.core.game import (
    apply_move,
    check_winner,
    game_status,
    other_player,
    replay_moves,
)


def test_other_player_alternates_players():
    assert other_player(PLAYER_1) == PLAYER_2
    assert other_player(PLAYER_2) == PLAYER_1


def test_apply_move_returns_new_board_without_mutating_original():
    board = create_empty_board(3)
    next_board = apply_move(board, (1, 1), PLAYER_1)
    assert board[1, 1] == 0
    assert next_board[1, 1] == PLAYER_1


def test_apply_move_rejects_occupied_cell():
    board = create_empty_board(3)
    board = apply_move(board, (0, 0), PLAYER_1)
    with pytest.raises(ValueError, match="Illegal move"):
        apply_move(board, (0, 0), PLAYER_2)


def test_check_winner_detects_player_1_left_to_right_connection():
    board = np.array([[1, 1, 1], [0, 2, 0], [0, 0, 2]], dtype=np.int8)
    assert check_winner(board) == PLAYER_1
    assert game_status(board) == "winner_1"


def test_check_winner_detects_player_2_top_to_bottom_connection():
    board = np.array([[2, 1, 0], [2, 1, 0], [2, 0, 1]], dtype=np.int8)
    assert check_winner(board) == PLAYER_2
    assert game_status(board) == "winner_2"


def test_replay_moves_returns_expected_final_board():
    final_board = replay_moves(
        3, [(PLAYER_1, (0, 0)), (PLAYER_2, (1, 1)), (PLAYER_1, (1, 0))]
    )
    assert final_board.tolist() == [[1, 0, 0], [1, 2, 0], [0, 0, 0]]
