import numpy as np
import pytest

from hex_ml.core.board import create_empty_board, get_legal_moves, is_legal_move


def test_create_empty_board_has_expected_shape_and_values():
    board = create_empty_board(3)
    assert board.shape == (3, 3)
    assert board.dtype == np.int8
    assert np.all(board == 0)


def test_get_legal_moves_returns_only_empty_cells_in_row_major_order():
    board = np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]], dtype=np.int8)
    assert get_legal_moves(board) == [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]


def test_is_legal_move_rejects_occupied_and_out_of_bounds_cells():
    board = np.array([[0, 1], [2, 0]], dtype=np.int8)
    assert is_legal_move(board, (0, 0)) is True
    assert is_legal_move(board, (0, 1)) is False
    assert is_legal_move(board, (-1, 0)) is False
    assert is_legal_move(board, (2, 0)) is False


def test_get_legal_moves_rejects_invalid_board_values():
    board = np.array([[0, 3], [1, 2]], dtype=np.int8)
    with pytest.raises(ValueError):
        get_legal_moves(board)
