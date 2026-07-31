"""Simple Hex board helpers."""

from __future__ import annotations


import numpy as np

EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2
PLAYER_ONE = 1
PLAYER_TWO = 2

Move = tuple[int, int]


def create_empty_board(size: int) -> np.ndarray:
    """Create a square Hex board filled with EMPTY values."""
    if size <= 0:
        raise ValueError("board size must be positive")
    return np.zeros((size, size), dtype=np.int8)


def validate_board(board: np.ndarray) -> None:
    """Validate that a board is a square 2D NumPy array with known cell values."""
    if not isinstance(board, np.ndarray):
        raise TypeError("board must be a numpy.ndarray")
    if board.ndim != 2:
        raise ValueError("board must be 2-dimensional")
    if board.shape[0] != board.shape[1]:
        raise ValueError("board must be square")
    allowed_values = {EMPTY, PLAYER_ONE, PLAYER_TWO}
    actual_values = set(np.unique(board).tolist())
    if not actual_values.issubset(allowed_values):
        raise ValueError(f"board contains invalid values: {sorted(actual_values)}")


def get_legal_moves(board: np.ndarray) -> list[Move]:
    """Return all empty cells as legal moves in row-major order."""
    validate_board(board)
    rows, cols = np.where(board == EMPTY)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def is_legal_move(board: np.ndarray, move: Move) -> bool:
    """Return True when move is on the board and targets an empty cell."""
    validate_board(board)
    row, col = move
    size = board.shape[0]
    return bool(0 <= row < size and 0 <= col < size and board[row, col] == EMPTY)
