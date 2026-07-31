"""Minimal Hex game helpers."""

from __future__ import annotations

from collections import deque

import numpy as np

from hex_ml.core.board import (
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    Move,
    is_legal_move,
    validate_board,
)

NEIGHBOR_OFFSETS: tuple[Move, ...] = (
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
)


def other_player(player: int) -> int:
    """Return the other Hex player."""
    if player == PLAYER_1:
        return PLAYER_2
    if player == PLAYER_2:
        return PLAYER_1
    raise ValueError(f"Invalid player: {player}. Expected {PLAYER_1} or {PLAYER_2}.")


def apply_move(board: np.ndarray, move: Move, player: int) -> np.ndarray:
    """Return a new board with player's move applied. The input board is not mutated."""
    validate_board(board)
    if player not in (PLAYER_1, PLAYER_2):
        raise ValueError(
            f"Invalid player: {player}. Expected {PLAYER_1} or {PLAYER_2}."
        )
    if not is_legal_move(board, move):
        raise ValueError(f"Illegal move: {move}")
    next_board = board.copy()
    row, col = move
    next_board[row, col] = player
    return next_board


def iter_neighbors(row: int, col: int, size: int) -> list[Move]:
    """Return valid Hex-neighbor coordinates for a board cell."""
    neighbors: list[Move] = []
    for row_delta, col_delta in NEIGHBOR_OFFSETS:
        next_row = row + row_delta
        next_col = col + col_delta
        if 0 <= next_row < size and 0 <= next_col < size:
            neighbors.append((next_row, next_col))
    return neighbors


def check_winner(board: np.ndarray) -> int | None:
    """Return winning player using the hackathon/gamelib orientation.

    Local player 1 maps to gamelib player 0, which connects left-to-right.
    Local player 2 maps to gamelib player 1, which connects top-to-bottom.
    """
    validate_board(board)
    if _has_connection(board, PLAYER_1):
        return PLAYER_1
    if _has_connection(board, PLAYER_2):
        return PLAYER_2
    return None

def game_status(board: np.ndarray) -> str:
    """Return a simple status string for logs."""
    winner = check_winner(board)
    if winner is not None:
        return f"winner_{winner}"
    if not np.any(board == EMPTY):
        return "draw_or_full_board"
    return "in_progress"


def replay_moves(board_size: int, moves: list[tuple[int, Move]]) -> np.ndarray:
    """Replay player/move pairs and return the final board."""
    board = np.zeros((board_size, board_size), dtype=np.int8)
    for player, move in moves:
        board = apply_move(board, move, player)
    return board


def _has_connection(board: np.ndarray, player: int) -> bool:
    size = board.shape[0]
    queue: deque[Move] = deque()
    visited: set[Move] = set()

    if player == PLAYER_1:
        # Local player 1 / gamelib player 0 connects left to right.
        for row in range(size):
            if board[row, 0] == player:
                queue.append((row, 0))
                visited.add((row, 0))

        def target_reached(_row: int, col: int) -> bool:
            return col == size - 1

    elif player == PLAYER_2:
        # Local player 2 / gamelib player 1 connects top to bottom.
        for col in range(size):
            if board[0, col] == player:
                queue.append((0, col))
                visited.add((0, col))

        def target_reached(row: int, _col: int) -> bool:
            return row == size - 1

    else:
        raise ValueError(f"Invalid player: {player}")

    while queue:
        row, col = queue.popleft()
        if target_reached(row, col):
            return True
        for neighbor in iter_neighbors(row, col, size):
            n_row, n_col = neighbor
            if neighbor not in visited and board[n_row, n_col] == player:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
