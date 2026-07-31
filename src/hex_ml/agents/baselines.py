from __future__ import annotations
import random
import numpy as np
from hex_ml.core.board import Move, get_legal_moves, validate_board


class FirstLegalAgent:
    name = "first_legal_agent"
    agent_type = "baseline_first_legal"

    def choose_move(self, board: np.ndarray, player_to_move: int) -> Move:
        validate_board(board)
        moves = get_legal_moves(board)
        if not moves:
            raise ValueError("No legal moves available.")
        return moves[0]


class RandomLegalAgent:
    name = "random_legal_agent"
    agent_type = "baseline_random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: np.ndarray, player_to_move: int) -> Move:
        validate_board(board)
        moves = get_legal_moves(board)
        if not moves:
            raise ValueError("No legal moves available.")
        return self._rng.choice(moves)
