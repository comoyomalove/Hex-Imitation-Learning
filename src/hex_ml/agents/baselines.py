"""Non-ML baseline agents used only as sanity-check opponents.

These do not learn anything; they exist to give the trained imitation agent
(and the dataset-generation pipeline) an easy opponent to validate against.
They are not part of the delivered machine-learning model.
"""
from __future__ import annotations
import random
import numpy as np
from hex_ml.core.board import Move, get_legal_moves, validate_board


class FirstLegalAgent:
    """Always plays the first legal move found; useful for smoke tests."""

    name = "first_legal_agent"
    agent_type = "baseline_first_legal"

    def choose_move(self, board: np.ndarray, player_to_move: int) -> Move:
        validate_board(board)
        moves = get_legal_moves(board)
        if not moves:
            raise ValueError("No legal moves available.")
        return moves[0]


class RandomLegalAgent:
    """Plays a uniformly random legal move; the weakest baseline opponent."""

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
