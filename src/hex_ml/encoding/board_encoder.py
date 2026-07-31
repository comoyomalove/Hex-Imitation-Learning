"""Board encoders for imitation-learning models.

The board is encoded as a full state (own stones, opponent stones, empty
cells, whose turn it is), not as move history. This matches the
behavioral-cloning framing used throughout this project: the network learns a
mapping from "current state" to "expert's action", so its input must fully
describe the state at decision time.
"""

from __future__ import annotations

import numpy as np

from hex_ml.core.board import EMPTY, PLAYER_ONE, PLAYER_TWO, validate_board


def encode_board_channels(board: np.ndarray, player_to_move: int) -> np.ndarray:
    """Encode a Hex board as channels with shape (4, size, size).

    Channels:
    0. current player's stones
    1. opponent player's stones
    2. empty cells
    3. player-to-move indicator, filled with 1 for player 1 and 2 for player 2
    """
    validate_board(board)
    if player_to_move not in {PLAYER_ONE, PLAYER_TWO}:
        raise ValueError("player_to_move must be 1 or 2")

    opponent = PLAYER_TWO if player_to_move == PLAYER_ONE else PLAYER_ONE
    size = board.shape[0]
    encoded = np.zeros((4, size, size), dtype=np.float32)
    encoded[0] = board == player_to_move
    encoded[1] = board == opponent
    encoded[2] = board == EMPTY
    encoded[3].fill(float(player_to_move))
    return encoded
