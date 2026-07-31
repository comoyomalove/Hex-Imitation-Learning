"""Inference wrapper for a trained Hex imitation policy."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from hex_ml.core.board import Move, get_legal_moves, validate_board
from hex_ml.encoding.board_encoder import encode_board_channels
from hex_ml.training.model import TinyHexPolicyNet


def mask_illegal_moves(
    logits: torch.Tensor, legal_moves: list[Move], board_size: int
) -> torch.Tensor:
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported for single-agent inference."
            )
        flat_logits = logits[0].clone()
    elif logits.ndim == 1:
        flat_logits = logits.clone()
    else:
        raise ValueError("logits must be 1D or batch-size-1 2D tensor.")
    expected = board_size * board_size
    if flat_logits.numel() != expected:
        raise ValueError(
            f"Expected {expected} logits for board size {board_size}, got {flat_logits.numel()}."
        )
    if not legal_moves:
        raise ValueError("Cannot mask logits because there are no legal moves.")
    masked = torch.full_like(flat_logits, -1e9)
    for row, col in legal_moves:
        masked[row * board_size + col] = flat_logits[row * board_size + col]
    return masked


class TrainedHexAgent:
    name = "trained_imitation_agent"
    agent_type = "trained_model"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, map_location: str = "cpu"
    ) -> "TrainedHexAgent":
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=map_location
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise ValueError(
                "Checkpoint does not contain model_state_dict or state_dict."
            )
        config = checkpoint.get("model_config", {})
        board_size = int(config.get("board_size", checkpoint.get("board_size", 3)))
        input_channels = int(config.get("input_channels", config.get("in_channels", 4)))
        hidden_size = int(
            config.get(
                "hidden_size",
                config.get("hidden_channels", _infer_hidden_size(state_dict)),
            )
        )
        model = TinyHexPolicyNet(
            board_size=board_size,
            input_channels=input_channels,
            hidden_size=hidden_size,
        )
        model.load_state_dict(state_dict)
        return cls(model)

    @torch.no_grad()
    def choose_move(self, board: np.ndarray, player_to_move: int) -> Move:
        validate_board(board)
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        board_size = int(board.shape[0])
        encoded = encode_board_channels(board, player_to_move)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        logits = self.model(x)
        masked = mask_illegal_moves(logits, legal_moves, board_size)
        selected_index = int(torch.argmax(masked).item())
        return divmod(selected_index, board_size)


def _infer_hidden_size(state_dict: dict[str, torch.Tensor]) -> int:
    first_weight = state_dict.get("net.1.weight")
    if first_weight is None:
        return 64
    return int(first_weight.shape[0])
