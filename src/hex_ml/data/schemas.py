"""Schema helpers for logs and imitation-learning records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Move = tuple[int, int]
BoardList = list[list[int]]


@dataclass(frozen=True)
class MoveLogRecord:
    """One full-state move record from a generated Hex game."""

    game_id: str
    turn_number: int
    board_size: int
    player_to_move: int
    board_before: BoardList
    legal_moves_before: list[Move]
    selected_move: Move
    agent_name: str
    agent_type: str
    board_after: BoardList
    was_legal: bool
    game_status_after_move: str
    final_winner: int | None
    opponent_name: str
    opponent_type: str
    timestamp: str
    path_costs: dict[str, Any] | None = None
    model_outputs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImitationRecord:
    """One supervised learning example derived from an expert move."""

    game_id: str
    turn_number: int
    board_size: int
    player_to_move: int
    board_before: BoardList
    legal_moves: list[Move]
    expert_move: Move
    expert_agent_name: str
    opponent_name: str
    source_game_log: str
    split: str | None = None
