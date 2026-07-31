"""Helpers for writing and replaying small full-state game logs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hex_ml.core.game import apply_move
from hex_ml.data.schemas import MoveLogRecord


def board_to_list(board: np.ndarray) -> list[list[int]]:
    return board.astype(int).tolist()


def board_from_list(board: list[list[int]]) -> np.ndarray:
    return np.array(board, dtype=np.int8)


def move_log_record_to_dict(record: MoveLogRecord) -> dict[str, Any]:
    data = asdict(record)
    data["legal_moves_before"] = [list(move) for move in record.legal_moves_before]
    data["selected_move"] = list(record.selected_move)
    return data


def save_game_log(path: Path, records: list[MoveLogRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "move_log_v1",
        "records": [move_log_record_to_dict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_game_log(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_game_log(path: Path) -> np.ndarray:
    payload = load_game_log(path)
    records = payload["records"]
    if not records:
        raise ValueError("Cannot replay an empty game log.")
    board = board_from_list(records[0]["board_before"])
    for record in records:
        expected_before = board_from_list(record["board_before"])
        if not np.array_equal(board, expected_before):
            raise ValueError(
                "Replay mismatch: board_before does not match current board."
            )
        move = tuple(record["selected_move"])
        board = apply_move(board, move, int(record["player_to_move"]))
        expected_after = board_from_list(record["board_after"])
        if not np.array_equal(board, expected_after):
            raise ValueError(
                "Replay mismatch: board_after does not match applied move."
            )
    return board
