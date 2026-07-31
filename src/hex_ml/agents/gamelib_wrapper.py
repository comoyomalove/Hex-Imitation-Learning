from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hex_ml.core.board import Move, get_legal_moves, is_legal_move, validate_board


@dataclass(frozen=True)
class GamelibAgentMoveResult:
    move: Move
    agent_name: str
    agent_type: str
    source_path: str

    @property
    def expert_name(self) -> str:
        return self.agent_name

    @property
    def source(self) -> str:
        return self.agent_type


class GamelibAgentWrapper:
    def __init__(self, agent_path: str | Path, agent_name: str | None = None, agent_type: str = 'external_gamelib_agent') -> None:
        self.agent_path = Path(agent_path)
        self.agent_name = agent_name or self.agent_path.parent.name
        self.agent_type = agent_type
        self._agent_class: Any | None = None
        self._load_error: Exception | None = None
        self._load_agent_class()

    @property
    def available(self) -> bool:
        return self._agent_class is not None

    def choose_move(self, board: np.ndarray, player_to_move: int) -> GamelibAgentMoveResult:
        validate_board(board)
        board_copy = board.copy()
        legal_moves = get_legal_moves(board_copy)
        if not legal_moves:
            raise ValueError('No legal moves available.')
        move = self._choose_with_agent(board_copy, player_to_move, legal_moves)
        if not is_legal_move(board, move):
            raise ValueError(f'External agent produced illegal move: {move}')
        if not np.array_equal(board, board_copy):
            raise RuntimeError('External agent wrapper mutated the input board.')
        return GamelibAgentMoveResult(move=move, agent_name=self.agent_name, agent_type=self.agent_type, source_path=str(self.agent_path))

    def _load_agent_class(self) -> None:
        if not self.agent_path.exists():
            self._load_error = FileNotFoundError(self.agent_path)
            return
        try:
            module_name = 'hex_ml_external_agent_' + self.agent_path.parent.name.replace('-', '_')
            spec = importlib.util.spec_from_file_location(module_name, self.agent_path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Could not load spec for {self.agent_path}')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._agent_class = _find_agent_class(module)
        except Exception as exc:
            self._load_error = exc
            self._agent_class = None

    def _choose_with_agent(self, board: np.ndarray, player_to_move: int, legal_moves: list[Move]) -> Move:
        if self._agent_class is None:
            raise RuntimeError(f'Agent is unavailable: {self.agent_path}; error={self._load_error!r}')
        agent = self._agent_class()
        if hasattr(agent, 'initialize'):
            agent.initialize({'board_size': int(board.shape[0]), 'player_id': _to_gamelib_player(player_to_move)})
        state = make_gamelib_state(board.copy(), player_to_move)
        raw_move = agent.get_move(state)
        move = normalise_gamelib_move(raw_move)
        if move not in legal_moves:
            raise ValueError(f'Agent {self.agent_name} returned illegal move {move}')
        return move



def _find_agent_class(module: Any) -> Any:
    if hasattr(module, "HexAgent"):
        return getattr(module, "HexAgent")
    candidates = []
    for name, value in module.__dict__.items():
        if name == "Agent":
            continue
        if isinstance(value, type) and hasattr(value, "get_move"):
            candidates.append(value)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise AttributeError("No compatible agent class found. Expected HexAgent or one non-base class with get_move().")
    names = [candidate.__name__ for candidate in candidates]
    raise AttributeError(f"Multiple compatible agent classes found: {names}")

def _to_gamelib_player(player: int) -> int:
    if player == 1:
        return 0
    if player == 2:
        return 1
    raise ValueError(f'Invalid local player id: {player}')


def to_gamelib_board(board: np.ndarray) -> list[list[int]]:
    return [[-1 if cell == 0 else 0 if cell == 1 else 1 for cell in row] for row in board.tolist()]


def make_gamelib_state(board: np.ndarray, player_to_move: int) -> Any:
    from gamelib.hex.gamestate import GameState
    return GameState(board_size=int(board.shape[0]), board=to_gamelib_board(board), turn=_to_gamelib_player(player_to_move), status=0)


def normalise_gamelib_move(raw_move: Any) -> Move:
    if isinstance(raw_move, tuple) and len(raw_move) == 2:
        return (int(raw_move[0]), int(raw_move[1]))
    if hasattr(raw_move, 'position'):
        pos = raw_move.position
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            return (int(pos[0]), int(pos[1]))
    if hasattr(raw_move, 'row') and hasattr(raw_move, 'col'):
        return (int(raw_move.row), int(raw_move.col))
    if hasattr(raw_move, 'x') and hasattr(raw_move, 'y'):
        return (int(raw_move.x), int(raw_move.y))
    raise TypeError(f'Unsupported move object from gamelib agent: {raw_move!r}')
