"""Tiny evaluation helpers for Hex agents."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol

import numpy as np

from hex_ml.core.board import (
    PLAYER_1,
    PLAYER_2,
    create_empty_board,
    get_legal_moves,
    is_legal_move,
)
from hex_ml.core.game import apply_move, check_winner, other_player


class HexAgentProtocol(Protocol):
    name: str
    agent_type: str

    def choose_move(
        self, board: np.ndarray, player_to_move: int
    ) -> tuple[int, int]: ...


@dataclass(frozen=True)
class EvaluationResult:
    games: int
    trained_agent_wins: int
    opponent_wins: int
    draws: int
    illegal_moves: int
    crashes: int
    win_rate: float
    opponent_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def play_game(
    agent_1: HexAgentProtocol, agent_2: HexAgentProtocol, board_size: int = 3
) -> tuple[int | None, int, int]:
    board = create_empty_board(board_size)
    player = PLAYER_1
    illegal_moves = 0
    crashes = 0
    agents = {PLAYER_1: agent_1, PLAYER_2: agent_2}

    for _turn in range(board_size * board_size):
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            return None, illegal_moves, crashes
        try:
            move = agents[player].choose_move(board.copy(), player)
        except Exception:
            crashes += 1
            return other_player(player), illegal_moves, crashes
        if not is_legal_move(board, move):
            illegal_moves += 1
            return other_player(player), illegal_moves, crashes
        board = apply_move(board, move, player)
        winner = check_winner(board)
        if winner is not None:
            return winner, illegal_moves, crashes
        player = other_player(player)
    return check_winner(board), illegal_moves, crashes


def evaluate_trained_agent(
    agent: HexAgentProtocol,
    opponent: HexAgentProtocol,
    games: int = 4,
    board_size: int = 3,
) -> EvaluationResult:
    trained_wins = 0
    opponent_wins = 0
    draws = 0
    illegal_moves = 0
    crashes = 0
    for game_index in range(games):
        if game_index % 2 == 0:
            winner, illegal, crash = play_game(agent, opponent, board_size)
            if winner == PLAYER_1:
                trained_wins += 1
            elif winner == PLAYER_2:
                opponent_wins += 1
            else:
                draws += 1
        else:
            winner, illegal, crash = play_game(opponent, agent, board_size)
            if winner == PLAYER_2:
                trained_wins += 1
            elif winner == PLAYER_1:
                opponent_wins += 1
            else:
                draws += 1
        illegal_moves += illegal
        crashes += crash
    return EvaluationResult(
        games=games,
        trained_agent_wins=trained_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        illegal_moves=illegal_moves,
        crashes=crashes,
        win_rate=trained_wins / games if games else 0.0,
        opponent_name=opponent.name,
    )
