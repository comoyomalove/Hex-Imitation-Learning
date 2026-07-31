from pathlib import Path

from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.game import apply_move, game_status
from hex_ml.data.game_log import (
    board_to_list,
    load_game_log,
    replay_game_log,
    save_game_log,
)
from hex_ml.data.schemas import MoveLogRecord


def test_save_and_replay_tiny_full_state_game_log(tmp_path: Path):
    game_id = "tiny_game_001"
    board = create_empty_board(3)
    records = []
    moves = [(PLAYER_1, (0, 0), "expert"), (PLAYER_2, (1, 1), "random")]

    for turn_number, (player, move, agent_type) in enumerate(moves):
        before = board.copy()
        legal_moves = get_legal_moves(before)
        after = apply_move(before, move, player)
        records.append(
            MoveLogRecord(
                game_id=game_id,
                turn_number=turn_number,
                board_size=3,
                player_to_move=player,
                board_before=board_to_list(before),
                legal_moves_before=legal_moves,
                selected_move=move,
                agent_name=f"{agent_type}_agent",
                agent_type=agent_type,
                board_after=board_to_list(after),
                was_legal=True,
                game_status_after_move=game_status(after),
                final_winner=None,
                opponent_name="test_opponent",
                opponent_type="test",
                timestamp="2026-06-18T18:41:00",
            )
        )
        board = after

    log_path = tmp_path / "tiny_game.json"
    save_game_log(log_path, records)
    payload = load_game_log(log_path)
    assert payload["schema_version"] == "move_log_v1"
    assert payload["records"][0]["game_id"] == game_id
    assert payload["records"][0]["board_before"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    replayed = replay_game_log(log_path)
    assert replayed.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 0]]
