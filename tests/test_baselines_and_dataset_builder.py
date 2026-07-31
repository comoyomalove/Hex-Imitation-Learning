from pathlib import Path
from hex_ml.agents.baselines import FirstLegalAgent, RandomLegalAgent
from hex_ml.core.board import PLAYER_1, create_empty_board, is_legal_move
from hex_ml.data.dataset_builder import (
    imitation_records_from_game_log,
    load_imitation_records_jsonl,
    save_imitation_records_jsonl,
)
from hex_ml.data.game_log import board_to_list, save_game_log
from hex_ml.data.schemas import MoveLogRecord


def test_baseline_agents_return_legal_moves():
    board = create_empty_board(3)
    assert is_legal_move(board, FirstLegalAgent().choose_move(board, PLAYER_1))
    assert is_legal_move(board, RandomLegalAgent(seed=1).choose_move(board, PLAYER_1))


def test_build_imitation_records_from_winner_agent_moves(tmp_path: Path):
    log_path = tmp_path / "game.json"
    board = create_empty_board(3)
    record = MoveLogRecord(
        game_id="g1",
        turn_number=0,
        board_size=3,
        player_to_move=PLAYER_1,
        board_before=board_to_list(board),
        legal_moves_before=[(0, 0), (1, 1)],
        selected_move=(1, 1),
        agent_name="hackathon_winner_agent",
        agent_type="winner_agent",
        board_after=board_to_list(board),
        was_legal=True,
        game_status_after_move="in_progress",
        final_winner=None,
        opponent_name="random",
        opponent_type="baseline",
        timestamp="2026-06-18T19:09:00",
    )
    save_game_log(log_path, [record])
    records = imitation_records_from_game_log(log_path)
    assert len(records) == 1
    assert records[0].game_id == "g1"
    assert records[0].expert_move == (1, 1)
    jsonl_path = tmp_path / "records.jsonl"
    save_imitation_records_jsonl(jsonl_path, records)
    loaded = load_imitation_records_jsonl(jsonl_path)
    assert loaded[0]["expert_move"] == [1, 1]
