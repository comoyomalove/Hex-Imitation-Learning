from hex_ml.data.schemas import ImitationRecord, MoveLogRecord


def test_move_log_record_contains_full_state_logging_fields():
    record = MoveLogRecord(
        game_id="game_001",
        turn_number=3,
        board_size=3,
        player_to_move=1,
        board_before=[[1, 0, 0], [0, 2, 0], [0, 0, 0]],
        legal_moves_before=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
        selected_move=(0, 1),
        agent_name="winner_agent",
        agent_type="expert",
        board_after=[[1, 1, 0], [0, 2, 0], [0, 0, 0]],
        was_legal=True,
        game_status_after_move="in_progress",
        final_winner=None,
        opponent_name="random_agent",
        opponent_type="baseline",
        timestamp="2026-06-18T18:40:00",
        path_costs=None,
        model_outputs=None,
    )

    assert record.game_id == "game_001"
    assert record.board_before[0][1] == 0
    assert record.board_after[0][1] == 1
    assert record.selected_move in record.legal_moves_before
    assert record.was_legal is True


def test_imitation_record_contains_leakage_safe_training_fields():
    record = ImitationRecord(
        game_id="game_001",
        turn_number=3,
        board_size=3,
        player_to_move=1,
        board_before=[[1, 0, 0], [0, 2, 0], [0, 0, 0]],
        legal_moves=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
        expert_move=(0, 1),
        expert_agent_name="winner_agent",
        opponent_name="random_agent",
        source_game_log="logs/games/game_001.json",
        split=None,
    )

    assert record.game_id == "game_001"
    assert record.expert_move in record.legal_moves
    assert record.split is None
    assert record.source_game_log.endswith("game_001.json")
