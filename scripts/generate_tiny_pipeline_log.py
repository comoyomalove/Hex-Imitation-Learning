from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.core.board import PLAYER_1, create_empty_board, get_legal_moves
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord


def main() -> Path:
    board_size = 3
    max_turns = 4
    game_id = "tiny_real_expert_pipeline_001"
    timestamp = datetime.now(timezone.utc).isoformat()
    output_path = Path("logs/games/game_tiny_real_expert_pipeline_001.json")
    wrapper = ExpertAgentWrapper()
    board = create_empty_board(board_size)
    player = PLAYER_1
    records = []
    for turn_number in range(max_turns):
        before = board.copy()
        legal_moves = get_legal_moves(before)
        if not legal_moves:
            break
        if player == PLAYER_1:
            result = wrapper.choose_move(before, player)
            move = result.move
            agent_name = result.expert_name
            agent_type = result.source
        else:
            move = legal_moves[0]
            agent_name = "deterministic_first_legal_baseline"
            agent_type = "baseline"
        after = apply_move(before, move, player)
        status = game_status(after)
        winner = int(status.split("_")[1]) if status.startswith("winner_") else None
        records.append(
            MoveLogRecord(
                game_id=game_id,
                turn_number=turn_number,
                board_size=board_size,
                player_to_move=player,
                board_before=board_to_list(before),
                legal_moves_before=legal_moves,
                selected_move=move,
                agent_name=agent_name,
                agent_type=agent_type,
                board_after=board_to_list(after),
                was_legal=True,
                game_status_after_move=status,
                final_winner=winner,
                opponent_name=(
                    "deterministic_first_legal_baseline"
                    if player == PLAYER_1
                    else "hackathon_winner_agent"
                ),
                opponent_type="baseline" if player == PLAYER_1 else "winner_agent",
                timestamp=timestamp,
            )
        )
        board = after
        if winner is not None:
            break
        player = other_player(player)
    save_game_log(output_path, records)
    replayed = replay_game_log(output_path)
    print(f"saved={output_path}")
    print(f"records={len(records)}")
    print(f"final_board={replayed.tolist()}")
    print(f"wrapper_available={wrapper.available}")
    print(f'wrapper_load_error={repr(getattr(wrapper, "_load_error", None))}')
    print(
        f'expert_record_sources={[record.agent_type for record in records if record.agent_name == "hackathon_winner_agent"]}'
    )
    return output_path


if __name__ == "__main__":
    main()
