from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from hex_ml.agents.baselines import RandomLegalAgent
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.core.board import PLAYER_1, create_empty_board, get_legal_moves
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.dataset_builder import (
    imitation_records_from_game_log,
    save_imitation_records_jsonl,
)
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord
from hex_ml.data.splitting import save_split_manifest, split_records_by_game_id


def generate_one_game(
    game_index: int, board_size: int, output_dir: Path, seed: int
) -> Path:
    game_id = f"small_expert_vs_random_{game_index:03d}"
    timestamp = datetime.now(timezone.utc).isoformat()
    output_path = output_dir / f"game_{game_id}.json"
    expert = ExpertAgentWrapper()
    baseline = RandomLegalAgent(seed=seed + game_index)
    board = create_empty_board(board_size)
    player = PLAYER_1
    records = []
    for turn_number in range(board_size * board_size):
        before = board.copy()
        legal_moves = get_legal_moves(before)
        if not legal_moves:
            break
        if player == PLAYER_1:
            result = expert.choose_move(before, player)
            move = result.move
            agent_name = result.expert_name
            agent_type = result.source
            opponent_name = baseline.name
            opponent_type = baseline.agent_type
        else:
            move = baseline.choose_move(before, player)
            agent_name = baseline.name
            agent_type = baseline.agent_type
            opponent_name = expert.agent_name
            opponent_type = "winner_agent"
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
                opponent_name=opponent_name,
                opponent_type=opponent_type,
                timestamp=timestamp,
            )
        )
        board = after
        if winner is not None:
            break
        player = other_player(player)
    save_game_log(output_path, records)
    replay_game_log(output_path)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--board-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-dir", type=Path, default=Path("logs/games/small_dataset"))
    p.add_argument(
        "--records-path",
        type=Path,
        default=Path("data/processed/tiny_imitation_records.jsonl"),
    )
    p.add_argument(
        "--split-path", type=Path, default=Path("data/splits/tiny_split_manifest.json")
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [
        generate_one_game(i, args.board_size, args.output_dir, args.seed)
        for i in range(args.games)
    ]
    imitation_records = []
    for path in log_paths:
        imitation_records.extend(imitation_records_from_game_log(path))
    split_records, split_map = split_records_by_game_id(
        imitation_records, seed=args.seed
    )
    save_imitation_records_jsonl(args.records_path, split_records)
    save_split_manifest(args.split_path, split_map, seed=args.seed)
    print(f"games={len(log_paths)}")
    print(f"imitation_records={len(split_records)}")
    print(f"split_map={split_map}")
    print(f"records_path={args.records_path}")
    print(f"split_path={args.split_path}")


if __name__ == "__main__":
    main()
