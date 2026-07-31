from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from hex_ml.agents.discovery import discover_gamelib_agents
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.dataset_builder import imitation_records_from_game_log, save_imitation_records_jsonl
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord
from hex_ml.data.splitting import save_split_manifest, split_records_by_game_id


def play_game(game_id, board_size, expert_starts, opponent_spec, output_dir):
    start = time.perf_counter()
    expert = ExpertAgentWrapper()
    opponent = GamelibAgentWrapper(
        opponent_spec.path,
        agent_name=opponent_spec.name,
        agent_type="structured_opponent",
    )
    board = create_empty_board(board_size)
    player = PLAYER_1
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    expert_player = PLAYER_1 if expert_starts else PLAYER_2

    for turn_number in range(board_size * board_size):
        before = board.copy()
        legal = get_legal_moves(before)
        if not legal:
            break

        if player == expert_player:
            res = expert.choose_move(before, player)
            move = res.move
            agent_name = res.expert_name
            agent_type = "winner_agent"
            opponent_name = opponent_spec.name
            opponent_type = "structured_opponent"
        else:
            res = opponent.choose_move(before, player)
            move = res.move
            agent_name = opponent_spec.name
            agent_type = "structured_opponent"
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
                legal_moves_before=legal,
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
                path_costs={"opponent_strategy": opponent_spec.name},
            )
        )
        board = after
        if winner is not None:
            break
        player = other_player(player)

    path = output_dir / f"game_{game_id}.json"
    save_game_log(path, records)
    replay_game_log(path)
    elapsed_seconds = time.perf_counter() - start
    return {
        "path": str(path),
        "game_id": game_id,
        "agent": opponent_spec.name,
        "expert_starts": expert_starts,
        "turns": len(records),
        "elapsed_seconds": elapsed_seconds,
        "winner": records[-1].final_winner if records else None,
    }


def build_jobs(specs, games_per_agent, board_size, output_dir):
    jobs = []
    for spec in specs:
        for i in range(games_per_agent):
            for expert_starts in (True, False):
                role = "expert_p1" if expert_starts else "expert_p2"
                gid = f"structured_{spec.name}_{i:03d}_{role}"
                jobs.append((gid, board_size, expert_starts, spec, output_dir))
    return jobs


def run_jobs(jobs, workers):
    results = []
    failures = []
    if workers <= 1:
        for job in jobs:
            try:
                results.append(play_game(*job))
            except Exception as exc:
                failures.append({"game": job[0], "agent": job[3].name, "error": repr(exc)})
        return results, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(play_game, *job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"game": job[0], "agent": job[3].name, "error": repr(exc)})
    return results, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-agent", type=int, default=1)
    ap.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include-agents", nargs="*", default=None, help="Optional agent names to include.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip jobs whose game log already exists.")
    ap.add_argument("--output-dir", type=Path, default=Path("logs/games/structured_opponents_stage1"))
    ap.add_argument("--records-path", type=Path, default=Path("data/processed/structured_stage1_imitation_records.jsonl"))
    ap.add_argument("--split-path", type=Path, default=Path("data/splits/structured_stage1_split_manifest.json"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_gamelib_agents()
    if args.include_agents:
        requested = set(args.include_agents)
        specs = [spec for spec in specs if spec.name in requested]
        found = {spec.name for spec in specs}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested agents not found: {missing}")
    jobs = build_jobs(specs, args.games_per_agent, args.board_size, args.output_dir)
    if args.skip_existing:
        jobs = [job for job in jobs if not (args.output_dir / f"game_{job[0]}.json").exists()]
    run_start = time.perf_counter()
    results, failures = run_jobs(jobs, args.workers)
    total_elapsed_seconds = time.perf_counter() - run_start

    paths = [Path(result["path"]) for result in results]
    records = []
    for path in paths:
        records.extend(imitation_records_from_game_log(path))

    if records:
        split_records, split_map = split_records_by_game_id(records, seed=42)
        save_imitation_records_jsonl(args.records_path, split_records)
        save_split_manifest(args.split_path, split_map, seed=42)
    else:
        split_map = {"train": [], "validation": [], "test": []}
        save_imitation_records_jsonl(args.records_path, [])

    summary = {
        "agents": [s.name for s in specs],
        "board_size": args.board_size,
        "workers": args.workers,
        "include_agents": args.include_agents,
        "skip_existing": args.skip_existing,
        "jobs_requested": len(jobs),
        "games_written": len(paths),
        "failures": failures,
        "imitation_records": len(records),
        "total_elapsed_seconds": total_elapsed_seconds,
        "average_seconds_per_completed_game": total_elapsed_seconds / len(paths) if paths else None,
        "game_results": sorted(results, key=lambda item: item["game_id"]),
        "records_path": str(args.records_path),
        "split_path": str(args.split_path),
        "split_counts": {name: len(ids) for name, ids in split_map.items()},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
