"""Evaluate trained imitation agent against random, external agents, and winner agent.

This logs full-state evaluation games. Winner-agent moves in trained-vs-winner
logs can later be extracted as expert demonstrations for DAgger-style follow-up.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hex_ml.agents.baselines import RandomLegalAgent
from hex_ml.agents.discovery import discover_gamelib_agents
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.agents.trained_agent import TrainedHexAgent
from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord


def make_opponent(kind: str, name: str | None, seed: int | None = None) -> tuple[Any, str, str]:
    if kind == "random":
        return RandomLegalAgent(seed=seed), "random_legal_agent", "baseline_random"
    if kind == "winner":
        agent = ExpertAgentWrapper()
        return agent, agent.agent_name, "winner_agent"
    if kind == "external":
        specs = {spec.name: spec for spec in discover_gamelib_agents()}
        if name not in specs:
            raise ValueError(f"Unknown external agent: {name}")
        spec = specs[name]
        agent = GamelibAgentWrapper(spec.path, agent_name=spec.name, agent_type="structured_opponent")
        return agent, spec.name, "structured_opponent"
    raise ValueError(f"Unknown opponent kind: {kind}")


def choose(agent: Any, board, player: int):
    result = agent.choose_move(board, player)
    if hasattr(result, "move"):
        return result.move
    return result


def play_game(job: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    board_size = job["board_size"]
    trained = TrainedHexAgent.from_checkpoint(job["checkpoint"])
    opponent, opponent_name, opponent_type = make_opponent(
        job["opponent_kind"], job.get("opponent_name"), seed=job["game_index"]
    )
    trained_player = PLAYER_1 if job["trained_starts"] else PLAYER_2
    board = create_empty_board(board_size)
    player = PLAYER_1
    records: list[MoveLogRecord] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for turn_number in range(board_size * board_size):
        before = board.copy()
        legal = get_legal_moves(before)
        if not legal:
            break

        if player == trained_player:
            move = choose(trained, before, player)
            agent_name = trained.name
            agent_type = trained.agent_type
            opp_name = opponent_name
            opp_type = opponent_type
        else:
            move = choose(opponent, before, player)
            agent_name = opponent_name
            agent_type = opponent_type
            opp_name = trained.name
            opp_type = trained.agent_type

        after = apply_move(before, move, player)
        status = game_status(after)
        winner = int(status.split("_")[1]) if status.startswith("winner_") else None
        records.append(
            MoveLogRecord(
                game_id=job["game_id"],
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
                opponent_name=opp_name,
                opponent_type=opp_type,
                timestamp=timestamp,
                model_outputs={"checkpoint": job["checkpoint"]} if agent_type == "trained_model" else None,
            )
        )
        board = after
        if winner is not None:
            break
        player = other_player(player)

    path = Path(job["output_dir"]) / f"game_{job['game_id']}.json"
    save_game_log(path, records)
    replay_game_log(path)
    elapsed = time.perf_counter() - start
    trained_local_win = None
    if records and records[-1].final_winner is not None:
        trained_local_win = records[-1].final_winner == trained_player
    return {
        "game_id": job["game_id"],
        "path": str(path),
        "opponent_kind": job["opponent_kind"],
        "opponent_name": opponent_name,
        "trained_starts": job["trained_starts"],
        "trained_player": trained_player,
        "turns": len(records),
        "elapsed_seconds": elapsed,
        "local_final_winner": records[-1].final_winner if records else None,
        "trained_local_win": trained_local_win,
    }


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    external_names = [spec.name for spec in discover_gamelib_agents()]
    opponents: list[tuple[str, str | None]] = [("random", None), ("winner", None)]
    opponents.extend(("external", name) for name in external_names)

    jobs = []
    for kind, name in opponents:
        label = name or kind
        for i in range(args.games_per_opponent):
            trained_starts = i % 2 == 0
            game_id = f"eval_{label}_{i:03d}_{'trained_p1' if trained_starts else 'trained_p2'}"
            jobs.append(
                {
                    "game_id": game_id,
                    "game_index": i,
                    "opponent_kind": kind,
                    "opponent_name": name,
                    "trained_starts": trained_starts,
                    "checkpoint": str(args.checkpoint),
                    "board_size": args.board_size,
                    "output_dir": str(args.output_dir),
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/final/imitation_11x11_best.pt"))
    parser.add_argument("--games-per-opponent", type=int, default=5)
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/games/evaluation_round1"))
    parser.add_argument("--summary-path", type=Path, default=Path("logs/evaluation/eval_round1_summary.json"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/evaluation/eval_round1_summary.md"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    start = time.perf_counter()
    results = []
    failures = []

    if args.workers <= 1:
        for job in jobs:
            try:
                results.append(play_game(job))
            except Exception as exc:  # noqa: BLE001
                failures.append({"game_id": job["game_id"], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(play_game, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append({"game_id": job["game_id"], "error": repr(exc)})

    elapsed = time.perf_counter() - start
    by_opponent: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result["opponent_name"]
        row = by_opponent.setdefault(
            key,
            {"games": 0, "trained_local_wins": 0, "unknown_winners": 0, "turns": 0, "elapsed_seconds": 0.0},
        )
        row["games"] += 1
        row["turns"] += result["turns"]
        row["elapsed_seconds"] += result["elapsed_seconds"]
        if result["trained_local_win"] is True:
            row["trained_local_wins"] += 1
        if result["trained_local_win"] is None:
            row["unknown_winners"] += 1
    for row in by_opponent.values():
        row["avg_turns"] = row["turns"] / row["games"]
        row["avg_elapsed_seconds"] = row["elapsed_seconds"] / row["games"]
        row["trained_local_win_rate"] = row["trained_local_wins"] / row["games"]

    summary = {
        "checkpoint": str(args.checkpoint),
        "board_size": args.board_size,
        "workers": args.workers,
        "games_requested": len(jobs),
        "games_completed": len(results),
        "failures": failures,
        "total_elapsed_seconds": elapsed,
        "average_wall_seconds_per_game": elapsed / len(results) if results else None,
        "by_opponent": by_opponent,
        "results": sorted(results, key=lambda item: item["game_id"]),
        "orientation_caveat": "Local winner labels may not fully match gamelib orientation; use win rates cautiously.",
        "expert_training_note": "Winner-agent moves in trained-vs-winner logs may later be extracted as expert demonstrations.",
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Evaluation Round 1 Summary",
        "",
        f"- checkpoint: {args.checkpoint}",
        f"- games completed: {len(results)}",
        f"- failures: {len(failures)}",
        f"- total elapsed seconds: {elapsed:.2f}",
        "",
        "## By opponent",
        "",
        "| Opponent | Games | Local trained wins | Local win rate | Avg turns | Avg seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(by_opponent.items()):
        lines.append(
            f"| {name} | {row['games']} | {row['trained_local_wins']} | "
            f"{row['trained_local_win_rate']:.2f} | {row['avg_turns']:.1f} | {row['avg_elapsed_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Caveat",
        "Local winner labels may not fully match gamelib orientation. Use win/loss values cautiously until orientation is verified.",
        "",
        "## Future training note",
        "Winner-agent moves from trained-vs-winner games can later be extracted as expert demonstrations for DAgger-style follow-up training.",
    ]
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
