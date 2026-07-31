"""Generate full-state logs for trained imitation agent vs winner agent.

Supports deterministic greedy play and smart opening exploration using top-k
sampling from the trained model. Winner-agent moves in these logs can later be
used as expert correction examples.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.trained_agent import TrainedHexAgent
from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord
from hex_ml.encoding.board_encoder import encode_board_channels


def _move_index(move: tuple[int, int], board_size: int) -> int:
    return int(move[0]) * board_size + int(move[1])


def _would_win(board, move: tuple[int, int], player: int) -> bool:
    return game_status(apply_move(board, move, player)) == f"winner_{player}"


def _allows_opponent_immediate_win(board, move: tuple[int, int], player: int) -> bool:
    after = apply_move(board, move, player)
    opponent = other_player(player)
    return any(_would_win(after, reply, opponent) for reply in get_legal_moves(after))


def _choose_external(agent: Any, board, player: int) -> tuple[int, int]:
    result = agent.choose_move(board, player)
    return result.move if hasattr(result, "move") else result


def _choose_trained(
    trained: TrainedHexAgent,
    board,
    player: int,
    job: dict[str, Any],
    trained_turn_index: int,
) -> tuple[tuple[int, int], dict[str, Any]]:
    legal_moves = get_legal_moves(board)

    if job.get("prefer_immediate_win"):
        for move in legal_moves:
            if _would_win(board, move, player):
                return move, {"selection_policy": "immediate_win", "trained_turn_index": trained_turn_index}

    should_sample = (
        job.get("trained_policy") == "opening_topk_sample"
        and trained_turn_index < int(job.get("opening_sample_moves", 0))
    )
    if not should_sample:
        move = _choose_external(trained, board, player)
        return move, {"selection_policy": "greedy", "trained_turn_index": trained_turn_index}

    device = next(trained.model.parameters()).device
    encoded = encode_board_channels(board, player)
    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = trained.model(x).squeeze(0).detach().cpu()

    board_size = board.shape[0]
    legal_indices = [_move_index(move, board_size) for move in legal_moves]
    legal_logits = logits[legal_indices]
    top_k = min(int(job.get("top_k", 5)), len(legal_moves))
    top_values, top_positions = torch.topk(legal_logits, k=top_k)
    candidates = [legal_moves[int(pos)] for pos in top_positions.tolist()]
    values = [float(v) for v in top_values.tolist()]

    if job.get("avoid_immediate_blunder"):
        safe = [(move, value) for move, value in zip(candidates, values) if not _allows_opponent_immediate_win(board, move, player)]
        if safe:
            candidates = [move for move, _ in safe]
            values = [value for _, value in safe]

    temperature = max(float(job.get("temperature", 0.8)), 1e-6)
    probs = torch.softmax(torch.tensor(values) / temperature, dim=0).tolist()
    rng = random.Random(int(job.get("seed", 0)) + int(job.get("game_index", 0)) * 1009 + trained_turn_index)
    threshold = rng.random()
    cumulative = 0.0
    chosen = candidates[-1]
    for move, prob in zip(candidates, probs):
        cumulative += float(prob)
        if threshold <= cumulative:
            chosen = move
            break

    return chosen, {
        "selection_policy": "opening_topk_sample",
        "trained_turn_index": trained_turn_index,
        "top_k": top_k,
        "temperature": temperature,
        "candidate_count": len(candidates),
    }


def play_game(job: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    trained = TrainedHexAgent.from_checkpoint(job["checkpoint"])
    winner = ExpertAgentWrapper()
    board_size = int(job["board_size"])
    trained_player = PLAYER_1 if job["trained_starts"] else PLAYER_2
    board = create_empty_board(board_size)
    player = PLAYER_1
    records: list[MoveLogRecord] = []
    trained_turn_index = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for turn_number in range(board_size * board_size):
        before = board.copy()
        legal = get_legal_moves(before)
        if not legal:
            break

        if player == trained_player:
            move, selection_info = _choose_trained(trained, before, player, job, trained_turn_index)
            trained_turn_index += 1
            agent_name = trained.name
            agent_type = trained.agent_type
            opponent_name = winner.agent_name
            opponent_type = "winner_agent"
            model_outputs = {"checkpoint": job["checkpoint"], **selection_info}
        else:
            move = _choose_external(winner, before, player)
            agent_name = winner.agent_name
            agent_type = "winner_agent"
            opponent_name = trained.name
            opponent_type = trained.agent_type
            model_outputs = None

        after = apply_move(before, move, player)
        status = game_status(after)
        final_winner = int(status.split("_")[1]) if status.startswith("winner_") else None
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
                final_winner=final_winner,
                opponent_name=opponent_name,
                opponent_type=opponent_type,
                timestamp=timestamp,
                model_outputs=model_outputs,
                path_costs={"collection_type": "trained_vs_winner_dagger_candidate"},
            )
        )
        board = after
        if final_winner is not None:
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
        "trained_starts": job["trained_starts"],
        "trained_player": trained_player,
        "turns": len(records),
        "elapsed_seconds": elapsed,
        "local_final_winner": records[-1].final_winner if records else None,
        "trained_local_win": trained_local_win,
    }


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index in range(args.games):
        trained_starts = index % 2 == 0
        role = "trained_p1" if trained_starts else "trained_p2"
        game_id = f"{args.run_id}_{index:04d}_{role}"
        path = args.output_dir / f"game_{game_id}.json"
        if args.skip_existing and path.exists():
            continue
        jobs.append(
            {
                "game_id": game_id,
                "game_index": index,
                "checkpoint": str(args.checkpoint),
                "board_size": args.board_size,
                "trained_starts": trained_starts,
                "output_dir": str(args.output_dir),
                "trained_policy": args.trained_policy,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "opening_sample_moves": args.opening_sample_moves,
                "seed": args.seed,
                "prefer_immediate_win": args.prefer_immediate_win,
                "avoid_immediate_blunder": args.avoid_immediate_blunder,
            }
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/final/imitation_11x11_best.pt"))
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--trained-policy", choices=["greedy", "opening_topk_sample"], default="greedy")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--opening-sample-moves", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prefer-immediate-win", action="store_true")
    parser.add_argument("--avoid-immediate-blunder", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

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
    trained_wins = sum(1 for r in results if r["trained_local_win"] is True)
    unknown = sum(1 for r in results if r["trained_local_win"] is None)
    summary = {
        "run_id": args.run_id,
        "checkpoint": str(args.checkpoint),
        "board_size": args.board_size,
        "workers": args.workers,
        "trained_policy": args.trained_policy,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "opening_sample_moves": args.opening_sample_moves,
        "prefer_immediate_win": args.prefer_immediate_win,
        "avoid_immediate_blunder": args.avoid_immediate_blunder,
        "games_requested": args.games,
        "jobs_after_skip_existing": len(jobs),
        "games_completed": len(results),
        "failures": failures,
        "total_elapsed_seconds": elapsed,
        "average_wall_seconds_per_game": elapsed / len(results) if results else None,
        "trained_local_wins": trained_wins,
        "unknown_winners": unknown,
        "expert_training_note": "Extract winner_agent moves from these logs for DAgger-style expert correction data.",
        "results": sorted(results, key=lambda item: item["game_id"]),
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
