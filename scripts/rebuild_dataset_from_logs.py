from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from hex_ml.core.board import get_legal_moves, is_legal_move
from hex_ml.core.game import apply_move
from hex_ml.data.dataset_builder import imitation_records_from_game_log, save_imitation_records_jsonl
from hex_ml.data.game_log import board_from_list
from hex_ml.data.splitting import save_split_manifest, split_records_by_game_id


def audit_log(path: Path) -> tuple[list[str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    problems: list[str] = []
    if not records:
        return ["empty_records"], {"moves": 0}

    board = board_from_list(records[0]["board_before"])
    opponent = next((r["agent_name"] for r in records if r.get("agent_type") != "winner_agent"), "unknown")
    winner_examples = 0

    for i, record in enumerate(records):
        before = board_from_list(record["board_before"])
        after = board_from_list(record["board_after"])
        move = tuple(record["selected_move"])
        player = record["player_to_move"]

        if record.get("agent_type") == "winner_agent":
            winner_examples += 1
        if not np.array_equal(board, before):
            problems.append(f"turn_{i}_board_before_mismatch")
            break
        if sorted(get_legal_moves(before)) != sorted(tuple(x) for x in record["legal_moves_before"]):
            problems.append(f"turn_{i}_legal_moves_mismatch")
            break
        if not is_legal_move(before, move):
            problems.append(f"turn_{i}_illegal_move_{move}")
            break
        computed_after = apply_move(before, move, player)
        if not np.array_equal(computed_after, after):
            problems.append(f"turn_{i}_board_after_mismatch")
            break
        board = after

    meta = {
        "moves": len(records),
        "opponent": opponent,
        "winner_examples": winner_examples,
        "final_status": records[-1].get("game_status_after_move"),
        "final_winner": records[-1].get("final_winner"),
    }
    return problems, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    game_logs = sorted(args.log_dir.glob("game_*.json"))
    all_problems: dict[str, list[str]] = {}
    metas = []
    by_opponent: dict[str, Counter] = defaultdict(Counter)

    for path in game_logs:
        problems, meta = audit_log(path)
        if problems:
            all_problems[str(path)] = problems
            continue
        metas.append(meta)
        opp = meta["opponent"]
        by_opponent[opp]["games"] += 1
        by_opponent[opp]["moves"] += meta["moves"]
        by_opponent[opp]["winner_examples"] += meta["winner_examples"]
        by_opponent[opp][f"status_{meta['final_status']}"] += 1

    if all_problems:
        print(json.dumps({"problem_count": len(all_problems), "sample": dict(list(all_problems.items())[:5])}, indent=2))
        raise SystemExit(1)

    imitation_records = []
    for path in game_logs:
        imitation_records.extend(imitation_records_from_game_log(path))

    split_records, split_map = split_records_by_game_id(imitation_records, seed=args.seed)
    save_imitation_records_jsonl(args.records_path, split_records)
    save_split_manifest(args.split_path, split_map, seed=args.seed)

    move_lengths = [m["moves"] for m in metas]
    summary = {
        "log_dir": str(args.log_dir),
        "game_logs": len(game_logs),
        "problem_count": 0,
        "imitation_records": len(imitation_records),
        "records_path": str(args.records_path),
        "split_path": str(args.split_path),
        "split_counts_games": {k: len(v) for k, v in split_map.items()},
        "total_moves": sum(move_lengths),
        "avg_moves_per_game": mean(move_lengths) if move_lengths else 0,
        "min_moves_per_game": min(move_lengths) if move_lengths else 0,
        "max_moves_per_game": max(move_lengths) if move_lengths else 0,
        "by_opponent": {
            opponent: {
                **dict(counter),
                "avg_moves_per_game": counter["moves"] / counter["games"] if counter["games"] else 0,
            }
            for opponent, counter in sorted(by_opponent.items())
        },
    }

    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Partial 11x11 Dataset Rebuild Summary",
        "",
        f"- source log directory: `{args.log_dir}`",
        f"- game logs audited: {summary['game_logs']}",
        f"- audit problem count: {summary['problem_count']}",
        f"- imitation records: {summary['imitation_records']}",
        f"- total moves: {summary['total_moves']}",
        f"- average moves/game: {summary['avg_moves_per_game']:.2f}",
        f"- min/max moves/game: {summary['min_moves_per_game']} / {summary['max_moves_per_game']}",
        f"- records path: `{args.records_path}`",
        f"- split path: `{args.split_path}`",
        "",
        "## Split counts by game id",
        "",
        json.dumps(summary["split_counts_games"], indent=2),
        "",
        "## By opponent",
        "",
        "| Opponent | Games | Winner examples | Avg moves/game |",
        "|---|---:|---:|---:|",
    ]
    for opponent, data in summary["by_opponent"].items():
        lines.append(
            f"| {opponent} | {data['games']} | {data['winner_examples']} | {data['avg_moves_per_game']:.2f} |"
        )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
