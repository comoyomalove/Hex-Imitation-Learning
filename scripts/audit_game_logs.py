from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hex_ml.core.board import get_legal_moves, is_legal_move
from hex_ml.core.game import apply_move, check_winner
from hex_ml.data.game_log import board_from_list


def audit_dir(log_dir: Path) -> dict:
    summary = Counter()
    by_agent: dict[str, Counter] = defaultdict(Counter)
    problems: list[dict] = []

    for path in sorted(log_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append({"path": str(path), "problem": "json_error", "detail": repr(exc)})
            continue

        records = payload.get("records")
        if not isinstance(records, list):
            summary["non_game_json"] += 1
            continue
        summary["game_logs"] += 1
        if not records:
            problems.append({"path": str(path), "problem": "empty_records"})
            continue

        replay_board = board_from_list(records[0]["board_before"])
        for index, record in enumerate(records):
            agent_name = record.get("agent_name", "unknown")
            by_agent[agent_name]["moves"] += 1

            before = board_from_list(record["board_before"])
            after = board_from_list(record["board_after"])
            move = tuple(record["selected_move"])
            player = int(record["player_to_move"])

            if not np.array_equal(replay_board, before):
                problems.append({"path": str(path), "problem": "board_before_mismatch", "turn": index})

            recorded_legal = sorted(tuple(move) for move in record["legal_moves_before"])
            computed_legal = sorted(get_legal_moves(before))
            if recorded_legal != computed_legal:
                problems.append({"path": str(path), "problem": "legal_moves_mismatch", "turn": index})

            if not is_legal_move(before, move):
                problems.append({"path": str(path), "problem": "illegal_move", "turn": index, "move": move})
                by_agent[agent_name]["illegal_moves"] += 1
                continue

            computed_after = apply_move(before, move, player)
            if not np.array_equal(computed_after, after):
                problems.append({"path": str(path), "problem": "board_after_mismatch", "turn": index})

            replay_board = computed_after

        actual_winner = check_winner(replay_board)
        recorded_final = records[-1].get("final_winner")
        if recorded_final is not None and actual_winner != recorded_final:
            problems.append(
                {
                    "path": str(path),
                    "problem": "final_winner_mismatch",
                    "actual": actual_winner,
                    "recorded": recorded_final,
                }
            )
        summary["total_moves"] += len(records)

    return {
        "log_dir": str(log_dir),
        "summary": dict(summary),
        "by_agent": {name: dict(counts) for name, counts in sorted(by_agent.items())},
        "problem_count": len(problems),
        "problems_preview": problems[:50],
    }


if __name__ == "__main__":
    result = audit_dir(Path("logs/games/structured_opponents_stage2"))
    out = Path("logs/development/structured_stage2_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["problem_count"]:
        raise SystemExit(1)
