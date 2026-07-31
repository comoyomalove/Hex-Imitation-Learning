"""Generate demo sample games for browser replay.

Creates one game with the trained agent as Player 1 and one game with each
opponent as Player 1, then writes demo_hex_board/sample_games.js.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hex_ml.agents.baselines import RandomLegalAgent
from hex_ml.agents.discovery import discover_gamelib_agents
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.agents.trained_agent import TrainedHexAgent
from hex_ml.core.board import PLAYER_1, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, save_game_log
from hex_ml.data.schemas import MoveLogRecord

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "games" / "demo_samples_all_agents"
SAMPLE_JS = ROOT / "demo_hex_board" / "sample_games.js"
BOARD_SIZE = DEFAULT_BOARD_SIZE


def agent_specs() -> list[dict[str, Any]]:
    specs = [
        {"id": "random_legal_agent", "label": "Random legal baseline", "kind": "random"},
        {"id": "hackathon_winner_agent", "label": "Old hackathon winner / expert", "kind": "winner"},
        {"id": "trained_imitation_agent", "label": "Trained imitation agent", "kind": "trained"},
    ]
    for spec in discover_gamelib_agents():
        specs.append({"id": spec.name, "label": spec.name, "kind": "external", "path": str(spec.path)})
    order = {"trained_imitation_agent": 0, "hackathon_winner_agent": 1, "v7_resistance_agent": 2, "random_legal_agent": 3}
    return sorted(specs, key=lambda x: (order.get(x["id"], 10), x["id"]))


def make_agent(spec: dict[str, Any], seed: int = 0):
    if spec["kind"] == "trained":
        agent = TrainedHexAgent.from_checkpoint(ROOT / "models" / "final" / "imitation_11x11_best.pt")
        return agent, agent.name, agent.agent_type
    if spec["kind"] == "winner":
        agent = ExpertAgentWrapper()
        return agent, agent.agent_name, "winner_agent"
    if spec["kind"] == "random":
        return RandomLegalAgent(seed=seed), "random_legal_agent", "baseline_random"
    agent = GamelibAgentWrapper(Path(spec["path"]), agent_name=spec["id"], agent_type="external_gamelib_agent")
    return agent, spec["id"], "external_gamelib_agent"


def choose(agent: Any, board, player: int):
    result = agent.choose_move(board, player)
    return result.move if hasattr(result, "move") else result


def play_game(game_id: str, p1_spec: dict[str, Any], p2_spec: dict[str, Any]) -> dict[str, Any]:
    p1, p1_name, p1_type = make_agent(p1_spec, seed=1)
    p2, p2_name, p2_type = make_agent(p2_spec, seed=2)
    board = create_empty_board(BOARD_SIZE)
    player = PLAYER_1
    records: list[MoveLogRecord] = []
    timings = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for turn in range(BOARD_SIZE * BOARD_SIZE):
        before = board.copy()
        legal = get_legal_moves(before)
        if player == PLAYER_1:
            agent, name, typ, opp_name, opp_type = p1, p1_name, p1_type, p2_name, p2_type
        else:
            agent, name, typ, opp_name, opp_type = p2, p2_name, p2_type, p1_name, p1_type
        start = time.perf_counter()
        move = choose(agent, before, player)
        elapsed = time.perf_counter() - start
        after = apply_move(before, move, player)
        status = game_status(after)
        winner = int(status.split("_")[1]) if status.startswith("winner_") else None
        records.append(
            MoveLogRecord(
                game_id=game_id,
                turn_number=turn,
                board_size=BOARD_SIZE,
                player_to_move=player,
                board_before=board_to_list(before),
                legal_moves_before=legal,
                selected_move=move,
                agent_name=name,
                agent_type=typ,
                board_after=board_to_list(after),
                was_legal=True,
                game_status_after_move=status,
                final_winner=winner,
                opponent_name=opp_name,
                opponent_type=opp_type,
                timestamp=timestamp,
                model_outputs={"move_time_seconds": elapsed},
            )
        )
        timings.append({"turn_number": turn, "agent_name": name, "player": player, "move": list(move), "seconds": elapsed})
        board = after
        if winner is not None:
            break
        player = other_player(player)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"game_{game_id}.json"
    save_game_log(path, records)
    (OUT_DIR / f"timings_{game_id}.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return {"path": path, "game_id": game_id, "p1": p1_name, "p2": p2_name, "turns": len(records), "final_winner": records[-1].final_winner if records else None}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = agent_specs()
    trained = next(s for s in specs if s["id"] == "trained_imitation_agent")
    entries = []
    summaries = []
    for spec in specs:
        if spec["id"] == "trained_imitation_agent":
            continue
        jobs = [
            (f"demo_trained_p1_vs_{spec['id']}", trained, spec),
            (f"demo_{spec['id']}_p1_vs_trained", spec, trained),
        ]
        for game_id, p1, p2 in jobs:
            summary = play_game(game_id, p1, p2)
            summaries.append(summary)
            data = json.loads(summary["path"].read_text(encoding="utf-8"))
            records = data.get("records", [])
            entries.append({
                "id": summary["path"].stem,
                "label": f"{summary['p1']} P1 vs {summary['p2']} P2 ({summary['turns']} moves)",
                "data": {"game_id": records[0]["game_id"] if records else game_id, "records": records},
            })
    SAMPLE_JS.write_text("const SAMPLE_GAMES = " + json.dumps(entries) + ";\n", encoding="utf-8")
    payload = {
        "output_dir": str(OUT_DIR),
        "sample_js": str(SAMPLE_JS),
        "agents_tested": [s["id"] for s in specs if s["id"] != "trained_imitation_agent"],
        "games_written": len(entries),
        "games": [{k: (str(v) if isinstance(v, Path) else v) for k, v in row.items()} for row in summaries],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
