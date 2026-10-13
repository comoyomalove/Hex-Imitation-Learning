"""Optional local backend for the Hex board browser demo.

Static fallback:
    Open demo_hex_board/index.html directly in a browser to replay bundled logs.

Live backend mode:
    cd /Users/nazthebussinessprincess/Dev/agentic/Hex-ML
    source .venv/bin/activate
    PYTHONPATH=src python scripts/demo_server.py
    open http://127.0.0.1:5050
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
DEMO_DIR = ROOT / "demo_hex_board"
LIVE_LOG_DIR = ROOT / "logs" / "games" / "demo_live"
HOST = "127.0.0.1"
PORT = 5050
GAMES: dict[str, dict[str, Any]] = {}


AGENT_CREDITS: dict[str, dict[str, str]] = {
    "trained_imitation_agent": {
        "author": "Naz (this project)",
        "note": "ML policy trained by imitation learning - the subject of this project.",
    },
    "hackathon_winner_agent": {
        "author": "Adyansh Gupta",
        "note": "Hackathon winner agent; used as the expert/teacher for imitation learning.",
    },
    "v7_resistance_agent": {
        "author": "Adyansh Gupta",
        "note": "Later candidate expert agent by the same author as the hackathon winner.",
    },
    "random_legal_agent": {
        "author": "Naz (this project)",
        "note": "Utility baseline that picks a uniformly random legal move.",
    },
    "hybrid_phase_agent": {
        "author": "Naz (earlier hackathon work)",
        "note": "Algorithm-based (search/heuristic) agent, not ML. Used as a structured opponent for data generation.",
    },
    "mcts_lite_hex_agent": {
        "author": "Naz (earlier hackathon work)",
        "note": "Algorithm-based (lightweight MCTS-style) agent, not ML. Used as a structured opponent for data generation.",
    },
    "minimax_lite_hex_agent": {
        "author": "Naz (earlier hackathon work)",
        "note": "Algorithm-based (lightweight minimax-style) agent, not ML. Used as a structured opponent for data generation.",
    },
    "smart_hex_agent": {
        "author": "Naz (earlier hackathon work)",
        "note": "Algorithm-based (Dijkstra/path-based) agent, not ML. Used as a structured opponent for data generation.",
    },
    "threat_adaptive_dijkstra_agent": {
        "author": "Naz (earlier hackathon work)",
        "note": "Algorithm-based (threat-adaptive Dijkstra-style) agent, not ML. Used as a structured opponent for data generation.",
    },
}


def agent_credit(agent_id: str) -> dict[str, str]:
    return AGENT_CREDITS.get(
        agent_id,
        {"author": "Unknown / unspecified origin", "note": "External gamelib agent discovered from external_agents/."},
    )


def available_agents() -> list[dict[str, str]]:
    agents = [
        {"id": "trained_imitation_agent", "label": "Trained imitation agent", "kind": "trained"},
        {"id": "hackathon_winner_agent", "label": "Old hackathon winner / expert", "kind": "winner"},
        {"id": "random_legal_agent", "label": "Random legal baseline", "kind": "random"},
    ]
    for spec in discover_gamelib_agents():
        agents.append({"id": spec.name, "label": spec.name, "kind": "external", "path": str(spec.path)})
    for agent in agents:
        credit = agent_credit(agent["id"])
        agent["author"] = credit["author"]
        agent["note"] = credit["note"]
    return agents


def make_agent(agent_id: str, seed: int | None = None):
    if agent_id == "trained_imitation_agent":
        agent = TrainedHexAgent.from_checkpoint(ROOT / "models" / "final" / "imitation_11x11_best.pt")
        return agent, agent.name, agent.agent_type
    if agent_id == "hackathon_winner_agent":
        agent = ExpertAgentWrapper()
        return agent, agent.agent_name, "winner_agent"
    if agent_id == "random_legal_agent":
        return RandomLegalAgent(seed=seed), "random_legal_agent", "baseline_random"
    specs = {spec.name: spec for spec in discover_gamelib_agents()}
    if agent_id not in specs:
        raise ValueError(f"Unknown agent id: {agent_id}")
    spec = specs[agent_id]
    return GamelibAgentWrapper(spec.path, agent_name=spec.name, agent_type="external_gamelib_agent"), spec.name, "external_gamelib_agent"


def choose(agent: Any, board, player: int):
    result = agent.choose_move(board, player)
    return result.move if hasattr(result, "move") else result


def record_to_dict(record: MoveLogRecord) -> dict[str, Any]:
    return dict(record.__dict__)


def public_state(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": game["game_id"],
        "board_size": game["board_size"],
        "board": board_to_list(game["board"]),
        "current_player": game["current_player"],
        "status": game["status"],
        "final_winner": game["final_winner"],
        "turn_number": len(game["records"]),
        "records": [record_to_dict(record) for record in game["records"]],
        "move_timings": game["move_timings"],
        "saved_log_path": game.get("saved_log_path"),
        "players": {
            "1": {"id": game["p1_id"], "name": game["p1_name"], "type": game["p1_type"]},
            "2": {"id": game["p2_id"], "name": game["p2_name"], "type": game["p2_type"]},
        },
    }


def start_game(payload: dict[str, Any]) -> dict[str, Any]:
    p1_id = payload.get("player1", "trained_imitation_agent")
    p2_id = payload.get("player2", "hackathon_winner_agent")
    board_size = int(payload.get("board_size", DEFAULT_BOARD_SIZE))
    game_id = "demo_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    p1, p1_name, p1_type = make_agent(p1_id, seed=1)
    p2, p2_name, p2_type = make_agent(p2_id, seed=2)
    GAMES[game_id] = {
        "game_id": game_id,
        "board_size": board_size,
        "board": create_empty_board(board_size),
        "current_player": PLAYER_1,
        "status": "in_progress",
        "final_winner": None,
        "records": [],
        "move_timings": [],
        "p1_agent": p1,
        "p2_agent": p2,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "p1_name": p1_name,
        "p2_name": p2_name,
        "p1_type": p1_type,
        "p2_type": p2_type,
    }
    return public_state(GAMES[game_id])


def save_live_log(game: dict[str, Any]) -> Path:
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_LOG_DIR / f"game_{game['game_id']}.json"
    save_game_log(path, game["records"])
    game["saved_log_path"] = str(path)
    timing_path = LIVE_LOG_DIR / f"timings_{game['game_id']}.json"
    timing_path.write_text(json.dumps(game["move_timings"], indent=2), encoding="utf-8")
    return path


def step_game(game_id: str) -> dict[str, Any]:
    if game_id not in GAMES:
        raise ValueError(f"Unknown game id: {game_id}")
    game = GAMES[game_id]
    if game["status"] != "in_progress":
        return public_state(game)

    player = game["current_player"]
    board_before = game["board"].copy()
    legal = get_legal_moves(board_before)
    if not legal:
        game["status"] = "draw"
        save_live_log(game)
        return public_state(game)

    if player == 1:
        agent, agent_name, agent_type = game["p1_agent"], game["p1_name"], game["p1_type"]
        opponent_name, opponent_type = game["p2_name"], game["p2_type"]
    else:
        agent, agent_name, agent_type = game["p2_agent"], game["p2_name"], game["p2_type"]
        opponent_name, opponent_type = game["p1_name"], game["p1_type"]

    started = time.perf_counter()
    move = choose(agent, board_before, player)
    elapsed = time.perf_counter() - started
    board_after = apply_move(board_before, move, player)
    status = game_status(board_after)
    final_winner = int(status.split("_")[1]) if status.startswith("winner_") else None
    record = MoveLogRecord(
        game_id=game_id,
        turn_number=len(game["records"]),
        board_size=game["board_size"],
        player_to_move=player,
        board_before=board_to_list(board_before),
        legal_moves_before=legal,
        selected_move=move,
        agent_name=agent_name,
        agent_type=agent_type,
        board_after=board_to_list(board_after),
        was_legal=True,
        game_status_after_move=status,
        final_winner=final_winner,
        opponent_name=opponent_name,
        opponent_type=opponent_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_outputs={"move_time_seconds": elapsed},
    )
    game["records"].append(record)
    game["move_timings"].append({
        "turn_number": record.turn_number,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "player": player,
        "move": list(move),
        "seconds": elapsed,
    })
    game["board"] = board_after
    game["status"] = status
    game["final_winner"] = final_winner
    if final_winner is None:
        game["current_player"] = other_player(player)
    else:
        save_live_log(game)
    return public_state(game)


def play_to_end(game_id: str) -> dict[str, Any]:
    while game_id in GAMES and GAMES[game_id]["status"] == "in_progress":
        step_game(game_id)
    return public_state(GAMES[game_id])


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send(204, b"")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/agents":
                self._json(200, {"agents": available_agents()})
                return
            if path.startswith("/api/state/"):
                game_id = path.rsplit("/", 1)[-1]
                self._json(200, public_state(GAMES[game_id]))
                return
            if path == "/":
                path = "/index.html"
            file_path = (DEMO_DIR / path.lstrip("/")).resolve()
            if DEMO_DIR.resolve() not in file_path.parents and file_path != DEMO_DIR.resolve():
                self._json(403, {"error": "forbidden"})
                return
            if not file_path.exists():
                self._json(404, {"error": "not found"})
                return
            content_type = "text/html" if file_path.suffix == ".html" else "text/javascript" if file_path.suffix == ".js" else "text/css" if file_path.suffix == ".css" else "text/plain"
            self._send(200, file_path.read_bytes(), content_type)
        except Exception as exc:
            self._json(500, {"error": repr(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                self._json(200, start_game(self._read_json()))
                return
            if path.startswith("/api/step/"):
                self._json(200, step_game(path.rsplit("/", 1)[-1]))
                return
            if path.startswith("/api/play-to-end/"):
                self._json(200, play_to_end(path.rsplit("/", 1)[-1]))
                return
            if path.startswith("/api/save/"):
                game_id = path.rsplit("/", 1)[-1]
                self._json(200, {"saved_log_path": str(save_live_log(GAMES[game_id]))})
                return
            self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": repr(exc)})


def main() -> None:
    print(f"Hex demo server running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
