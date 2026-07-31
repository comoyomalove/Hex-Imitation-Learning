from __future__ import annotations
import json, time
from pathlib import Path
from hex_ml.agents.discovery import discover_gamelib_agents
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord

def play_game(spec, expert_player, out_dir):
    expert = ExpertAgentWrapper(); opponent = GamelibAgentWrapper(spec.path, spec.name, spec.agent_type)
    board = create_empty_board(DEFAULT_BOARD_SIZE); player = PLAYER_1; records = []
    start = time.perf_counter(); game_id = f"pilot11_{spec.name}_expert{expert_player}"
    for turn in range(DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE):
        before = board.copy(); legal = get_legal_moves(before)
        if not legal: break
        if player == expert_player:
            result = expert.choose_move(before, player); move = result.move; agent_name = result.expert_name; agent_type = result.source; opponent_name = opponent.agent_name; opponent_type = opponent.agent_type
        else:
            result = opponent.choose_move(before, player); move = result.move; agent_name = result.agent_name; agent_type = result.agent_type; opponent_name = expert.agent_name; opponent_type = "winner_agent"
        after = apply_move(before, move, player); status = game_status(after)
        winner = int(status.split("_")[1]) if status.startswith("winner_") else None
        records.append(MoveLogRecord(game_id, turn, DEFAULT_BOARD_SIZE, player, board_to_list(before), legal, move, agent_name, agent_type, board_to_list(after), True, status, winner, opponent_name, opponent_type, "pilot"))
        board = after
        if winner is not None: break
        player = other_player(player)
    elapsed = time.perf_counter() - start; path = out_dir / f"game_{game_id}.json"
    save_game_log(path, records); replay_game_log(path)
    return {"game_id": game_id, "opponent": spec.name, "expert_player": expert_player, "turns": len(records), "elapsed_seconds": elapsed, "winner": records[-1].final_winner if records else None, "log_path": str(path)}

def main():
    out_dir = Path("logs/games/pilot_11x11"); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; failures=[]; specs=discover_gamelib_agents()
    for spec in specs:
        for expert_player in (PLAYER_1, PLAYER_2):
            try: rows.append(play_game(spec, expert_player, out_dir))
            except Exception as exc: failures.append({"opponent": spec.name, "expert_player": expert_player, "error": repr(exc)})
    summary={"board_size": DEFAULT_BOARD_SIZE, "games_attempted": len(specs)*2, "games_completed": len(rows), "failures": failures, "games": rows, "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows)}
    p=Path("logs/development/pilot_11x11_summary.json"); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if failures: raise SystemExit(1)
if __name__ == "__main__": main()
