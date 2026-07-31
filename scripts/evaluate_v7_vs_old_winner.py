from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from hex_ml.agents.expert_wrapper import ExpertAgentWrapper
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.core.board import PLAYER_1, PLAYER_2, create_empty_board, get_legal_moves
from hex_ml.core.constants import DEFAULT_BOARD_SIZE
from hex_ml.core.game import apply_move, game_status, other_player
from hex_ml.data.game_log import board_to_list, replay_game_log, save_game_log
from hex_ml.data.schemas import MoveLogRecord

V7_PATH = Path('external_agents/other_gamelib_agents/v7_resistance_agent/agent.py')

def choose(agent: Any, board, player: int):
    result = agent.choose_move(board, player)
    return result.move if hasattr(result, 'move') else result

def make_agent(kind: str):
    if kind == 'old_winner':
        a = ExpertAgentWrapper()
        return a, a.agent_name, 'old_winner_agent'
    if kind == 'v7':
        a = GamelibAgentWrapper(V7_PATH, agent_name='v7_resistance_agent', agent_type='candidate_expert')
        return a, 'v7_resistance_agent', 'candidate_expert'
    raise ValueError(kind)

def play(job):
    t0 = time.perf_counter()
    board = create_empty_board(job['board_size'])
    p1, p1_name, p1_type = make_agent(job['p1_kind'])
    p2, p2_name, p2_type = make_agent(job['p2_kind'])
    player = PLAYER_1
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for turn in range(job['board_size'] * job['board_size']):
        before = board.copy()
        legal = get_legal_moves(before)
        if player == PLAYER_1:
            agent, name, typ, opp_name, opp_type = p1, p1_name, p1_type, p2_name, p2_type
        else:
            agent, name, typ, opp_name, opp_type = p2, p2_name, p2_type, p1_name, p1_type
        move = choose(agent, before, player)
        after = apply_move(before, move, player)
        status = game_status(after)
        winner = int(status.split('_')[1]) if status.startswith('winner_') else None
        records.append(MoveLogRecord(
            game_id=job['game_id'], turn_number=turn, board_size=job['board_size'], player_to_move=player,
            board_before=board_to_list(before), legal_moves_before=legal, selected_move=move,
            agent_name=name, agent_type=typ, board_after=board_to_list(after), was_legal=True,
            game_status_after_move=status, final_winner=winner, opponent_name=opp_name, opponent_type=opp_type,
            timestamp=timestamp))
        board = after
        if winner is not None:
            break
        player = other_player(player)
    path = Path(job['output_dir']) / f"game_{job['game_id']}.json"
    save_game_log(path, records)
    replay_game_log(path)
    final = records[-1].final_winner if records else None
    winner_name = p1_name if final == PLAYER_1 else p2_name if final == PLAYER_2 else None
    return {'game_id': job['game_id'], 'path': str(path), 'p1': p1_name, 'p2': p2_name, 'turns': len(records), 'final_winner_player': final, 'winner_name': winner_name, 'elapsed_seconds': time.perf_counter()-t0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games-per-role', type=int, default=5)
    ap.add_argument('--board-size', type=int, default=DEFAULT_BOARD_SIZE)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--output-dir', type=Path, default=Path('logs/games/v7_vs_old_winner_test'))
    ap.add_argument('--summary-path', type=Path, default=Path('logs/evaluation/v7_vs_old_winner_test_summary.json'))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    jobs=[]
    for i in range(args.games_per_role):
        jobs.append({'game_id': f'v7_p1_old_p2_{i:03d}', 'p1_kind':'v7', 'p2_kind':'old_winner', 'board_size':args.board_size, 'output_dir':str(args.output_dir)})
        jobs.append({'game_id': f'old_p1_v7_p2_{i:03d}', 'p1_kind':'old_winner', 'p2_kind':'v7', 'board_size':args.board_size, 'output_dir':str(args.output_dir)})
    start=time.perf_counter(); results=[]; failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(play,j):j for j in jobs}
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception as e: failures.append({'game_id': futs[fut]['game_id'], 'error': repr(e)})
    v7_wins=sum(1 for r in results if r['winner_name']=='v7_resistance_agent')
    old_wins=sum(1 for r in results if r['winner_name']=='hackathon_winner_agent')
    summary={'games_requested':len(jobs),'games_completed':len(results),'failures':failures,'v7_wins':v7_wins,'old_winner_wins':old_wins,'total_elapsed_seconds':time.perf_counter()-start,'results':sorted(results,key=lambda x:x['game_id'])}
    args.summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__ == '__main__': main()
