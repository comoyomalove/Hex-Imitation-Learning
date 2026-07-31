from hex_ml.agents.discovery import discover_gamelib_agents
from hex_ml.agents.gamelib_wrapper import GamelibAgentWrapper
from hex_ml.core.board import PLAYER_1, create_empty_board, is_legal_move

def test_discover_external_gamelib_agents_finds_dropped_agents():
    specs = discover_gamelib_agents()
    names = {spec.name for spec in specs}
    assert {'hybrid_phase_agent', 'mcts_lite_hex_agent', 'minimax_lite_hex_agent', 'smart_hex_agent', 'threat_adaptive_dijkstra_agent'}.issubset(names)
    assert all(spec.path.name in {'agent.py', 'final_agent.py'} for spec in specs)

def test_generic_gamelib_wrapper_loads_winner_agent():
    wrapper = GamelibAgentWrapper('Hackaton-Winner-Agent/final_agent.py', agent_name='winner', agent_type='winner_agent')
    result = wrapper.choose_move(create_empty_board(3), PLAYER_1)
    assert wrapper.available is True
    assert result.agent_type == 'winner_agent'
    assert is_legal_move(create_empty_board(3), result.move)
