from __future__ import annotations
from pathlib import Path
from hex_ml.agents.gamelib_wrapper import GamelibAgentMoveResult as ExpertMoveResult, GamelibAgentWrapper

class ExpertAgentWrapper(GamelibAgentWrapper):
    def __init__(self, agent_path: Path | str = Path('Hackaton-Winner-Agent/final_agent.py')) -> None:
        super().__init__(agent_path=agent_path, agent_name='hackathon_winner_agent', agent_type='winner_agent')
