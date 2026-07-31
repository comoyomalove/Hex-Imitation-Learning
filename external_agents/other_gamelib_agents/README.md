# Other Gamelib Agents

Drop older hackathon-compatible agents here.

Use one subfolder per agent:

external_agents/other_gamelib_agents/my_agent_name/final_agent.py

Expected file/class:

final_agent.py
class HexAgent

Expected methods:

initialize(init_data: dict)
get_move(state)

These agents do not need to be arena winners. They are useful because they create structured, non-random behavior for the winner agent to respond to during data generation.
