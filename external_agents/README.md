# External Agents

This folder is for Hex agents that come from outside the main Hex-ML training code.

These agents are used as opponents for the winner/expert agent during data generation and optional evaluation opponents.

They should not be called by the trained imitation agent at inference time.

## Recommended layout

external_agents/
  other_gamelib_agents/
    agent_name_1/
      final_agent.py
    agent_name_2/
      final_agent.py

Each agent should live in its own subfolder.

## Expected agent interface

These agents are expected to use the same hackathon / aica-gamelib style as the winner agent.

Expected class and methods:

class HexAgent:
    def initialize(self, init_data: dict) -> None:
        ...

    def get_move(self, state):
        ...

The generic wrapper will load the agent file, instantiate HexAgent, convert the Hex-ML board into gamelib format, call get_move, and convert the move back into Hex-ML row/col format.

## Important rule

External agents are for data generation and evaluation only.

The trained imitation model must make moves independently and must not call these agents at inference time.
