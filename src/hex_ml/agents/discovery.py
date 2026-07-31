"""Auto-discovery of external agents for opponents and the demo dropdown.

Scans external_agents/other_gamelib_agents/ and returns one entry per agent
folder that contains an agent.py or final_agent.py file. Any agent dropped
into that folder becomes usable by the dataset-generation scripts and appears
automatically in the demo's live-mode player dropdowns, with no code changes.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExternalAgentSpec:
    name: str
    path: Path
    agent_type: str
    strategy_label: str

def discover_gamelib_agents(base_dir: Path | str = Path('external_agents/other_gamelib_agents')) -> list[ExternalAgentSpec]:
    base = Path(base_dir)
    specs = []
    if not base.exists():
        return specs
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        for filename in ('final_agent.py', 'agent.py'):
            path = folder / filename
            if path.exists():
                specs.append(ExternalAgentSpec(name=folder.name, path=path, agent_type='structured_hackathon_agent', strategy_label=folder.name))
                break
    return specs
