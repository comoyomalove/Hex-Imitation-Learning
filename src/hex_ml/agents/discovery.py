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
