from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from hex_ml.data.game_log import load_game_log
from hex_ml.data.schemas import ImitationRecord


def imitation_records_from_game_log(
    path: Path, expert_agent_type: str = "winner_agent"
) -> list[ImitationRecord]:
    payload = load_game_log(path)
    out = []
    for r in payload["records"]:
        if r["agent_type"] != expert_agent_type:
            continue
        out.append(
            ImitationRecord(
                game_id=r["game_id"],
                turn_number=r["turn_number"],
                board_size=r["board_size"],
                player_to_move=r["player_to_move"],
                board_before=r["board_before"],
                legal_moves=[tuple(m) for m in r["legal_moves_before"]],
                expert_move=tuple(r["selected_move"]),
                expert_agent_name=r["agent_name"],
                opponent_name=r["opponent_name"],
                source_game_log=str(path),
            )
        )
    return out


def save_imitation_records_jsonl(path: Path, records: list[ImitationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            d: dict[str, Any] = asdict(rec)
            d["legal_moves"] = [list(m) for m in rec.legal_moves]
            d["expert_move"] = list(rec.expert_move)
            f.write(json.dumps(d) + "\n")


def load_imitation_records_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
