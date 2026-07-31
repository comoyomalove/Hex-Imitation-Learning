"""Build a balanced 11x11 structured-opponent dataset from completed game logs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from hex_ml.data.dataset_builder import imitation_records_from_game_log, save_imitation_records_jsonl
from hex_ml.data.game_log import replay_game_log
from hex_ml.data.splitting import save_split_manifest, split_records_by_game_id

SOURCES = [
    Path("logs/games/structured_opponents_11x11_500_per_opponent_w8"),
    Path("logs/games/structured_opponents_11x11_smart_500_w4"),
    Path("logs/games/structured_opponents_11x11_threat_adaptive_500_w8"),
    Path("logs/games/structured_opponents_11x11_minimax_500_w8"),
]
TARGET_BY_AGENT = {
    "hybrid_phase_agent": 500,
    "mcts_lite_hex_agent": 500,
    "minimax_lite_hex_agent": 500,
    "smart_hex_agent": 500,
    "threat_adaptive_dijkstra_agent": 500,
}


def opponent_from_log(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload.get("records", []):
        if record.get("agent_type") != "winner_agent":
            return str(record["agent_name"])
    return "unknown"


def main() -> None:
    selected = []
    counts = Counter()
    examples_by_agent = defaultdict(int)
    scanned = 0

    for source_dir in SOURCES:
        for path in sorted(source_dir.glob("game_*.json")):
            scanned += 1
            opponent = opponent_from_log(path)
            if opponent not in TARGET_BY_AGENT:
                continue
            if counts[opponent] >= TARGET_BY_AGENT[opponent]:
                continue
            replay_game_log(path)
            selected.append(path)
            counts[opponent] += 1

    missing = {
        agent: TARGET_BY_AGENT[agent] - counts[agent]
        for agent in TARGET_BY_AGENT
        if counts[agent] < TARGET_BY_AGENT[agent]
    }
    if missing:
        raise RuntimeError(f"Missing requested games: {missing}")

    records = []
    for path in selected:
        opponent = opponent_from_log(path)
        recs = imitation_records_from_game_log(path)
        examples_by_agent[opponent] += len(recs)
        records.extend(recs)

    split_records, split_map = split_records_by_game_id(records, seed=42)
    records_path = Path("data/processed/structured_11x11_balanced_2500_imitation_records.jsonl")
    split_path = Path("data/splits/structured_11x11_balanced_2500_split_manifest.json")
    summary_path = Path("logs/development/structured_11x11_balanced_2500_summary.json")
    report_path = Path("reports/data/structured_11x11_balanced_2500_summary.md")

    save_imitation_records_jsonl(records_path, split_records)
    save_split_manifest(split_path, split_map, seed=42)

    summary = {
        "selected_games": len(selected),
        "scanned_logs": scanned,
        "games_by_opponent": dict(sorted(counts.items())),
        "imitation_records": len(records),
        "examples_by_opponent": dict(sorted(examples_by_agent.items())),
        "records_path": str(records_path),
        "split_path": str(split_path),
        "split_counts_games": {k: len(v) for k, v in split_map.items()},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Balanced 11x11 Structured Dataset Summary",
        "",
        f"- selected games: {len(selected)}",
        f"- imitation records: {len(records)}",
        "",
        "## Games by opponent",
        "",
        "| Opponent | Games | Expert examples |",
        "|---|---:|---:|",
    ]
    for agent in sorted(counts):
        lines.append(f"| {agent} | {counts[agent]} | {examples_by_agent[agent]} |")
    lines += ["", "## Split counts by game", "", "```json", json.dumps(summary["split_counts_games"], indent=2), "```"]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
