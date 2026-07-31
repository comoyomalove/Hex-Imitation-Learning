"""Build DAgger round 2 dataset from smart-open trained-vs-winner logs.

Extracts winner-agent moves only. These are expert correction examples on states
influenced by the trained imitation agent.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

from hex_ml.data.dataset_builder import imitation_records_from_game_log
from hex_ml.data.splitting import save_split_manifest, split_records_by_game_id

LOG_DIRS = [
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch001_w8"),
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch002_w8"),
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch003_w8"),
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch004_w8"),
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch005_w8"),
    Path("logs/games/trained_vs_winner_11x11_smartopen_batch006_w8"),
]

RECORDS_PATH = Path("data/processed/dagger_round2_smartopen_3000_records.jsonl")
SPLIT_PATH = Path("data/splits/dagger_round2_smartopen_3000_split_manifest.json")
SUMMARY_JSON = Path("reports/data/dagger_round2_smartopen_3000_summary.json")
SUMMARY_MD = Path("reports/data/dagger_round2_smartopen_3000_summary.md")


def to_dict(record):
    if is_dataclass(record):
        return asdict(record)
    return dict(record)


def phase_from_turn(turn_number: int, board_size: int) -> str:
    frac = turn_number / float(board_size * board_size)
    if frac < 1 / 3:
        return "opening"
    if frac < 2 / 3:
        return "midgame"
    return "endgame"


def main() -> None:
    records = []
    phase_counts = Counter()
    games = 0

    missing_dirs = [str(path) for path in LOG_DIRS if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing log dirs: {missing_dirs}")

    for log_dir in LOG_DIRS:
        for path in sorted(log_dir.glob("game_*.json")):
            games += 1
            for record in imitation_records_from_game_log(path):
                item = to_dict(record)
                board_size = len(item["board_before"])
                phase = phase_from_turn(int(item["turn_number"]), board_size)
                item["source_dataset"] = "dagger_round2_smartopen_3000"
                item["source_type"] = "trained_vs_winner_smartopen"
                item["phase"] = phase
                item["model_version"] = "imitation_11x11_expanded5000"
                item["source_game_log"] = str(path)
                records.append(item)
                phase_counts[phase] += 1

    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECORDS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    _split_records, split_map = split_records_by_game_id(records, seed=42)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_split_manifest(SPLIT_PATH, split_map, seed=42)

    summary = {
        "source": "smart-open trained-vs-winner batches 001-006",
        "games": games,
        "records": len(records),
        "phase_counts": dict(phase_counts),
        "records_path": str(RECORDS_PATH),
        "split_path": str(SPLIT_PATH),
        "split_game_counts": {key: len(value) for key, value in split_map.items()},
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# DAgger Round 2 Smart-Open Dataset Summary",
        "",
        "- source: " + summary["source"],
        "- games: " + str(games),
        "- winner-agent correction records: " + str(len(records)),
        "- records path: `" + str(RECORDS_PATH) + "`",
        "- split path: `" + str(SPLIT_PATH) + "`",
        "",
        "## Phase counts",
        "",
        "| Phase | Records |",
        "|---|---:|",
    ]
    for phase, count in sorted(phase_counts.items()):
        lines.append("| " + phase + " | " + str(count) + " |")

    lines += [
        "",
        "## Split game counts",
        "",
        "```json",
        json.dumps(summary["split_game_counts"], indent=2),
        "```",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

