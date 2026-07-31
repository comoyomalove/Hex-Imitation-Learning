"""Analyze opening/full trajectory diversity in game logs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def signature(records: list[dict], n: int) -> tuple:
    return tuple((r["agent_name"], tuple(r["selected_move"])) for r in records[:n])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--opening-moves", type=int, default=10)
    args = parser.parse_args()

    files = sorted(args.log_dir.glob("game_*.json"))
    opening = Counter()
    full = Counter()
    lengths: list[int] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        lengths.append(len(records))
        opening[signature(records, args.opening_moves)] += 1
        full[signature(records, len(records))] += 1

    summary = {
        "log_dir": str(args.log_dir),
        "games": len(files),
        "opening_moves": args.opening_moves,
        "unique_opening_signatures": len(opening),
        "unique_full_trajectories": len(full),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "most_common_opening_counts": opening.most_common(10),
        "most_common_full_counts": full.most_common(10),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
