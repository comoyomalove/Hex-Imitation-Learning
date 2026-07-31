from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, median


def summarize(prefix: str):
    rows = []
    all_turns = []
    total_wins = 0
    total_games = 0
    for b in ["001", "002", "003", "004", "005", "006"]:
        folder = Path(f"logs/games/{prefix}_batch{b}_w8")
        sp = Path(f"logs/evaluation/{prefix}_batch{b}_w8_summary.json")
        logs = sorted(folder.glob("game_*.json")) if folder.exists() else []
        turns = []
        wins = 0
        for p in logs:
            data = json.loads(p.read_text(encoding="utf-8"))
            recs = data.get("records", [])
            turns.append(len(recs))
            if recs:
                trained_players = {r["player_to_move"] for r in recs if r.get("agent_type") == "trained_model"}
                final_winner = recs[-1].get("final_winner")
                if final_winner in trained_players:
                    wins += 1
        item = {
            "batch": b,
            "folder_exists": folder.exists(),
            "games": len(logs),
            "summary_exists": sp.exists(),
            "trained_local_wins_from_logs": wins,
        }
        if turns:
            item.update({
                "avg_turns": mean(turns),
                "median_turns": median(turns),
                "min_turns": min(turns),
                "max_turns": max(turns),
                "count_ge_30": sum(t >= 30 for t in turns),
                "count_ge_40": sum(t >= 40 for t in turns),
                "count_ge_50": sum(t >= 50 for t in turns),
            })
        if sp.exists():
            summary = json.loads(sp.read_text(encoding="utf-8"))
            for key in [
                "games_completed", "failures", "total_elapsed_seconds",
                "average_wall_seconds_per_game", "trained_local_wins",
                "trained_policy", "top_k", "temperature", "opening_sample_moves",
            ]:
                if key in summary:
                    item[key] = summary[key]
        rows.append(item)
        all_turns.extend(turns)
        total_wins += wins
        total_games += len(logs)
    combined = {"prefix": prefix, "games": total_games, "trained_local_wins": total_wins}
    if all_turns:
        combined.update({
            "avg_turns": mean(all_turns),
            "median_turns": median(all_turns),
            "min_turns": min(all_turns),
            "max_turns": max(all_turns),
            "count_ge_30": sum(t >= 30 for t in all_turns),
            "count_ge_40": sum(t >= 40 for t in all_turns),
            "count_ge_50": sum(t >= 50 for t in all_turns),
        })
    return {"combined": combined, "batches": rows}


def fmt(value, digits=2):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    smart = summarize("trained_vs_winner_11x11_smartopen")
    old = summarize("trained_vs_winner_11x11_expandedmodel")
    out = {"smartopen_latest": smart, "old_deterministic_partial": old}
    out_dir = Path("reports/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trained_vs_winner_game_length_comparison.json"
    md_path = out_dir / "trained_vs_winner_game_length_comparison.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = []
    lines.append("# Trained-vs-Winner Game Length Comparison")
    lines.append("")
    lines.append("## Smart-open latest batches")
    lines.append("")
    lines.append("| Batch | Games | Avg turns | Median | Min | Max | >=30 | >=40 | >=50 | Trained local wins |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in smart["batches"]:
        lines.append(
            "| " + r["batch"]
            + " | " + str(r.get("games", 0))
            + " | " + fmt(r.get("avg_turns", 0))
            + " | " + fmt(r.get("median_turns", 0), 1)
            + " | " + str(r.get("min_turns", 0))
            + " | " + str(r.get("max_turns", 0))
            + " | " + str(r.get("count_ge_30", 0))
            + " | " + str(r.get("count_ge_40", 0))
            + " | " + str(r.get("count_ge_50", 0))
            + " | " + str(r.get("trained_local_wins_from_logs", 0))
            + " |"
        )
    c = smart["combined"]
    lines.append("")
    lines.append("## Combined smart-open")
    lines.append("")
    lines.append("- games: " + str(c.get("games", 0)))
    lines.append("- average turns: " + fmt(c.get("avg_turns", 0)))
    lines.append("- median turns: " + fmt(c.get("median_turns", 0), 1))
    lines.append("- min/max turns: " + str(c.get("min_turns", 0)) + " / " + str(c.get("max_turns", 0)))
    lines.append("- games with at least 30 turns: " + str(c.get("count_ge_30", 0)))
    lines.append("- games with at least 40 turns: " + str(c.get("count_ge_40", 0)))
    lines.append("- games with at least 50 turns: " + str(c.get("count_ge_50", 0)))
    lines.append("- trained local wins: " + str(c.get("trained_local_wins", 0)))
    lines.append("")
    lines.append("## Old deterministic partial comparison")
    lines.append("")
    oc = old["combined"]
    lines.append("- games: " + str(oc.get("games", 0)))
    lines.append("- average turns: " + fmt(oc.get("avg_turns", 0)))
    lines.append("- median turns: " + fmt(oc.get("median_turns", 0), 1))
    lines.append("- min/max turns: " + str(oc.get("min_turns", 0)) + " / " + str(oc.get("max_turns", 0)))
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append("Earlier smart-open diversity analysis averaged about 23 moves/game. This report recomputes the final completed batches from logs using corrected winner orientation.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"smartopen_combined": smart["combined"], "old_partial_combined": old["combined"], "report_md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
