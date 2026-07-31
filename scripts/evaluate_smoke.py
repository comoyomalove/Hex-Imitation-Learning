from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from hex_ml.agents.baselines import RandomLegalAgent
from hex_ml.agents.trained_agent import TrainedHexAgent
from hex_ml.evaluation.evaluator import evaluate_trained_agent


def latest_checkpoint() -> Path:
    candidates = sorted(Path("models/checkpoints").glob("*.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint files found in models/checkpoints")
    return candidates[-1]


def main() -> None:
    final = Path("models/final/imitation_agent_best.pt")
    checkpoint = final if final.exists() else latest_checkpoint()
    agent = TrainedHexAgent.from_checkpoint(checkpoint)
    opponent = RandomLegalAgent(seed=99)
    result = evaluate_trained_agent(agent, opponent, games=4, board_size=3)
    run_id = datetime.now().strftime("eval_smoke_%Y%m%d_%H%M%S")
    json_path = Path("logs/evaluation") / f"{run_id}.json"
    md_path = Path("reports/evaluation") / f"{run_id}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["checkpoint"] = str(checkpoint)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_lines = [
        "# Smoke Evaluation Report",
        "",
        f"- checkpoint: {checkpoint}",
        f"- games: {result.games}",
        f"- opponent: {result.opponent_name}",
        f"- wins: {result.trained_agent_wins}",
        f"- losses: {result.opponent_wins}",
        f"- draws: {result.draws}",
        f"- win_rate: {result.win_rate:.2f}",
        f"- illegal_moves: {result.illegal_moves}",
        f"- crashes: {result.crashes}",
        "",
        "Interpretation: this is a wiring smoke test, not a strength claim.",
        "",
    ]
    md_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"json={json_path}")
    print(f"report={md_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
