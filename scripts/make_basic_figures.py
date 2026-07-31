from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def latest_file(pattern: str) -> Path:
    files = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def plot_training_curves(log_path: Path, output_path: Path) -> None:
    df = read_jsonl(log_path)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train loss")
    if "validation_loss" in df:
        plt.plot(
            df["epoch"], df["validation_loss"], marker="o", label="validation loss"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Controlled training loss")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_validation_accuracy(log_path: Path, output_path: Path) -> None:
    df = read_jsonl(log_path)
    if "validation_accuracy" not in df:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["validation_accuracy"], marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.ylim(0, 1)
    plt.title("Controlled training validation accuracy")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_eval_win_rate(eval_path: Path, output_path: Path) -> None:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    opponent = data.get("opponent_name", "opponent")
    win_rate = float(data.get("win_rate", 0.0))
    plt.figure(figsize=(5, 4))
    plt.bar([opponent], [win_rate], color="purple")
    plt.ylim(0, 1)
    plt.ylabel("Win rate")
    plt.title("Smoke evaluation win rate")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    training_log = latest_file("logs/training/controlled_*.jsonl")
    eval_log = latest_file("logs/evaluation/*.json")
    plot_training_curves(training_log, Path("reports/figures/training_loss_curve.png"))
    plot_validation_accuracy(
        training_log, Path("reports/figures/validation_accuracy_curve.png")
    )
    plot_eval_win_rate(eval_log, Path("reports/figures/evaluation_win_rate.png"))
    print(f"training_log={training_log}")
    print(f"eval_log={eval_log}")
    print("figures saved")


if __name__ == "__main__":
    main()
