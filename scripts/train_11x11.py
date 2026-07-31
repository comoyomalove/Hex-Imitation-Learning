"""Train an 11x11 Hex imitation-learning policy model.

This script is the main supervised-learning training script for the project.
It trains a neural network to imitate expert moves from saved JSONL records.

Important idea:
    The expert/winner agent is NOT called during training.
    The model only learns from examples that were already generated and saved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hex_ml.training.dataset import ImitationJsonlDataset
from hex_ml.training.model import TinyHexPolicyNet


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------
# PyTorch can train on different devices:
# - cpu: always available, but slower
# - mps: Apple Silicon GPU backend, useful on your Mac
# - cuda: NVIDIA GPU backend, useful on some other machines
# - auto: choose the best available option
#
# This only affects neural-network training. It does not affect the search-based
# winner agent, because the winner agent is not used inside this training script.
# -----------------------------------------------------------------------------
def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Leakage-free split handling
# -----------------------------------------------------------------------------
# The split manifest contains game ids for train, validation, and test.
# We split by game id rather than by individual move/example.
#
# Why this matters:
# Moves from the same game are highly related. If one game appears in both train
# and validation, the validation score can become too optimistic. Splitting by
# game id prevents that leakage.
# -----------------------------------------------------------------------------
def split_indices(dataset: ImitationJsonlDataset, split_manifest_path: Path):
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    split_by_game = {
        game_id: split_name
        for split_name, game_ids in manifest["splits"].items()
        for game_id in game_ids
    }

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for index, record in enumerate(dataset.records):
        split = split_by_game.get(record["game_id"])

        if split == "train":
            train_indices.append(index)
        elif split == "validation":
            validation_indices.append(index)
        elif split == "test":
            test_indices.append(index)

    return train_indices, validation_indices, test_indices


# -----------------------------------------------------------------------------
# One training epoch
# -----------------------------------------------------------------------------
# An epoch means the model sees the full training set once, in mini-batches.
#
# For each batch:
# 1. move inputs/targets to the selected device,
# 2. compute model predictions,
# 3. compare predictions with expert moves using cross-entropy loss,
# 4. backpropagate the error,
# 5. update the neural-network weights.
# -----------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # The model outputs one score for each board cell.
        # On an 11x11 board this means 121 move scores.
        logits = model(inputs)

        # CrossEntropyLoss treats the expert move as the correct class.
        # Lower loss means the model is assigning more probability to the expert move.
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


# -----------------------------------------------------------------------------
# Validation / evaluation during training
# -----------------------------------------------------------------------------
# Validation does not update the model weights.
# It answers: how well does the model imitate expert moves on games it did not
# train on?
#
# Metrics:
# - loss: lower is better
# - top-1 accuracy: model's highest-scoring move is the expert move
# - top-5 accuracy: expert move is one of the model's five best guesses
# -----------------------------------------------------------------------------
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            top1_correct += int((torch.argmax(logits, dim=1) == targets).sum().item())

            topk_indices = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            top5_correct += int((topk_indices == targets.unsqueeze(1)).any(dim=1).sum().item())

    denominator = max(total_examples, 1)

    return (
        total_loss / denominator,
        top1_correct / denominator,
        top5_correct / denominator,
    )


# -----------------------------------------------------------------------------
# Main training run
# -----------------------------------------------------------------------------
# This function does the full training workflow:
# 1. read command-line options,
# 2. load saved imitation examples,
# 3. load the leakage-free game-id split,
# 4. build train and validation DataLoaders,
# 5. create the neural network,
# 6. train for several epochs,
# 7. validate after each epoch,
# 8. save checkpoints and a Markdown report.
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Dataset and split inputs.
    parser.add_argument("--records-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)

    # Hardware/device choice.
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    # Naming prefix for logs/reports/checkpoints.
    parser.add_argument("--run-prefix", default="imitation_11x11")

    args = parser.parse_args()

    seed = 123
    torch.manual_seed(seed)

    device = choose_device(args.device)

    # Load all imitation examples from JSONL.
    # Each line is one expert example extracted from a game log.
    dataset = ImitationJsonlDataset(args.records_path)

    train_idx, val_idx, test_idx = split_indices(dataset, args.split_path)

    if not train_idx or not val_idx:
        raise ValueError("Need non-empty train and validation splits.")

    board_size = int(dataset.records[0]["board_size"])

    # Create the policy neural network.
    # Input: encoded board channels.
    # Output: one score per board cell.
    model = TinyHexPolicyNet(
        board_size=board_size,
        input_channels=4,
        hidden_size=args.hidden_size,
    ).to(device)

    # Adam updates the model weights to reduce prediction loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # The 121 board cells are treated as 121 possible move classes.
    loss_fn = nn.CrossEntropyLoss()

    # DataLoaders turn the dataset into mini-batches.
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    run_id = datetime.now().strftime(f"{args.run_prefix}_%Y%m%d_%H%M%S")

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_dir = Path("models/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs/training") / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path("reports/training") / f"{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    best_path = checkpoint_dir / f"{run_id}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_loss, validation_top1, validation_top5 = evaluate(model, val_loader, loss_fn, device)

        row = {
            "run_id": run_id,
            "epoch": epoch,
            "device": str(device),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_top1_accuracy": validation_top1,
            "validation_top5_accuracy": validation_top5,
            "train_examples": len(train_idx),
            "validation_examples": len(val_idx),
            "test_examples": len(test_idx),
            "board_size": board_size,
            "records_path": str(args.records_path),
            "split_path": str(args.split_path),
        }

        history.append(row)

        # Machine-readable per-epoch training log.
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\\n")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": {
                "board_size": board_size,
                "input_channels": 4,
                "hidden_size": args.hidden_size,
            },
            "metrics": row,
            "random_seed": seed,
            "dataset_path": str(args.records_path),
            "split_manifest": str(args.split_path),
            "run_id": run_id,
        }

        epoch_path = checkpoint_dir / f"{run_id}_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)

        # Save the best checkpoint by validation loss, not simply the last epoch.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(checkpoint, best_path)

        print(json.dumps(row, indent=2))

    final_path = final_dir / "imitation_11x11_best.pt"
    final_path.write_bytes(best_path.read_bytes())

    best = min(history, key=lambda r: r["validation_loss"])

    report_lines = [
        "# 11x11 Imitation Training Report",
        "",
        f"- run_id: {run_id}",
        f"- device: {device}",
        f"- board_size: {board_size}",
        f"- train_examples: {len(train_idx)}",
        f"- validation_examples: {len(val_idx)}",
        f"- test_examples: {len(test_idx)}",
        f"- epochs: {args.epochs}",
        f"- best_epoch: {best['epoch']}",
        f"- best_validation_loss: {best['validation_loss']:.4f}",
        f"- best_validation_top1_accuracy: {best['validation_top1_accuracy']:.4f}",
        f"- best_validation_top5_accuracy: {best['validation_top5_accuracy']:.4f}",
        f"- best_checkpoint: {best_path}",
        f"- final_model_copy: {final_path}",
        "",
        "Interpretation: this trains on saved 11x11 expert demonstrations.",
        "The trained model learns to predict expert moves from board states.",
        "It does not call the expert agent during inference.",
        "",
    ]
    report = "\n".join(report_lines)

    report_path.write_text(report, encoding="utf-8")

    print(f"best_checkpoint={best_path}")
    print(f"final_model={final_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
