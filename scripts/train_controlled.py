from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from hex_ml.training.dataset import ImitationJsonlDataset
from hex_ml.training.model import TinyHexPolicyNet


def main() -> None:
    dataset_path = Path("data/processed/tiny_imitation_records.jsonl")
    split_path = Path("data/splits/tiny_split_manifest.json")
    seed = 123
    torch.manual_seed(seed)
    dataset = ImitationJsonlDataset(dataset_path)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_by_game = {
        game_id: split_name
        for split_name, game_ids in split_manifest["splits"].items()
        for game_id in game_ids
    }
    train_indices = [
        i
        for i, record in enumerate(dataset.records)
        if split_by_game[record["game_id"]] == "train"
    ]
    val_indices = [
        i
        for i, record in enumerate(dataset.records)
        if split_by_game[record["game_id"]] == "validation"
    ]
    if not train_indices or not val_indices:
        raise ValueError(
            "Controlled training requires non-empty train and validation splits."
        )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)
    board_size = int(dataset.records[0]["board_size"])
    model = TinyHexPolicyNet(board_size=board_size, input_channels=4, hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    run_id = datetime.now().strftime("controlled_%Y%m%d_%H%M%S")
    log_path = Path("logs/training") / f"{run_id}.jsonl"
    report_path = Path("reports/training") / f"{run_id}.md"
    checkpoint_dir = Path("models/checkpoints")
    final_dir = Path("models/final")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / f"imitation_{run_id}_best.pt"
    history = []
    for epoch in range(1, 6):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)
        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "train_examples": len(train_indices),
            "validation_examples": len(val_indices),
            "dataset_path": str(dataset_path),
            "split_manifest": str(split_path),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": {
                "board_size": board_size,
                "input_channels": 4,
                "hidden_size": 64,
            },
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "random_seed": seed,
            "dataset_path": str(dataset_path),
            "split_manifest": str(split_path),
            "run_id": run_id,
        }
        torch.save(
            checkpoint, checkpoint_dir / f"imitation_{run_id}_epoch_{epoch:03d}.pt"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_checkpoint_path)
    final_path = final_dir / "imitation_agent_best.pt"
    final_path.write_bytes(best_checkpoint_path.read_bytes())
    best = min(history, key=lambda item: item["validation_loss"])
    report = [
        "# Controlled Training Report",
        "",
        f"- run_id: {run_id}",
        f"- dataset: {dataset_path}",
        f"- split_manifest: {split_path}",
        f"- train_examples: {len(train_indices)}",
        f"- validation_examples: {len(val_indices)}",
        f"- epochs: {len(history)}",
        f'- best_epoch: {best["epoch"]}',
        f'- best_validation_loss: {best["validation_loss"]:.4f}',
        f'- best_validation_accuracy: {best["validation_accuracy"]:.4f}',
        f"- best_checkpoint: {best_checkpoint_path}",
        f"- final_model_copy: {final_path}",
        "",
        "## Interpretation",
        "This is a controlled tiny training run with a leakage-free game-id split. It checks validation logging and checkpoint selection, but it is still too small to support strong performance claims.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"best_checkpoint={best_checkpoint_path}")
    print(f"final_model={final_path}")
    print(f"report={report_path}")
    print(json.dumps(best, indent=2))


def run_epoch(model, loader, loss_fn, optimizer) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(inputs.shape[0])
        total_examples += int(inputs.shape[0])
    return total_loss / total_examples


def evaluate(model, loader, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == targets).sum().item())
            total_loss += float(loss.item()) * int(inputs.shape[0])
            total_examples += int(inputs.shape[0])
    return total_loss / total_examples, correct / total_examples


if __name__ == "__main__":
    main()
