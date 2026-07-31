"""Tiny PyTorch training smoke test for the imitation pipeline."""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from hex_ml.training.dataset import ImitationJsonlDataset
from hex_ml.training.model import TinyHexPolicyNet


def main() -> Path:
    torch.manual_seed(17)
    dataset_path = Path("data/processed/tiny_imitation_records.jsonl")
    dataset = ImitationJsonlDataset(dataset_path)
    board_size = int(dataset.records[0]["board_size"])
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = TinyHexPolicyNet(board_size=board_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for epoch in range(2):
        total_loss = 0.0
        batches = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        losses.append(total_loss / max(1, batches))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (
        Path("models/checkpoints") / f"imitation_smoke_{timestamp}_epoch_002.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "board_size": board_size,
            "epoch": 2,
            "losses": losses,
            "dataset_path": str(dataset_path),
            "seed": 17,
        },
        checkpoint_path,
    )
    log_path = Path("logs/training") / f"smoke_{timestamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for epoch, loss in enumerate(losses, start=1):
            f.write(json.dumps({"epoch": epoch, "train_loss": loss}) + "\n")
    report_path = Path("reports/training") / f"smoke_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Tiny Training Smoke Test\n\nExamples: {len(dataset)}\n\nLosses: {losses}\n\nCheckpoint: {checkpoint_path}\n",
        encoding="utf-8",
    )
    print(f"examples={len(dataset)}")
    print(f"losses={losses}")
    print(f"checkpoint={checkpoint_path}")
    print(f"log={log_path}")
    print(f"report={report_path}")
    return checkpoint_path


if __name__ == "__main__":
    main()
