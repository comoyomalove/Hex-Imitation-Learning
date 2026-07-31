"""Analyze trained Hex imitation model performance by game phase."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hex_ml.training.dataset import ImitationJsonlDataset
from hex_ml.agents.trained_agent import TrainedHexAgent


def phase_from_record(record: dict) -> str:
    """Classify a position as opening/midgame/endgame by fraction of the board filled.

    This is a simple heuristic (first third of moves = opening, etc.), used
    only to break down evaluation accuracy by game phase for analysis
    reporting. It has no effect on training.
    """
    if "phase" in record and record["phase"]:
        return str(record["phase"])
    turn = int(record["turn_number"]); board_size = int(record["board_size"])
    frac = turn / float(board_size * board_size)
    if frac < 1 / 3: return "opening"
    if frac < 2 / 3: return "midgame"
    return "endgame"


def load_split_ids(path: Path, split: str) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return set(manifest["splits"][split])


def evaluate_phase(model, dataset, indices, device):
    if not indices:
        return {"examples": 0, "loss": None, "top1_accuracy": None, "top5_accuracy": None}
    loader = DataLoader(Subset(dataset, indices), batch_size=256, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0; total = 0; top1 = 0; top5 = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            total_loss += float(loss_fn(logits, y).item())
            total += int(x.shape[0])
            top1 += int((torch.argmax(logits, dim=1) == y).sum().item())
            k = min(5, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            top5 += int((topk == y.unsqueeze(1)).any(dim=1).sum().item())
    return {"examples": total, "loss": total_loss / max(total, 1), "top1_accuracy": top1 / max(total, 1), "top5_accuracy": top5 / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", type=Path, default=Path("data/processed/structured_11x11_expanded_5000_imitation_records.jsonl"))
    parser.add_argument("--split-path", type=Path, default=Path("data/splits/structured_11x11_expanded_5000_split_manifest.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/final/imitation_11x11_best.pt"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output-json", type=Path, default=Path("reports/analysis/phase_validation_current_model.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/analysis/phase_validation_current_model.md"))
    args = parser.parse_args()

    agent = TrainedHexAgent.from_checkpoint(args.checkpoint)
    model = agent.model
    device = next(model.parameters()).device
    dataset = ImitationJsonlDataset(args.records_path)
    split_ids = load_split_ids(args.split_path, args.split)
    phase_indices = defaultdict(list)
    for idx, record in enumerate(dataset.records):
        if record["game_id"] in split_ids:
            phase_indices[phase_from_record(record)].append(idx)

    phases = ["opening", "midgame", "endgame"]
    results = {phase: evaluate_phase(model, dataset, phase_indices.get(phase, []), device) for phase in phases}
    summary = {"records_path": str(args.records_path), "split_path": str(args.split_path), "checkpoint": str(args.checkpoint), "split": args.split, "device": str(device), "by_phase": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Phase-Based Validation Analysis", "", "- records: `" + str(args.records_path) + "`", "- split: `" + args.split + "`", "- checkpoint: `" + str(args.checkpoint) + "`", "", "| Phase | Examples | Loss | Top-1 accuracy | Top-5 accuracy |", "|---|---:|---:|---:|---:|"]
    for phase in phases:
        row = results[phase]
        if row["examples"] == 0:
            lines.append("| " + phase + " | 0 | n/a | n/a | n/a |")
        else:
            lines.append("| " + phase + " | " + str(row["examples"]) + " | " + format(row["loss"], ".4f") + " | " + format(row["top1_accuracy"], ".4f") + " | " + format(row["top5_accuracy"], ".4f") + " |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()