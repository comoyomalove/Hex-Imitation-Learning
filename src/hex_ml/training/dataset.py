"""Tiny dataset loader for imitation-learning JSONL records."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from hex_ml.encoding.board_encoder import encode_board_channels


class ImitationJsonlDataset(Dataset):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        board = np.array(record["board_before"], dtype=np.int8)
        player = int(record["player_to_move"])
        encoded = encode_board_channels(board, player)
        row, col = record["expert_move"]
        target = int(row) * int(record["board_size"]) + int(col)
        return torch.tensor(encoded, dtype=torch.float32), torch.tensor(
            target, dtype=torch.long
        )
