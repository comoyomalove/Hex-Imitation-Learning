"""Leakage-free dataset splitting helpers."""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterable
from hex_ml.data.schemas import ImitationRecord

SplitMap = dict[str, list[str]]


def unique_game_ids(records: Iterable[ImitationRecord]) -> list[str]:
    return sorted({record.game_id for record in records})


def split_game_ids(
    game_ids: Iterable[str],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> SplitMap:
    ids = sorted(set(game_ids))
    if not ids:
        raise ValueError("At least one game_id is required for splitting.")
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_fraction)
    validation_count = int(total * validation_fraction)
    if total >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
        if train_count + validation_count >= total:
            validation_count = max(1, total - train_count - 1)
    test_count = total - train_count - validation_count
    if total >= 3 and test_count < 1:
        test_count = 1
        if train_count > validation_count:
            train_count -= 1
        else:
            validation_count -= 1
    split_map = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count : train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count :]),
    }
    assert_no_game_id_leakage(split_map)
    return split_map


def assign_split(record: ImitationRecord, split_name: str) -> ImitationRecord:
    return ImitationRecord(
        game_id=record.game_id,
        turn_number=record.turn_number,
        board_size=record.board_size,
        player_to_move=record.player_to_move,
        board_before=record.board_before,
        legal_moves=record.legal_moves,
        expert_move=record.expert_move,
        expert_agent_name=record.expert_agent_name,
        opponent_name=record.opponent_name,
        source_game_log=record.source_game_log,
        split=split_name,
    )


def split_records_by_game_id(
    records: list[ImitationRecord], seed: int = 0
) -> tuple[list[ImitationRecord], SplitMap]:
    split_map = split_game_ids(unique_game_ids(records), seed=seed)
    game_to_split = {
        game_id: split_name
        for split_name, split_ids in split_map.items()
        for game_id in split_ids
    }
    return [
        assign_split(record, game_to_split[record.game_id]) for record in records
    ], split_map


def assert_no_game_id_leakage(split_map: SplitMap) -> None:
    seen: dict[str, str] = {}
    for split_name, game_ids in split_map.items():
        for game_id in game_ids:
            if game_id in seen:
                raise ValueError(
                    f"game_id {game_id!r} appears in both {seen[game_id]!r} and {split_name!r}"
                )
            seen[game_id] = split_name


def save_split_manifest(path: Path, split_map: SplitMap, seed: int) -> None:
    assert_no_game_id_leakage(split_map)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "split_manifest_v1",
        "split_method": "by_game_id",
        "seed": seed,
        "splits": split_map,
        "counts": {name: len(ids) for name, ids in split_map.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_fractions(
    train_fraction: float, validation_fraction: float, test_fraction: float
) -> None:
    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("Split fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Train/validation/test fractions must sum to 1.0.")
