from pathlib import Path
import pytest
from hex_ml.data.schemas import ImitationRecord
from hex_ml.data.splitting import (
    assert_no_game_id_leakage,
    load_split_manifest,
    save_split_manifest,
    split_game_ids,
    split_records_by_game_id,
    unique_game_ids,
)


def make_record(game_id: str, turn_number: int = 0) -> ImitationRecord:
    return ImitationRecord(
        game_id=game_id,
        turn_number=turn_number,
        board_size=3,
        player_to_move=1,
        board_before=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        legal_moves=[(0, 0), (0, 1)],
        expert_move=(0, 0),
        expert_agent_name="winner_agent",
        opponent_name="random_agent",
        source_game_log=f"logs/games/{game_id}.json",
    )


def test_unique_game_ids_uses_game_ids_not_records():
    assert unique_game_ids(
        [make_record("game_b"), make_record("game_a"), make_record("game_a", 1)]
    ) == ["game_a", "game_b"]


def test_split_game_ids_is_reproducible_with_seed():
    ids = [f"game_{i}" for i in range(10)]
    assert split_game_ids(ids, seed=123) == split_game_ids(ids, seed=123)


def test_split_game_ids_has_no_overlap_between_splits():
    split_map = split_game_ids([f"game_{i}" for i in range(12)], seed=5)
    train, validation, test = (
        set(split_map["train"]),
        set(split_map["validation"]),
        set(split_map["test"]),
    )
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_assert_no_game_id_leakage_raises_for_duplicate_game_id():
    with pytest.raises(ValueError, match="appears in both"):
        assert_no_game_id_leakage(
            {"train": ["game_1"], "validation": ["game_1"], "test": []}
        )


def test_split_records_by_game_id_keeps_same_game_in_one_split():
    records = [
        make_record("game_1", 0),
        make_record("game_1", 1),
        make_record("game_2"),
        make_record("game_3"),
        make_record("game_4"),
    ]
    split_records, split_map = split_records_by_game_id(records, seed=7)
    assert (
        len({record.split for record in split_records if record.game_id == "game_1"})
        == 1
    )
    assert_no_game_id_leakage(split_map)


def test_save_and_load_split_manifest(tmp_path: Path):
    split_map = split_game_ids([f"game_{i}" for i in range(6)], seed=11)
    manifest_path = tmp_path / "split_manifest.json"
    save_split_manifest(manifest_path, split_map, seed=11)
    manifest = load_split_manifest(manifest_path)
    assert manifest["schema_version"] == "split_manifest_v1"
    assert manifest["split_method"] == "by_game_id"
    assert manifest["seed"] == 11
    assert manifest["splits"] == split_map
    assert sum(manifest["counts"].values()) == 6
