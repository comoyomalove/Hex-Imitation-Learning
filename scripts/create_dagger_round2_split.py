from __future__ import annotations

import json
import random
from pathlib import Path

records_path = Path("data/processed/dagger_round2_smartopen_3000_records.jsonl")
out_path = Path("data/splits/dagger_round2_smartopen_3000_split_manifest.json")

ids = []
seen = set()
for line in records_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    game_id = json.loads(line)["game_id"]
    if game_id not in seen:
        seen.add(game_id)
        ids.append(game_id)

random.Random(42).shuffle(ids)
n = len(ids)
train_end = int(n * 0.70)
validation_end = train_end + int(n * 0.15)
splits = {
    "train": sorted(ids[:train_end]),
    "validation": sorted(ids[train_end:validation_end]),
    "test": sorted(ids[validation_end:]),
}
manifest = {
    "schema_version": "split_manifest_v1",
    "split_method": "by_game_id",
    "seed": 42,
    "source_records": str(records_path),
    "splits": splits,
    "counts": {key: len(value) for key, value in splits.items()},
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"games": n, "counts": manifest["counts"], "path": str(out_path)}, indent=2))
