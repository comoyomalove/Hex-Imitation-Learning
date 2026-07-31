# Partial 11x11 Dataset Rebuild Summary

- source log directory: `logs/games/structured_opponents_11x11_500_per_opponent_w8`
- game logs audited: 1005
- audit problem count: 0
- imitation records: 39652
- total moves: 79602
- average moves/game: 79.21
- min/max moves/game: 23 / 121
- records path: `data/processed/structured_11x11_partial1005_imitation_records.jsonl`
- split path: `data/splits/structured_11x11_partial1005_split_manifest.json`

## Split counts by game id

{
  "train": 703,
  "validation": 150,
  "test": 152
}

## By opponent

| Opponent | Games | Winner examples | Avg moves/game |
|---|---:|---:|---:|
| hybrid_phase_agent | 500 | 18055 | 72.46 |
| mcts_lite_hex_agent | 500 | 21348 | 85.75 |
| minimax_lite_hex_agent | 5 | 249 | 99.80 |
