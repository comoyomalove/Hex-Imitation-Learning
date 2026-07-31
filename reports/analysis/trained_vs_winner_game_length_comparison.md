# Trained-vs-Winner Game Length Comparison

## Smart-open latest batches

| Batch | Games | Avg turns | Median | Min | Max | >=30 | >=40 | >=50 | Trained local wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 500 | 22.94 | 23.0 | 21 | 30 | 2 | 0 | 0 | 0 |
| 002 | 500 | 23.14 | 23.0 | 21 | 32 | 3 | 0 | 0 | 0 |
| 003 | 500 | 22.99 | 23.0 | 21 | 28 | 0 | 0 | 0 | 0 |
| 004 | 500 | 23.05 | 23.0 | 21 | 28 | 0 | 0 | 0 | 0 |
| 005 | 500 | 22.98 | 23.0 | 21 | 30 | 1 | 0 | 0 | 0 |
| 006 | 500 | 23.02 | 23.0 | 21 | 28 | 0 | 0 | 0 | 0 |

## Combined smart-open

- games: 3000
- average turns: 23.02
- median turns: 23.0
- min/max turns: 21 / 32
- games with at least 30 turns: 6
- games with at least 40 turns: 0
- games with at least 50 turns: 0
- trained local wins: 0

## Old deterministic partial comparison

- games: 302
- average turns: 23.50
- median turns: 23.5
- min/max turns: 21 / 26

## Interpretation note

Earlier smart-open diversity analysis averaged about 23 moves/game. This report recomputes the final completed batches from logs using corrected winner orientation.
