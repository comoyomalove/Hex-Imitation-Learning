# Structured Stage 2 Data Quality Summary

- game logs analyzed: 50
- total turns: 350
- average turns/game: 7.00

## By opponent

| Opponent | Games | Winner examples | Avg turns | Min-Max turns | Illegal moves | Expert P1/P2 | Winner distribution |
|---|---:|---:|---:|---:|---:|---:|---|
| hybrid_phase_agent | 10 | 32 | 6.80 | 5-9 | 0 | 5/5 | {'2': 6, '1': 4} |
| mcts_lite_hex_agent | 10 | 33 | 6.90 | 5-9 | 0 | 5/5 | {'2': 7, '1': 3} |
| minimax_lite_hex_agent | 10 | 29 | 6.10 | 5-8 | 0 | 5/5 | {'2': 7, '1': 3} |
| smart_hex_agent | 10 | 40 | 7.90 | 5-9 | 0 | 5/5 | {'1': 9, '2': 1} |
| threat_adaptive_dijkstra_agent | 10 | 35 | 7.30 | 6-9 | 0 | 5/5 | {'2': 7, '1': 3} |

## Timing note

The Stage 2 logs do not yet store wall-clock elapsed seconds per game. This summary uses average turns per game as a rough size/speed proxy. Before choosing between 100, 500, or more games per agent, the next generation run should record elapsed seconds by opponent.

## Figures

- `reports/figures/structured_stage2_examples_by_opponent.png`
- `reports/figures/structured_stage2_avg_turns_by_opponent.png`
- `reports/figures/structured_stage2_winner_move_heatmap.png`
