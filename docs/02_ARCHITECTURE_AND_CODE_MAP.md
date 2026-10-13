# Hex-ML: Architecture and Code Map

Purpose of this document: let an auditor find "where is X" without reading
the full codebase, and understand how the pieces connect.

---

## 1. High-level pipeline

```
expert agent (+ opponents) play games
        v
full-state game logs (JSON)   <-- src/hex_ml/data/game_log.py, schemas.py
        v
imitation records (JSONL)     <-- src/hex_ml/data/dataset_builder.py
        v
leakage-free split by game_id <-- src/hex_ml/data/splitting.py
        v
PyTorch training               <-- scripts/train_11x11.py, src/hex_ml/training/
        v
trained checkpoint (.pt)      <-- models/checkpoints/, models/final/
        v
TrainedHexAgent (inference)   <-- src/hex_ml/agents/trained_agent.py
        v
evaluation / browser demo     <-- scripts/evaluate_*.py, demo_hex_board/
```

The trained agent never calls the expert agent at inference time. The
expert is only used offline, for data generation.

---

## 2. Class / module map

### Core game logic (`src/hex_ml/core/`)
| Symbol | File | Purpose |
|---|---|---|
| `create_empty_board`, `get_legal_moves`, `is_legal_move` | `board.py` | Board creation and legality checks |
| `apply_move`, `game_status`, `other_player` | `game.py` | Move application and win detection (Player 1 = left-right, Player 2 = top-bottom) |
| `DEFAULT_BOARD_SIZE` | `constants.py` | 11x11 is the delivered board size; smaller sizes were smoke-test only |

### Encoding (`src/hex_ml/encoding/`)
| Symbol | File | Purpose |
|---|---|---|
| `encode_board_channels` | `board_encoder.py` | Board -> `4 x 11 x 11` tensor for the model |

### Data (`src/hex_ml/data/`)
| Symbol | File | Purpose |
|---|---|---|
| `MoveLogRecord` | `schemas.py` | Full-state log record schema |
| `save_game_log`, `replay_game_log`, `board_to_list` | `game_log.py` | Persist and verify game logs |
| `imitation_records_from_game_log`, `save_imitation_records_jsonl` | `dataset_builder.py` | Extract expert-move training examples |
| `split_records_by_game_id`, `save_split_manifest` | `splitting.py` | Leakage-free split |

### Agents (`src/hex_ml/agents/`)
| Class | File | Purpose |
|---|---|---|
| `TrainedHexAgent` | `trained_agent.py` | **The delivered ML agent.** Loads a checkpoint, encodes board, scores 121 cells, masks illegal moves, picks a legal move. No expert calls. |
| `GamelibAgentWrapper` | `gamelib_wrapper.py` | Generic adapter for any `gamelib`-style agent (used for expert, v7, and the 5 algorithm-based agents) |
| `ExpertAgentWrapper` | `expert_wrapper.py` | Thin convenience wrapper around the hackathon winner agent |
| `discover_gamelib_agents` | `discovery.py` | Finds agents under `external_agents/other_gamelib_agents/*/agent.py` |
| `RandomLegalAgent`, `FirstLegalAgent` | `baselines.py` | Simple baselines for smoke tests/evaluation |

### Training (`src/hex_ml/training/`)
| Symbol | File | Purpose |
|---|---|---|
| `TinyHexPolicyNet` | `model.py` | The neural network: input `4x11x11`, output 121 move scores |
| `ImitationJsonlDataset` | `dataset.py` | PyTorch Dataset over imitation JSONL records |

### Scripts (`scripts/`, selected)
| Script | Purpose |
|---|---|
| `train_11x11.py` | Main training entry point (heavily commented) |
| `generate_structured_opponent_dataset.py` | Expert vs. algorithm-based opponents, parallel game generation |
| `generate_trained_vs_winner_logs.py` | Trained agent vs. expert, with smart opening sampling (top-k + temperature) to avoid deterministic repeat games |
| `build_dagger_round2_dataset.py` | Extracts expert corrections from trained-vs-expert logs |
| `build_combined_dataset.py` | Merges structured + DAgger datasets without overwriting inputs |
| `analyze_phase_performance.py`, `mine_hard_examples.py` | Post-hoc analysis of where the model is weak |
| `demo_server.py` | Optional local backend for live browser matches |
| `evaluate_trained_agent_round1.py`, `evaluate_v7_vs_old_winner.py` | Match evaluation scripts |

---

## 3. Where things live (quick index)

| Looking for... | Location |
|---|---|
| The trained model weights | `models/final/imitation_11x11_best.pt` |
| Model provenance (which dataset trained it) | embedded in checkpoint dict: `run_id`, `dataset_path`, `split_manifest`, `metrics` |
| The expert agent | `Hackaton-Winner-Agent/final_agent.py` (created by Adyansh Gupta, hackathon winner; external, used with permission) |
| The candidate "v7" expert | `external_agents/other_gamelib_agents/v7_resistance_agent/` (created by Adyansh Gupta; external, used with permission) |
| The 5 algorithm-based opponents | `external_agents/other_gamelib_agents/{hybrid_phase_agent, mcts_lite_hex_agent, minimax_lite_hex_agent, smart_hex_agent, threat_adaptive_dijkstra_agent}/` (Naz's earlier hackathon work) |
| Leakage-free split manifests | `data/splits/*.json` |
| Training reports (per run) | `reports/training/*.md` |
| Phase / hard-example analysis | `reports/analysis/` |
| Orientation bug regression test | `tests/test_gamelib_orientation.py` |
| Browser visualization | `demo_hex_board/index.html`, `app.js`, `styles.css` |

---

## 4. Design decisions worth knowing (for auditing)

- **Why 11x11 and not smaller boards?** 3x3/small boards were used only for
  pipeline smoke tests; they are not real Hex and were never used for the
  delivered model.
- **Why structured opponents instead of only random play?** Random play
  under-tests the expert; it rarely creates situations requiring real
  strategy. The 5 algorithm-based agents create more varied, meaningful
  board states for the expert to respond to.
- **Why a generic `GamelibAgentWrapper` instead of one wrapper per agent?**
  All opponent agents (expert, v7, and the 5 algorithm-based agents) share
  the same `gamelib` interface, so one adapter loads any of them by path.
- **Why is the DAgger-only model's accuracy not used as "the" result?** It
  was trained and evaluated on a narrower, easier distribution (short
  trained-vs-expert games). Comparing it to the structured-only model
  without a shared held-out benchmark would be misleading; this is flagged
  explicitly rather than silently reported as an improvement.
- **Why keep three model checkpoints instead of one?** For auditability:
  each stage (structured-only, DAgger-only, combined) is a separate,
  reproducible experiment with its own dataset, split, and report.

---

## 5. Future work (out of scope for this version)

1. **Convolutional policy network.** Replace `TinyHexPolicyNet` with a CNN
   that can learn local Hex patterns (bridges, blocks, edge shapes) more
   directly than the current architecture.
2. **Formal RL environment.** A `HexEnv` with `reset()`/`step()`, legal-action
   masks, and reward computed from the correct acting-player perspective,
   would be required before any reinforcement-learning experiment. The
   existing game-logic modules (`core/board.py`, `core/game.py`) would be
   reused; a new environment interface layer, reward function, and self-play
   loop would need to be added and tested.
3. **Reward design.** Terminal win/loss reward first; shaped/potential-based
   rewards (e.g. connection-path improvement, threat blocking) only after
   the terminal-reward version is validated, to avoid rewarding a heuristic
   instead of the actual game objective.
4. **Held-out benchmark.** Evaluate the structured-only, DAgger-only, and
   combined models on one shared, fixed set of games/positions to make a
   fair comparison claim.
5. **Visualization polish.** Improve the browser demo's visual styling and
   simplify the live-mode setup.