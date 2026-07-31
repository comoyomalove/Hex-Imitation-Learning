# Hex-ML: Imitation Learning Agent for 11x11 Hex

Course delivery repository. This project trains a neural-network Hex agent
through imitation learning (behavioral cloning) from a strong expert agent.

## What this project is

1. A strong expert agent plays Hex against several opponents.
2. Full board states and expert moves are logged.
3. A PyTorch policy network learns: board state + player to move -> expert move.
4. The trained agent plays legal 11x11 Hex without calling the expert at inference time.

This is supervised imitation learning. There is no reinforcement-learning
reward function: the training signal is the expert move label, and the loss is
cross-entropy (see scripts/train_11x11.py).

## Attribution: what was provided vs what I built

### Provided / external (credit: Adyansh Gupta)

- Hackaton-Winner-Agent/final_agent.py - the original hackathon-winning agent,
  created by Adyansh Gupta, used as the expert/teacher for imitation learning.
- external_agents/other_gamelib_agents/v7_resistance_agent/ - a later candidate
  expert agent, also created by Adyansh Gupta.
- The aica-gamelib package (pip dependency, imported as gamelib) provides the
  Hex game framework these agents are built on.

### My earlier hackathon agents (used as structured opponents for data generation)

- external_agents/other_gamelib_agents/hybrid_phase_agent/
- external_agents/other_gamelib_agents/mcts_lite_hex_agent/
- external_agents/other_gamelib_agents/minimax_lite_hex_agent/
- external_agents/other_gamelib_agents/smart_hex_agent/
- external_agents/other_gamelib_agents/threat_adaptive_dijkstra_agent/

These are algorithm-based (search/heuristic) agents, not ML agents. They were
used to create varied board states for the expert to respond to.

### Built for this course (the delivery focus)

- src/hex_ml/ - board logic, encoding, wrappers, dataset tools, training code,
  and the trained imitation agent (src/hex_ml/agents/trained_agent.py).
- scripts/ - data generation, training, evaluation, and analysis scripts.
- tests/ - test suite (37 tests), including leakage-free splitting and
  gamelib orientation regression tests.
- demo_hex_board/ - browser-based game visualization (see below).
- The trained model weights in models/.

## Setup

Requires Python 3.12+ (aica-gamelib requires >= 3.12).

    python3.12 -m venv .venv
    source .venv/bin/activate        # fish: source .venv/bin/activate.fish
    pip install -r requirements.txt

Run the tests:

    PYTHONPATH=src python -m pytest -q

## The trained model

- Final model: models/final/imitation_11x11_best.pt
- Input: 4 x 11 x 11 encoded board (own stones, opponent stones, empty cells,
  player-to-move). Output: 121 move scores, one per cell. Illegal moves are
  masked at inference.
- Training entry point: scripts/train_11x11.py (commented for readability).
- Data is split train/validation/test by game id (never by individual move)
  to prevent leakage; see data/splits/ manifests.

Model stages included under models/:

1. structured-only baseline (trained on expert demonstrations vs my structured agents),
2. DAgger-only experiment (expert corrections on states reached by the trained agent),
3. combined structured + DAgger model.

## Game visualization (browser demo)

Open demo_hex_board/index.html for replay mode, or run the optional backend
for live matches:

    PYTHONPATH=src python scripts/demo_server.py
    # then open http://127.0.0.1:5050

In live mode you can select two agents from dropdowns - for example the
trained imitation agent as Player A (Player 1 / Blue / left-to-right) against
any of the algorithm-based agents or the expert as Player B (Player 2 / Red /
top-to-bottom) - and step through the game move by move with per-move timing.

Known limitations (future polish): visual styling is functional rather than
final, and the live mode requires the local Python backend.

## Rules and orientation

Board: 11x11 Hex. Player 1 connects left-to-right; Player 2 connects
top-to-bottom (matching gamelib player 0 / player 1). An early orientation
mismatch was found, fixed, and is covered by a regression test
(tests/test_gamelib_orientation.py).

## Future work (not part of this delivery)

- Convolutional policy network (CNN) to better capture spatial Hex patterns.
- A formal RL environment (reset/step API, legal-action masks, reward
  perspective) and reward design for a hybrid IL-to-RL extension.
- Held-out benchmark comparison of the three model stages.
- Visualization polish.

## Where to look first (key classes and functions)

If you only have time to skim, these are the pieces that matter most:

| What | Where | Why it matters |
|---|---|---|
| Trained agent (the ML model) | `src/hex_ml/agents/trained_agent.py` -> `TrainedHexAgent`, `mask_illegal_moves` | Loads the trained weights, encodes the board, predicts a move, masks illegal cells. This is the delivered agent. |
| Policy network architecture | `src/hex_ml/training/model.py` -> `TinyHexPolicyNet` | The actual neural network: board in, per-cell move scores out. |
| State encoding | `src/hex_ml/encoding/board_encoder.py` -> `encode_board_channels` | Defines what the network sees (4-channel board state). |
| Training loop | `scripts/train_11x11.py` -> `main` | Where imitation learning happens: loads records, trains `TinyHexPolicyNet` with cross-entropy against expert moves. |
| Dataset construction | `src/hex_ml/data/dataset_builder.py` -> `imitation_records_from_game_log` | Turns a raw game log into (state, expert action) training pairs. |
| Leakage-free splitting | `src/hex_ml/data/splitting.py` -> `split_game_ids`, `assert_no_game_id_leakage` | Ensures train/validation/test never share a game. |
| External/expert agent adapter | `src/hex_ml/agents/gamelib_wrapper.py` -> `GamelibAgentWrapper` | Generic loader so any hackathon-style agent (expert or opponent) can be called the same way. |
| Expert agent specifically | `src/hex_ml/agents/expert_wrapper.py` -> `ExpertAgentWrapper` | The winner agent (Adyansh Gupta) used only to generate/label training data. |
| Opponent auto-discovery | `src/hex_ml/agents/discovery.py` -> `discover_gamelib_agents` | Finds every agent under `external_agents/` automatically. |
| Data generation with structured opponents | `scripts/generate_structured_opponent_dataset.py` | Explains and implements why algorithm-based opponents were used instead of random play. |
| Combining datasets | `scripts/build_combined_dataset.py` | Merges the structured and DAgger-round datasets with traceable provenance. |
| Phase-based evaluation | `scripts/analyze_phase_performance.py` -> `phase_from_record` | Breaks down accuracy by opening/midgame/endgame. |
| Tests | `tests/` | 37 tests covering board rules, encoding, splitting/leakage, and the gamelib orientation fix. |

Everything else (baselines, demo server, analysis scripts) is supporting
infrastructure around these core pieces.
