# Hex-ML: Imitation Learning Agent for 11x11 Hex

Naz Akgül's first machine-learning project. It trains a neural-network Hex
agent through imitation learning (behavioral cloning) from a strong expert agent.

## Results at a glance

- **Move agreement with the expert:** the delivered model picks exactly the
  expert's move in **70.8%** of positions from held-out games (games never seen
  in training), and the expert's move is among its top 5 guesses in **92.9%**.
  Source: `reports/training/imitation_11x11_structured_plus_dagger_round2_combined_20260729_011307.md`
  (best checkpoint, epoch 32, selected by lowest validation loss).
- **Playing strength:** the agent does **not** beat the expert. After the
  board-orientation fix (see "Rules and orientation"), it lost every recorded
  game against the expert, usually within ~23 moves.

## Current limitations

- The agent cannot yet beat strong agents. It imitates the expert well on
  familiar positions but breaks down in positions its own mistakes create
  (distribution shift), and it is weakest in the endgame.
- Training was limited by available compute: a small fully connected network
  (`TinyHexPolicyNet`, not a CNN) trained on a laptop GPU (Apple MPS).
- An early result that showed the trained agent "beating the expert 5/5" was
  caused by a board-orientation bug, not real play. It is not a valid result.
- The planned next step is reinforcement learning built on reward and
  environment design: starting from this imitation policy and rewarding
  checkable game conditions (wins, threats, blocks, connection progress).
  This idea is written up in an essay, *Designing Reward and Environment
  Signals for Reinforcement Learning in Hex* (unpublished), and summarised
  under "Future work" below.

## What this project is

1. A strong expert agent plays Hex against several opponents.
2. Full board states and expert moves are logged.
3. A PyTorch policy network learns: board state + player to move -> expert move.
4. The trained agent plays legal 11x11 Hex without calling the expert at inference time.

This is supervised imitation learning. There is no reinforcement-learning
reward function: the training signal is the expert move label, and the loss is
cross-entropy (see scripts/train_11x11.py).

## Attribution: what was provided vs what I built

### Provided / external

- **Expert/teacher agents - Adyansh Gupta (used with permission).**
  - Hackaton-Winner-Agent/final_agent.py - the original hackathon-winning agent,
    used as the expert/teacher for imitation learning.
  - external_agents/other_gamelib_agents/v7_resistance_agent/ - a later
    candidate expert agent.
- **Hex game framework - AI Club Aachen e.V.** The
  [aica-gamelib](https://github.com/ai-club-aachen/game-ai-platform) package
  (pip dependency, imported as `gamelib`; part of the AI Club Aachen Game AI
  Platform) provides the Hex game engine these agents are built on. It is
  MIT-licensed, copyright AI Club Aachen. It is installed via `requirements.txt`
  and not redistributed in this repository.

### My earlier hackathon agents (used as structured opponents for data generation)

- external_agents/other_gamelib_agents/hybrid_phase_agent/
- external_agents/other_gamelib_agents/mcts_lite_hex_agent/
- external_agents/other_gamelib_agents/minimax_lite_hex_agent/
- external_agents/other_gamelib_agents/smart_hex_agent/
- external_agents/other_gamelib_agents/threat_adaptive_dijkstra_agent/

These are algorithm-based (search/heuristic) agents, not ML agents. They were
used to create varied board states for the expert to respond to.

### Built by me for this project (the focus)

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

Known demo limitations (future polish): visual styling is functional rather than
final, and the live mode requires the local Python backend.

## Rules and orientation

Board: 11x11 Hex. Player 1 connects left-to-right; Player 2 connects
top-to-bottom (matching gamelib player 0 / player 1). An early orientation
mismatch was found, fixed, and is covered by a regression test
(tests/test_gamelib_orientation.py). Win/loss results produced before the fix
(including an apparent "5/5 wins against the expert") are invalid.

Note on figures: the charts in `reports/figures/` come from early smoke-test
and pilot runs (a few games or a 5-epoch toy run). They do not show the
delivered model's performance and are kept only as development history.

## Future work

- Convolutional policy network (CNN) to better capture spatial Hex patterns.
- A formal RL environment (reset/step API, legal-action masks, reward
  perspective) and reward design for a hybrid IL-to-RL extension. The design
  argument - that reward design and environment design are the same problem,
  because the environment decides which strategic ideas can be checked and
  rewarded - is developed in the unpublished essay *Designing Reward and
  Environment Signals for Reinforcement Learning in Hex*.
- Held-out benchmark comparison of the three model stages.
- Visualization polish.

## Room for improvement — ideas welcome

This is my first machine-learning project, and there is a lot left to try.
Suggestions, questions and pull requests are welcome (see
[CONTRIBUTING.md](CONTRIBUTING.md) and ready-made idea write-ups in
[docs/IDEAS.md](docs/IDEAS.md)). Honest list of open directions:

- **Stronger training with more compute.** The current policy is a small
  fully connected network trained on a laptop GPU. A CNN or residual network,
  longer runs and more data are likely to help.
- **RL fine-tuning via reward and environment design.** Start from the
  imitation policy and fine-tune with reinforcement learning: a reset/step
  environment, legal-action masks, acting-player reward perspective, and
  rewards built only from checkable game facts (wins, threats, blocks).
- **Endgame weakness.** Top-1 agreement with the expert drops to 61.1% in the
  endgame (vs ~72% in the opening). Targeted endgame data or search at the end
  of the game are open questions.
- **Evaluation against random and heuristic bots.** After the orientation fix
  there is no trustworthy win-rate evaluation against random play or the
  heuristic agents. A clean benchmark would be very valuable.
- **Board-size generalisation.** Everything is trained on 11x11. Whether a
  policy can transfer to other board sizes is untested.
- **Better visualisations.** Real training curves, move-probability heatmaps
  and a replay GIF would make results easier to read.

## Where to look first (key classes and functions)

If you only have time to skim, these are the pieces that matter most:

| What | Where | Why it matters |
|---|---|---|
| Trained agent (the ML model) | `src/hex_ml/agents/trained_agent.py` -> `TrainedHexAgent`, `mask_illegal_moves` | Loads the trained weights, encodes the board, predicts a move, masks illegal cells. This is the final agent. |
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
