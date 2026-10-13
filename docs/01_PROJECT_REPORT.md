# Hex-ML: Project Report (Imitation Learning Agent)

Naz Akgül's first machine-learning project: an imitation-learning agent for
11x11 Hex.

This report has four sections: application context, data structures,
demonstrator functionality, and evaluation results / limitations.

---

## 1. Application Context

### The game
Hex, played on an 11x11 board. Player 1 (Blue) must connect the left and
right edges; Player 2 (Red) must connect the top and bottom edges. This
orientation matches the `aica-gamelib` (`gamelib`) convention: gamelib
player 0 = left-to-right, gamelib player 1 = top-to-bottom.

### The goal
Build a small, testable machine-learning demonstrator for an AI agent playing
Hex. The method is imitation learning / behavioral cloning: a strong expert
agent plays many games, its board-state-to-move decisions are logged, and a
neural network is trained to reproduce those decisions.

This is explicitly **not** an attempt to build a tournament-winning Hex
agent. The goal is to demonstrate a complete, leakage-free, testable ML
pipeline: data generation, encoding, training, evaluation, and
visualization.

### Why imitation learning (and not reinforcement learning)
Reinforcement learning was considered but deliberately deferred. Behavioral
cloning gives a stable initial policy quickly, using an existing strong
agent as a teacher, without needing a reward function, an RL environment, or
self-play infrastructure. RL is documented as future work (see Section 4 and
the companion architecture document).

### Who provided what
- The expert ("winner") agent and a later candidate expert ("v7") agent were
  created by Adyansh Gupta, the winner of the AI Club Aachen Hex hackathon,
  and are used with his permission. They are used here only as
  data-generation/evaluation opponents, not modified.
- The Hex game framework is `aica-gamelib` by AI Club Aachen e.V.
  (https://github.com/ai-club-aachen/game-ai-platform, MIT licence, copyright
  AI Club Aachen). It is a pip dependency and is not redistributed here.
- Five earlier algorithm-based (search/heuristic) agents were built by the
  project owner (Naz) in an earlier hackathon. They are not machine-learning
  agents; they are used here only as varied opponents to generate richer
  training data.
- Everything under `src/hex_ml/`, the training/evaluation scripts, the test
  suite, and the trained model were built by Naz for this project.

---

## 2. Data Structures

### Board representation
The board is an 11x11 grid of integers:
- `0` = empty
- `1` = Player 1 stone
- `2` = Player 2 stone

### Board encoding (model input)
Before being given to the neural network, a board is encoded as 4 channels
of 11x11:
1. current player's stones
2. opponent's stones
3. empty cells
4. player-to-move indicator (constant plane)

Shape: `4 x 11 x 11`. See `src/hex_ml/encoding/board_encoder.py`.

### Full-state game log record
Every move made by any agent in any generated game is logged with:

```
game_id, turn_number, board_size, player_to_move,
board_before, legal_moves_before, selected_move,
agent_name, agent_type,
board_after, was_legal, game_status_after_move, final_winner,
opponent_name, opponent_type, timestamp,
model_outputs (optional: move timing / confidence)
```

See `src/hex_ml/data/schemas.py`.

### Imitation-learning record (training example)
From full-state logs, only the expert agent's moves are extracted into
imitation records:

```
board_before + player_to_move + legal_moves  ->  expert_move
```

plus metadata: `game_id`, `turn_number`, `source_dataset`, `source_type`.
The **label** is `expert_move`. There is no reward signal anywhere in this
pipeline; this is supervised classification over the 121 board cells.

### Leakage-free train/validation/test split
Splitting is done by `game_id`, never by individual move. Moves from the
same game are highly correlated (they share a board history), so splitting
by move would leak information between train and validation/test and make
validation scores optimistic. See `src/hex_ml/data/splitting.py` and the
split manifests in `data/splits/`.

### Datasets used to train the delivered model
| Dataset | Source | Games | Records |
|---|---|---:|---:|
| Structured expert dataset | expert agent vs. 5 algorithm-based opponents | 5,000 | 138,084 |
| DAgger-style correction dataset | trained agent vs. expert agent, expert's moves extracted | 3,000 | 35,281 |
| Combined dataset | union of the above, re-split by game id | 5,500 | 173,365 |

---

## 3. Demonstrator Functionality

### Setup
Requires Python 3.12+ (`aica-gamelib` requires >= 3.12).

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run tests
```
PYTHONPATH=src python -m pytest -q
```
37 tests currently pass, covering legal-move extraction, board encoding,
game replay, schema validation, leakage-free splitting, gamelib orientation,
and illegal-move masking by the trained agent.

### Train the model
```
PYTHONPATH=src python scripts/train_11x11.py \
  --records-path data/processed/structured_plus_dagger_round2_records.jsonl \
  --split-path data/splits/structured_plus_dagger_round2_split_manifest.json \
  --epochs 100 --batch-size 128 --learning-rate 0.001 --hidden-size 256 \
  --device auto --run-prefix imitation_11x11_structured_plus_dagger_round2_combined
```
`scripts/train_11x11.py` is commented section-by-section (device selection,
leakage-free split loading, one training epoch, validation, main flow).

### Play / evaluate
`src/hex_ml/agents/trained_agent.py` wraps the trained network as a legal
Hex agent: it encodes the board, scores all 121 cells, masks illegal moves,
and returns the top legal move. It never calls the expert agent at
inference time.

### Visualize (browser demo)
`demo_hex_board/index.html` replays saved games move-by-move with a color
legend (Player 1/Blue left-to-right, Player 2/Red top-to-bottom, gold
outline = latest move). An optional local backend
(`scripts/demo_server.py`) enables **live matches**: pick any two agents
from dropdowns (e.g. the trained imitation agent vs. any algorithm-based
agent or the expert), and step through moves with per-move timing shown.

```
PYTHONPATH=src python scripts/demo_server.py
# open http://127.0.0.1:5050
```

Known limitation: the visualization is functional but not visually final,
and live mode depends on the local backend being run separately. Polishing
this further is listed as future work.

---

## 4. Evaluation Results and Limitations

### Model provenance (three stages, kept separate and auditable)
| Stage | Dataset | Best validation top-1 | Best validation top-5 |
|---|---|---:|---:|
| Structured-only baseline | structured expert dataset | ~70.7% | ~93.1% |
| DAgger-only experiment | DAgger correction dataset | ~86.8%* | ~97.4% |
| Combined (final model) | structured + DAgger | **70.8%** | **92.9%** |

Accuracy here means exact-move agreement with the expert on held-out
validation games (split by game id, so no validation game was seen in
training). Figures are the best checkpoint, selected by lowest validation
loss, as recorded in the per-run reports under `reports/training/`. The
final model's best checkpoint is epoch 32 of 100. (An earlier version of this
report gave "~73-74%" for the final model; that figure matched later epochs,
not the saved checkpoint, and has been corrected.)

*The DAgger-only model's high score reflects a narrower, easier
distribution (short, learner-induced games), not superior general play; it
is not directly comparable to the structured-only score without a shared
held-out benchmark. This benchmark comparison was planned but not completed
and is listed as future work.

### Key evaluation findings and their limitations

1. **Phase weakness.** Splitting validation examples by (turn-count-based)
   game phase showed the model was weakest late in games: opening ~72.8%
   top-1, midgame ~71.2%, endgame ~61.1%. Hard-example mining confirmed most
   of the highest-loss validation examples were labeled "endgame."

2. **Move-count is not a reliable phase label.** On an 11x11 board a
   connection can be completed in as few as 11 stones, so a 20-30 move game
   can already be tactically decisive. The phase labels above should be read
   as an approximate, occupancy-based signal, not a rigorous measure of
   strategic urgency. A more meaningful phase definition (legal moves
   remaining, immediate-win/threat detection, path-cost estimates) is
   proposed as future work rather than implemented.

3. **Distribution shift against the expert.** Games between the trained
   agent and the expert averaged only ~23 moves and the expert won nearly
   all recorded matches after an orientation fix (see below). This is
   evidence of behavioral-cloning distribution shift: the model imitates
   expert-visited states well but is less robust in states it creates
   itself. This motivated the DAgger-style data collection.

4. **DAgger data is early-game correction, not full-game coverage.**
   Because trained-vs-expert games end quickly, DAgger corrections mostly
   cover early/opening decisions, not deep endgame situations. Combining
   DAgger data with the structured dataset improved training-set fit but did
   not, by itself, resolve the endgame weakness identified in (1).

5. **Orientation bug found and fixed.** An early version of the local
   win-condition logic had Player 1/Player 2 connection directions swapped
   relative to `gamelib`. This produced impossible-looking results (e.g. the
   trained agent appearing to beat the expert 5/5). **That result is invalid:
   after the fix, the trained agent did not win any recorded game against the
   expert.** It was diagnosed by
   comparing against `gamelib.hex.engine.Engine`, fixed in
   `src/hex_ml/core/game.py`, and is now covered by a regression test
   (`tests/test_gamelib_orientation.py`). Results generated before the fix
   should not be trusted for win/loss interpretation; the underlying
   board/move data remained valid throughout because training uses the
   expert-move label, not the winner label.

6. **No reinforcement-learning reward is used or evaluated.** This is a
   deliberate scope boundary. See the architecture document's "future work"
   section for what an RL migration would require.

### Current limitations

- The agent cannot yet beat strong agents, including the expert it learned
  from.
- Training was limited by available compute: a small fully connected network
  trained on a laptop GPU (Apple MPS); a CNN and larger runs were not feasible
  within that budget.
- The next step is reinforcement learning via reward and environment design,
  starting from this imitation policy. The approach is developed in the
  unpublished essay *Designing Reward and Environment Signals for
  Reinforcement Learning in Hex*.

### Reproducibility
All datasets were split by game id with a fixed seed; split manifests are
included under `data/splits/`. Training scripts accept explicit
`--records-path`/`--split-path` arguments and log per-epoch metrics plus a
Markdown report per run. Model checkpoints store their dataset path, split
path, and metrics for provenance verification.