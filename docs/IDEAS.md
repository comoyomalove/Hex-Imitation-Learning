# Ideas — ready-to-paste issue texts

Each block below can be pasted into a new GitHub Issue as-is.

---

## 1. Benchmark the trained agent against random and heuristic bots

**Problem.** After the board-orientation fix, there is no trustworthy win-rate
evaluation of the trained agent against a random player or the heuristic
agents in `external_agents/`. The only verified playing result is 0 wins
against the expert.

**Proposal.** Add an evaluation script that plays N games per opponent (both
colours, fixed seeds) and reports win rate with a confidence interval.

**Done when.** A reproducible report exists for random + each heuristic agent,
and the README results section links it.

---

## 2. Replace the MLP with a small CNN policy

**Problem.** `TinyHexPolicyNet` is Flatten → Linear → ReLU → Linear. It has no
built-in notion of neighbouring cells, which matters a lot in Hex.

**Proposal.** Add a small convolutional policy (a few 3x3 conv layers, same
4x11x11 input and 121-move output) and train it on the same game-id split.

**Done when.** Top-1/top-5 on the same validation split are compared with the
current model (70.8% / 92.9%) in a short report.

---

## 3. Investigate the endgame weakness (61.1% top-1)

**Problem.** Expert-move agreement drops from ~72% in the opening to 61.1% in
the endgame (`reports/analysis/phase_validation_current_model.md`).

**Proposal.** Explore one of: more endgame examples, phase-weighted loss, or a
shallow win/block check at the end of the game. Measure per-phase accuracy
before and after.

**Done when.** Per-phase accuracy is reported for the old and new approach.

---

## 4. Prototype an RL environment for fine-tuning

**Problem.** The project stops at imitation learning plus one DAgger-style
correction round. No RL environment or reward exists yet.

**Proposal.** Implement a minimal reset/step Hex environment with legal-action
masks and an acting-player reward perspective, starting with a win/loss
reward only. Fine-tune the imitation policy against a fixed opponent.

**Done when.** The environment has tests for legality, terminal detection and
reward sign, and one short fine-tuning run is logged.

---

## 5. Real training curves and a replay GIF

**Problem.** The PNGs in `reports/figures/` come from smoke tests and pilots,
not from the final model.

**Proposal.** Plot loss and top-1/top-5 per epoch for the final combined run,
and record a short GIF of a demo replay.

**Done when.** The new figures are in `reports/figures/` and embedded in the
README with a one-line caption each.
