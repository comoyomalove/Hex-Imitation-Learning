# Hex-ML: Development Journal (Curated)

This is a curated summary of the working development diary, design
decisions, and reflection notes. Full raw diary is kept in the working
project only and is not part of the course delivery.

---

## Timeline of major decisions

**Setup.** Python 3.12 virtual environment (required by `aica-gamelib`,
which needs Python >= 3.12). Dependencies pinned in `requirements.txt` /
`requirements.lock.txt`.

**Board size decision.** Early smoke tests used 3x3 boards to validate the
pipeline (legal moves, encoding, replay) quickly. This was explicitly never
meant as the delivered board size. All real data generation, training, and
evaluation use 11x11.

**Random vs. structured opponents.** Considered generating expert
demonstrations against a random-move baseline only. Rejected as the primary
data source: random play rarely forces the expert into genuinely strategic
decisions. Switched to using 5 of Naz's earlier algorithm-based hackathon
agents as opponents, producing more varied, meaningful board states.

**Generic agent wrapper.** Rather than writing one Python wrapper per
opponent agent, built a single `GamelibAgentWrapper` that loads any
`gamelib`-compatible agent by file path. This scaled cleanly when the "v7"
candidate expert agent was added later.

**Leakage-free splitting.** Chose to split datasets by `game_id` rather than
by individual move, specifically to avoid moves from the same game
appearing in both training and validation/test.

**Orientation bug (important critical-thinking episode).** Naz challenged
an evaluation result that looked impossible (trained agent apparently
beating the expert 5/5). Investigation found the local win-condition logic
had Player 1/Player 2 connection directions swapped relative to `gamelib`.
Fixed in `src/hex_ml/core/game.py` and locked in with a regression test
comparing local win detection against `gamelib.hex.engine.Engine`. Lesson:
do not trust win/loss interpretation from a system whose orientation has
not been independently verified against the reference implementation.

**Deterministic trained-vs-expert games.** Early trained-vs-expert data
generation produced near-identical repeated games because both the trained
model and the expert were deterministic. Fixed with "smart opening
sampling": top-k move sampling with temperature for the trained agent's
first few moves, plus an immediate-win check and a no-immediate-blunder
filter. This is engineered scaffolding around the learned policy, not a
replacement for it -- top-k sampling is only meaningful once the model has
already learned something; before training, an untrained model's "top
moves" would be arbitrary.

**DAgger-style data collection.** After confirming the trained agent still
lost quickly to the expert even with smart sampling, generated games where
the trained agent played the expert directly, then extracted the expert's
moves in those games as correction examples (DAgger-style: let the learner
create states, then ask the expert what it would have done).

**Phase-count caution.** Naz raised an important critique: game length
(number of moves) is not a reliable definition of "early/mid/late game" in
Hex, because a connection can be completed in as few as 11 stones on an
11x11 board. A 23-move game may already be tactically decisive. This
reframes the earlier phase-based analysis (opening/midgame/endgame by move
count) as an approximate signal, not a rigorous strategic label. Proposed
(not yet implemented) richer phase metadata: legal moves remaining,
immediate-win/threat detection, and path-cost estimates.

**Model provenance discipline.** After a period of active experimentation,
a provenance check (inspecting checkpoint metadata directly) revealed that
an assumed "combined structured+DAgger" training run had not actually
happened -- the active checkpoint was still the DAgger-only or
structured-only model. This was corrected by: (1) explicitly building the
combined dataset as a new artifact without overwriting either source
dataset, (2) training a clearly-named combined run, and (3) verifying via
the checkpoint's embedded `dataset_path`/`split_manifest`/`metrics` fields
rather than trusting memory or chat history. Lesson: always verify model
provenance from the artifact itself, not from conversation history.

---

## Reflection: what this project demonstrates about ML/agent design

- **Learned intelligence vs. engineered scaffolding.** The neural network
  provides learned intuition (board state -> plausible expert-like move).
  Legal-move masking, immediate-win checks, and opening-sampling are
  engineered rules around that learned policy. They serve different
  purposes and should not be confused with each other.
- **Dataset size is not the same as dataset coverage.** The DAgger dataset
  added tens of thousands of examples, but because trained-vs-expert games
  ended quickly, that data mostly covered early-game situations. More data
  did not automatically fix the previously identified endgame weakness.
- **Imitation learning replaces "reward" with "expert label."** There is no
  reward function anywhere in the delivered pipeline. The closest analogue
  to a reward is the expert-move label used in the cross-entropy loss. This
  is a deliberate and important distinction to make clearly, since it is a
  common point of confusion when comparing to reinforcement learning.
- **Why RL is future work, not the next trivial step.** A hybrid IL-to-RL
  approach is plausible (imitation learning provides a stable starting
  policy; RL could later optimize the actual win/loss objective through
  self-play), but it requires a formally tested environment (state
  transitions, legal-action masks, reward computed from the correct
  acting-player's perspective) before it can be trusted. Reward shaping
  (e.g. rewarding path improvement or threat-blocking) is distinct from
  outcome-based reward (win/loss) and distinct again from search/tree
  pruning, which is a decision-time action-selection method, not a reward
  design choice at all. Conflating these three ideas was an early source of
  confusion that is worth documenting for the essay/course discussion this
  project also supports.

---

## Appendix: environment notes

- Training used Apple Silicon MPS acceleration (`torch.device("mps")`) on
  the development Mac. Game generation (expert/opponent agents) is CPU-bound
  search/heuristic code and does not benefit from MPS.
- For running data generation on lab/Windows machines, minimal dependencies
  are: `numpy`, `torch`, `aica-gamelib`, `tqdm`, `pyyaml` (Python 3.12+
  required). Docker is optional; MPS acceleration is Mac-specific and would
  not be available in a container.
