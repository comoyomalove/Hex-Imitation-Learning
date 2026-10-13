# HEX-ML — case study

**One line:** my first machine-learning project — a small neural network that learned to play 11×11 Hex by copying expert agents, and an honest record of why it still loses.

Status: v1, 2026-10-13 (main `95d57bb`). Open source since 2026-10-13: my code under MIT, data, models and reports under CC BY 4.0. Tests were not re-run for this entry (no Python environment set up on this machine; the README lists 37 tests). Paths are relative to this file.

## The problem / why I built it

- In June 2026 I took part in the **AI Club Aachen e.V.** Hex game-AI hackathon. My friend **Adyansh Gupta**'s agents won; mine were search and heuristic bots.
- My question: can a neural network learn Hex just by watching a strong player? It was the first time I trained a model.
- Adyansh later shared a second agent, so I had two experts to learn from ([README](../../README.md)).

## What I decided

1. **Imitation, not reinforcement learning.** The expert's move is the label; the loss is cross-entropy (`scripts/train_11x11.py`). There is no reward function.
2. **Varied positions from my own bots.** The expert played against my five earlier hackathon agents, so it answered many kinds of positions.
3. **Correction rounds I designed (DAgger-style).** My trained agent played the expert, and the expert's answers in positions my agent got itself into became new training data. This targets the main weakness: small mistakes leading into positions it never saw.
4. **Split by game, never by move.** Held-out games are truly unseen; a test enforces it.
5. **A small model.** `TinyHexPolicyNet` in PyTorch: the 4×11×11 board (484 inputs) → 256 hidden → 121 move scores, illegal moves masked. Not a CNN. Trained on a laptop GPU (Apple MPS).
6. **Report the real numbers, drop the flattering one.** An early "beats the expert 5/5" came from a board-orientation bug. It is marked invalid, and a regression test guards the orientation (`f9d5cde`).

## Results

| What | Number | Source |
|---|---|---|
| Training data | 173,365 expert moves: 138,084 vs my agents + 35,281 from correction rounds | [01_PROJECT_REPORT.md](../01_PROJECT_REPORT.md), `reports/data/` |
| Exact expert move, held-out games | **70.8%** | `reports/training/imitation_11x11_structured_plus_dagger_round2_combined_20260729_011307.md` (epoch 32, lowest validation loss) |
| Expert move in its top 5 | **92.9%** | same |
| Against the expert | loses every recorded game after the orientation fix, usually within ~23 moves | [README](../../README.md) |

## Milestones

### 2026-07-31 → 08-03 — First delivery

- **What happened:** Agent, tests, browser replay demo, models and reports went into this repo (`bcf5ca9` … `64f0152`).

### 2026-10-13 — Open-sourced, with honest numbers

- **What happened:** I corrected the accuracy to the verified 70.8% / 92.9%, marked the 5/5 result invalid, added limitations, reframed the repo as my first ML project, and credited Adyansh and AI Club Aachen (`f9d5cde`). Then a contributing guide, ideas list and "Room for improvement" section (`71be23d`), a code of conduct and the project story (`cb7930d`).
- **Decision and why:** Licences follow who made what.
  - My code: MIT (`cb7930d`).
  - Adyansh's two agents: MIT with his agreement; his copyright stays on them (`6c0f1fb`).
  - Data, recorded games, reports and model weights: CC BY 4.0 under both our names, because his agents labelled every move (`7cf0c7a`).
  - Documentation prose: CC BY 4.0, mine (`95d57bb`).
  - AI Club Aachen's `aica-gamelib` (MIT) is installed as a dependency, not copied in.
- **Evidence:** [LICENSE](../../LICENSE), [LICENSE-DATA](../../LICENSE-DATA), [THIRD_PARTY.md](../../THIRD_PARTY.md), the commits above.
- **What I learned:** The honest number is lower than the one I first had, and it's the one worth publishing. Credit follows the data: his agents made the labels, so the dataset carries both our names.

## Tech stack

| Layer | Tool |
|---|---|
| Model and training | Python 3.12, PyTorch (Apple MPS) |
| Hex engine | `aica-gamelib` by AI Club Aachen e.V. (MIT) |
| Tests | pytest (37, per README) |
| Demo | HTML/JS replay viewer; optional local Python server for live matches |

## What's next

- Reinforcement learning that starts from this imitation policy and rewards checkable game conditions (wins, threats, blocks, connection progress). Written up as an essay, *Designing Reward and Environment Signals for Reinforcement Learning in Hex* — unpublished, a design proposal; no RL experiment has been run.
- More data, longer training and a CNN ([IDEAS.md](../IDEAS.md)). Contributions welcome ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

## Honest limits

- The agent does not beat the expert. It copies well in familiar positions and breaks down in positions its own mistakes create; it is weakest in the endgame.
- Compute limited me: I could not generate more data or train longer.
- I have not published a post-fix evaluation against weaker agents, so I claim no win rate at all.

---

Update this when: a milestone ends.
