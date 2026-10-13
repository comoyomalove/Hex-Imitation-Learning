# Contributing

Thank you for looking at this project. It is my first machine-learning
project, so please be kind: beginner-friendly explanations and constructive
suggestions are the most helpful kind of feedback.

## Run it locally

Requires Python 3.12+.

    python3.12 -m venv .venv
    source .venv/bin/activate        # fish: source .venv/bin/activate.fish
    pip install -r requirements.txt
    PYTHONPATH=src python -m pytest -q

Browser demo: open `demo_hex_board/index.html` (replay), or run
`PYTHONPATH=src python scripts/demo_server.py` and open http://127.0.0.1:5050
for live matches.

## Propose an idea

- **Issues** for concrete bugs or a specific improvement. Ready-to-adapt
  examples are in [docs/IDEAS.md](docs/IDEAS.md).
- **Discussions** (if enabled) for open-ended questions, research directions
  or "have you tried…?" ideas.
- Please say what you expect to change and how it could be measured
  (accuracy, win rate vs a named opponent, game length, etc.).

## Pull requests

- Keep changes small and focused; one idea per pull request.
- Run the tests before opening a pull request, and add a test for new behaviour.
- Do not commit large datasets, model checkpoints or any credentials.
- Credit external code and data sources (see "Attribution" in the README).

## Be kind

Assume good intent, explain rather than correct, and remember that everyone
here is learning.
