# Hex Board Demo - How to Use

Browser-based visualization for watching Hex agents play, including the
trained imitation-learning agent.

There are two modes. Replay mode works with zero setup; live mode needs the
small Python backend.

## Mode 1: Replay saved games (no setup needed)

1. Open demo_hex_board/index.html in any modern browser (double-click it).
2. Set Mode to "Replay saved game".
3. Pick a game from the "Replay log" dropdown (bundled sample games:
   the trained agent vs each opponent, both player orders).
4. Click "Load game", then use Play / Next / Prev / the slider to step
   through moves.

In replay mode the Player A / Player B dropdowns are intentionally disabled:
the players come from the saved log, so they cannot be changed.

## Mode 2: Live games (agents play in real time)

Requires the project environment (see the main README for setup).

    cd <project root>
    source .venv/bin/activate            # fish: source .venv/bin/activate.fish
    PYTHONPATH=src python scripts/demo_server.py

Then open http://127.0.0.1:5050 in a browser (do not open the file directly -
use the URL, so the page can reach the backend).

1. Set Mode to "Start new live game".
2. Choose Player A = Player 1 / Blue (connects left-to-right) and
   Player B = Player 2 / Red (connects top-to-bottom). The dropdowns list
   every available agent: the trained imitation agent, the expert
   (hackathon winner by Adyansh Gupta), the v7 candidate expert, a random
   baseline, and the five algorithm-based opponent agents. When the backend
   is running, this list is discovered automatically from the repository.
3. Click "Start new live game", then "Step one live move" for a single move,
   or "Auto-step live game" to watch the whole game unfold move by move.
4. The status line shows which agent is thinking and how long each move took
   (the expert searches for several seconds; the neural network answers in
   milliseconds - a nice visible contrast).
5. "Save live log" writes the full game to logs/games/demo_live/ in the same
   JSON format as the training data, including per-move timings.

## Reading the board

- Blue stones = Player 1, goal: connect left edge to right edge.
- Red stones = Player 2, goal: connect top edge to bottom edge.
- Gold outline = the most recent move.
- Hover a cell to see its row/column coordinates.

## Suggested demo matchups

- trained_imitation_agent vs random_legal_agent - fast, the trained agent
  usually wins.
- trained_imitation_agent vs hackathon_winner_agent - shows the gap to the
  expert and the per-move thinking-time difference.
- trained_imitation_agent vs any of the algorithm-based agents - mixed
  results, interesting games.

## Known limitations

- Live mode needs the local backend; the static page only replays.
- The static page's agent dropdowns use a built-in list; with the backend
  running, agents are discovered dynamically from external_agents/.
- Visual styling is functional rather than final.
