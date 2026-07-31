# Hex Board Demo

Browser-based replay/demo viewer for Hex-ML game logs.

Open `index.html` in a browser. The page includes:

- Hex board rendering for 11x11 logs
- previous/next/play controls
- player/opponent dropdowns listing available agents
- sample game selector
- generated summary panel

This demo is log-driven: it visualizes saved Hex game logs instead of reimplementing the full game engine.


## Demo mode controls

- Use **Start new live game** mode to select agents and generate a new game with the optional backend.
- Use **Replay saved game** mode to select an existing game log. In replay mode the agent dropdowns are disabled on purpose, because the players come from the saved log.
- The replay dropdown includes **-- No log: start a new live game --** as a safe option that switches back to live mode.
