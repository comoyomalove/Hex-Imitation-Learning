Short version:
- Tactics first.
- Evaluate Hex as a connection race.
- Search only meaningful moves.
- Cache aggressively.
- Avoid Python overhead.
- Never timeout.

[01:04]Sunday, 14. June 2026 at 01:04
Long Version
The main idea was not to search every move equally. Hex is a connection race, so the bot first evaluates which moves improve my connection and which moves damage the opponent’s connection.

FINAL AGENT FLOW

                 ┌────────────────────┐
                 │  get current board │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │  convert to 1D     │
                 │  + build hash/bits │
                 └─────────┬──────────┘
                           ↓
        ┌────────────────────────────────────┐
        │  emergency tactical layer          │
        │  1. can I win now?                 │
        │  2. can opponent win next? block   │
        └─────────┬──────────────────────────┘
                  ↓
        ┌────────────────────────────────────┐
        │  path evaluation with Dijkstra     │
        │  own stones = 0 cost               │
        │  empty cells = 1 cost              │
        │  enemy stones = blocked            │
        └─────────┬──────────────────────────┘
                  ↓
        ┌────────────────────────────────────┐
        │  rank moves                        │
        │  + shorten my path                 │
        │  + disrupt opponent path           │
        │  + bridge / virtual connection     │
        │  + avoid opponent fork threats     │
        └─────────┬──────────────────────────┘
                  ↓
        ┌────────────────────────────────────┐
        │  search only top candidates        │
        │  iterative-deepening alpha-beta    │
        │  + PVS / null-window optimization  │
        └─────────┬──────────────────────────┘
                  ↓
        ┌────────────────────────────────────┐
        │  return best move before 5 seconds │
        └────────────────────────────────────┘
 (edited)Sunday, 14. June 2026 at 01:05

[01:05]Sunday, 14. June 2026 at 01:05
Core strategy:

score ≈ opponent_connection_distance - my_connection_distance

So a good move either:

- makes my connection shorter,
- makes the opponent’s connection longer,
- creates/uses a bridge,
- attacks an opponent bridge,
- or prevents an immediate/fork threat.

The most important tactical part was the full-board win/block scan. I never pruned this step. Before doing any search, the bot checks every legal cell for:

my immediate winning move,
opponent’s immediate winning move.

Then I added a two-way danger guard. This catches positions where the opponent can create a double threat next turn (opponent can win by making two seperate moves). That was important because simply blocking one immediate threat can be too late in Hex.


Optimization summary:

+ 2D board list → flat 1D board of 121 cells
+ normal BFS win check → bitboard flood-fill win check
+ repeated Dijkstra calls → Dijkstra/path cache
+ repeated board states → Zobrist hash + transposition table
+ full-width search → top candidate search only
+ normal alpha-beta → alpha-beta + PVS/null-window search
+ basic move ordering → path ordering + killer moves
+ checking time every node → check time every 128 nodes

The biggest performance jump came from the 1D board, path cache, and transposition table. Earlier versions were usually reaching around depth 3 in ~4.3 seconds. After optimization, the agent could reach around depth 6 in roughly the same time, with tens of thousands of searched nodes and many cache/TT hits.


Adyansh: I'm new here, say hi! — 14.06.26, 01:07Sunday, 14. June 2026 at 01:07
Extra reference that inspired the idea:
https://cdn.aaai.org/AAAI/2000/AAAI00-029.pdf

This paper explains why  full width search is hard in Hex and why connection based evaluation, virtual connections, and electrical-resistance-style thinking are useful. My implementation is a much simpler hackathon version of that idea: Dijkstra path distance + bridge discounts + selective alpha-beta search.
