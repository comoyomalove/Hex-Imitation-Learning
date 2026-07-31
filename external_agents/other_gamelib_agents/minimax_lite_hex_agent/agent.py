import heapq
import random
from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


class MinimaxLiteAgent(Agent):
    """Safe Minimax-lite Hex agent.

    Strategy:
    - Dijkstra/path-cost candidate scoring.
    - Immediate win check.
    - Immediate opponent win block.
    - One-ply opponent best-response check.
    - Contested-cell bonus for moves that help us and hurt opponent.

    Higher evaluation is better for this agent.
    """

    OUR_CANDIDATES = 8
    OPP_RESPONSES = 5

    def initialize(self, init_data: dict) -> None:
        self.player_id = init_data["player_id"]
        self.opponent_id = 1 - self.player_id

    def _neighbors(self, r: int, c: int, size: int):
        for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    def _legal_moves(self, board):
        size = len(board)
        return [(r, c) for r in range(size) for c in range(size) if board[r][c] == -1]

    def _get_path_cost(self, board, player_id: int) -> int:
        """Dijkstra cost: own stones=0, empty=1, opponent stones=blocked."""
        size = len(board)
        opponent = 1 - player_id
        inf = 10**9
        dist = [[inf] * size for _ in range(size)]
        pq = []

        def cell_cost(r: int, c: int) -> int:
            if board[r][c] == player_id:
                return 0
            if board[r][c] == -1:
                return 1
            if board[r][c] == opponent:
                return inf
            return inf

        if player_id == 0:
            for r in range(size):
                cost = cell_cost(r, 0)
                if cost >= inf:
                    continue
                dist[r][0] = cost
                heapq.heappush(pq, (cost, r, 0))
            is_target = lambda r, c: c == size - 1
        else:
            for c in range(size):
                cost = cell_cost(0, c)
                if cost >= inf:
                    continue
                dist[0][c] = cost
                heapq.heappush(pq, (cost, 0, c))
            is_target = lambda r, c: r == size - 1

        while pq:
            cost, r, c = heapq.heappop(pq)
            if cost != dist[r][c]:
                continue
            if is_target(r, c):
                return cost
            for nr, nc in self._neighbors(r, c, size):
                step = cell_cost(nr, nc)
                if step >= inf:
                    continue
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    heapq.heappush(pq, (new_cost, nr, nc))
        return inf

    def _evaluate(self, board) -> float:
        """Positive is good for this agent."""
        own_cost = self._get_path_cost(board, self.player_id)
        opp_cost = self._get_path_cost(board, self.opponent_id)
        if own_cost >= 10**9 and opp_cost >= 10**9:
            return 0.0
        return float(opp_cost - own_cost)

    def _move_gradient_score(self, board, move, player_id: int) -> float:
        """Before/after path gradient. Higher is better for player_id."""
        r, c = move
        if board[r][c] != -1:
            return -10**12

        opponent = 1 - player_id
        old_own = self._get_path_cost(board, player_id)
        old_opp = self._get_path_cost(board, opponent)

        board[r][c] = player_id
        new_own = self._get_path_cost(board, player_id)
        new_opp = self._get_path_cost(board, opponent)
        own_neighbors = sum(1 for nr, nc in self._neighbors(r, c, len(board)) if board[nr][nc] == player_id)
        opp_neighbors = sum(1 for nr, nc in self._neighbors(r, c, len(board)) if board[nr][nc] == opponent)
        board[r][c] = -1

        own_improvement = old_own - new_own
        opp_worsening = new_opp - old_opp
        contested_bonus = 3.0 if own_improvement > 0 and opp_worsening > 0 else 0.0

        return (
            10.0 * own_improvement
            + 5.0 * opp_worsening
            + contested_bonus
            + 0.5 * own_neighbors
            + 0.4 * opp_neighbors
            + random.uniform(0, 0.001)
        )

    def _top_moves_for_player(self, board, player_id: int, n: int):
        legal = self._legal_moves(board)
        if not legal:
            return []
        scored = [(move, self._move_gradient_score(board, move, player_id)) for move in legal]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored[:n]]

    def _winning_moves_for_player(self, board, player_id: int):
        wins = []
        for move in self._legal_moves(board):
            r, c = move
            board[r][c] = player_id
            if self._get_path_cost(board, player_id) == 0:
                wins.append(move)
            board[r][c] = -1
        return wins

    def get_move(self, state: State) -> Move:
        board = [row.copy() for row in state.board]
        legal = self._legal_moves(board)
        if not legal:
            # Should not happen during normal play.
            return Move(player=self.player_id, position=[0, 0])

        # 1. Take immediate win.
        my_wins = self._winning_moves_for_player(board, self.player_id)
        if my_wins:
            # Prefer strongest gradient among winning moves.
            best = max(my_wins, key=lambda m: self._move_gradient_score(board, m, self.player_id))
            return Move(player=self.player_id, position=list(best))

        # 2. Block immediate opponent win.
        opp_wins = self._winning_moves_for_player(board, self.opponent_id)
        if opp_wins:
            # If multiple blocks, choose the one best for our path too.
            best = max(opp_wins, key=lambda m: self._move_gradient_score(board, m, self.player_id))
            return Move(player=self.player_id, position=list(best))

        # 3. Candidate pruning for our moves.
        candidates = self._top_moves_for_player(board, self.player_id, min(self.OUR_CANDIDATES, len(legal)))
        if not candidates:
            return Move(player=self.player_id, position=list(random.choice(legal)))

        best_move = candidates[0]
        best_value = -float("inf")

        # 4. One-ply minimax: assume opponent chooses the response worst for us.
        for move in candidates:
            r, c = move
            board[r][c] = self.player_id

            # If this candidate creates an immediate win, choose it.
            if self._get_path_cost(board, self.player_id) == 0:
                board[r][c] = -1
                return Move(player=self.player_id, position=list(move))

            opp_legal = self._legal_moves(board)
            if not opp_legal:
                value_after_opp = self._evaluate(board)
            else:
                opp_response_set = []
                # Always include immediate opponent wins if any exist after our move.
                opp_response_set.extend(self._winning_moves_for_player(board, self.opponent_id))
                # Add top gradient responses.
                opp_response_set.extend(
                    self._top_moves_for_player(board, self.opponent_id, min(self.OPP_RESPONSES, len(opp_legal)))
                )
                # Deduplicate while preserving order.
                seen = set()
                opp_candidates = []
                for om in opp_response_set:
                    if om not in seen and board[om[0]][om[1]] == -1:
                        seen.add(om)
                        opp_candidates.append(om)
                if not opp_candidates:
                    opp_candidates = [random.choice(opp_legal)]

                worst_for_us = float("inf")
                for opp_move in opp_candidates:
                    orow, ocol = opp_move
                    board[orow][ocol] = self.opponent_id
                    value = self._evaluate(board)
                    board[orow][ocol] = -1
                    if value < worst_for_us:
                        worst_for_us = value
                value_after_opp = worst_for_us

            board[r][c] = -1

            immediate = self._move_gradient_score(board, move, self.player_id)
            final_value = value_after_opp + 0.2 * immediate

            if final_value > best_value:
                best_value = final_value
                best_move = move

        if best_move not in legal:
            best_move = random.choice(legal)
        return Move(player=self.player_id, position=list(best_move))


if __name__ == "__main__":
    MinimaxLiteAgent().start()
