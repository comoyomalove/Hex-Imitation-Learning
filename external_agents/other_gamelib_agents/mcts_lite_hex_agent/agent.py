import random
import heapq
from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


class MctsliteAgent(Agent):
    """Time-safe MCTS-lite Hex sparring agent.

    This is intentionally conservative:
    - top 3 root candidates
    - 2 rollouts per candidate
    - rollout depth 6
    - Dijkstra-guided rollout choices among top 3 moves
    """

    ROOT_CANDIDATES = 3
    ROLLOUTS_PER_CANDIDATE = 2
    ROLLOUT_DEPTH = 6
    ROLLOUT_TOP_K = 3

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

    def _dijkstra_score_move(self, board, move, player_id: int) -> float:
        """Score a move for player_id using before/after path-cost gradient."""
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

        return (
            10.0 * (old_own - new_own)
            + 4.0 * (new_opp - old_opp)
            + 0.6 * own_neighbors
            + 0.3 * opp_neighbors
            + random.uniform(0, 0.01)
        )

    def _top_moves_for_player(self, board, player_id: int, n: int):
        legal = self._legal_moves(board)
        if not legal:
            return []
        scored = [(move, self._dijkstra_score_move(board, move, player_id)) for move in legal]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored[:n]]

    def _status_by_path_cost(self, board):
        """Cheap terminal check for simulations."""
        if self._get_path_cost(board, self.player_id) == 0:
            return 1.0
        if self._get_path_cost(board, self.opponent_id) == 0:
            return -1.0
        return None

    def _evaluate_rollout_state(self, board) -> float:
        """Positive is good for this agent."""
        own = self._get_path_cost(board, self.player_id)
        opp = self._get_path_cost(board, self.opponent_id)
        if own >= 10**9 and opp >= 10**9:
            return 0.0
        return float(opp - own) / max(1, len(board))

    def _rollout(self, board, player_turn: int, depth: int) -> float:
        terminal = self._status_by_path_cost(board)
        if terminal is not None:
            return terminal
        if depth >= self.ROLLOUT_DEPTH:
            return self._evaluate_rollout_state(board)

        legal = self._legal_moves(board)
        if not legal:
            return self._evaluate_rollout_state(board)

        top_moves = self._top_moves_for_player(board, player_turn, min(self.ROLLOUT_TOP_K, len(legal)))
        if not top_moves:
            move = random.choice(legal)
        else:
            move = random.choice(top_moves)

        r, c = move
        board[r][c] = player_turn
        result = self._rollout(board, 1 - player_turn, depth + 1)
        board[r][c] = -1
        return result

    def get_move(self, state: State) -> Move:
        # Work on a copy so simulations cannot corrupt real state.
        board = [row.copy() for row in state.board]
        legal = self._legal_moves(board)
        if not legal:
            # Should not happen during normal play, but keep interface safe.
            return Move(player=self.player_id, position=[0, 0])

        candidates = self._top_moves_for_player(board, self.player_id, min(self.ROOT_CANDIDATES, len(legal)))
        if not candidates:
            return Move(player=self.player_id, position=list(random.choice(legal)))

        best_move = candidates[0]
        best_value = -float("inf")

        for move in candidates:
            r, c = move
            total = 0.0
            for _ in range(self.ROLLOUTS_PER_CANDIDATE):
                board[r][c] = self.player_id
                total += self._rollout(board, self.opponent_id, 0)
                board[r][c] = -1
            avg = total / self.ROLLOUTS_PER_CANDIDATE
            # Include immediate Dijkstra value to stabilize noisy rollouts.
            avg += 0.25 * self._dijkstra_score_move(board, move, self.player_id)
            if avg > best_value:
                best_value = avg
                best_move = move

        if best_move not in legal:
            best_move = random.choice(legal)
        return Move(player=self.player_id, position=list(best_move))


if __name__ == "__main__":
    MctsliteAgent().start()
