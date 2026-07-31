import heapq
import random
from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


class ThreatAdaptiveDijkstraAgent(Agent):
    """Threat-adaptive Dijkstra Hex agent.

    Modes are chosen once from the current board before scoring candidates:
    1. Finish mode if our path is close.
    2. Panic block mode if opponent path is close.
    3. Defensive mode if opponent is ahead.
    4. Balanced mode otherwise.
    """

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

    def _winning_moves_for_player(self, board, player_id: int):
        wins = []
        for move in self._legal_moves(board):
            r, c = move
            board[r][c] = player_id
            if self._get_path_cost(board, player_id) == 0:
                wins.append(move)
            board[r][c] = -1
        return wins

    def _mode_weights(self, my_cost: int, opp_cost: int):
        if my_cost <= 2:
            return "finish", 25.0, 2.0
        if opp_cost <= 2:
            return "panic", 5.0, 25.0
        if opp_cost < my_cost:
            return "defensive", 6.0, 8.0
        return "balanced", 10.0, 4.0

    def _score_move(self, board, move, own_weight: float, block_weight: float) -> float:
        r, c = move
        if board[r][c] != -1:
            return -10**12

        old_own = self._get_path_cost(board, self.player_id)
        old_opp = self._get_path_cost(board, self.opponent_id)

        board[r][c] = self.player_id
        new_own = self._get_path_cost(board, self.player_id)
        new_opp = self._get_path_cost(board, self.opponent_id)
        own_neighbors = sum(1 for nr, nc in self._neighbors(r, c, len(board)) if board[nr][nc] == self.player_id)
        opp_neighbors = sum(1 for nr, nc in self._neighbors(r, c, len(board)) if board[nr][nc] == self.opponent_id)
        board[r][c] = -1

        own_reduction = old_own - new_own
        opp_increase = new_opp - old_opp
        contested_bonus = 3.0 if own_reduction > 0 and opp_increase > 0 else 0.0

        return (
            own_weight * own_reduction
            + block_weight * opp_increase
            + contested_bonus
            + 0.5 * own_neighbors
            + 0.4 * opp_neighbors
            + random.uniform(0, 0.001)
        )

    def get_move(self, state: State) -> Move:
        board = [row.copy() for row in state.board]
        legal = self._legal_moves(board)
        if not legal:
            # Should not happen during normal play.
            return Move(player=self.player_id, position=[0, 0])

        # Always take immediate win first.
        my_wins = self._winning_moves_for_player(board, self.player_id)
        if my_wins:
            best = max(my_wins, key=lambda m: self._score_move(board, m, 25.0, 2.0))
            return Move(player=self.player_id, position=list(best))

        # Block immediate opponent win second.
        opp_wins = self._winning_moves_for_player(board, self.opponent_id)
        if opp_wins:
            best = max(opp_wins, key=lambda m: self._score_move(board, m, 5.0, 25.0))
            return Move(player=self.player_id, position=list(best))

        my_cost = self._get_path_cost(board, self.player_id)
        opp_cost = self._get_path_cost(board, self.opponent_id)
        _mode, own_weight, block_weight = self._mode_weights(my_cost, opp_cost)

        best_move = legal[0]
        best_score = -float("inf")
        for move in legal:
            score = self._score_move(board, move, own_weight, block_weight)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move not in legal:
            best_move = random.choice(legal)
        return Move(player=self.player_id, position=list(best_move))


if __name__ == "__main__":
    ThreatAdaptiveDijkstraAgent().start()
