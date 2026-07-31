import random
import heapq
from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move

class DijkstraAgent(Agent):
    def initialize(self, init_data: dict) -> None:
        self.player_id = init_data["player_id"]
        self.opponent_id = 1 - self.player_id

    def _get_path_cost(self, board, player_id):
        size = len(board)
        dist = {}
        pq = []
        
        # Initialize Dijkstra
        if player_id == 0:
            # Left (col 0) to Right (col size-1)
            for r in range(size):
                if board[r][0] == self.opponent_id: continue
                cost = 0 if board[r][0] == player_id else 1
                dist[(r, 0)] = cost
                heapq.heappush(pq, (cost, r, 0))
        else:
            # Top (row 0) to Bottom (row size-1)
            for c in range(size):
                if board[0][c] == self.opponent_id: continue
                cost = 0 if board[0][c] == player_id else 1
                dist[(0, c)] = cost
                heapq.heappush(pq, (cost, 0, c))
                
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist.get((r, c), float('inf')): continue
            
            # Check goal
            if player_id == 0 and c == size - 1: return d
            if player_id == 1 and r == size - 1: return d
            
            for dr, dc in [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != self.opponent_id:
                    weight = 0 if board[nr][nc] == player_id else 1
                    new_cost = d + weight
                    if new_cost < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_cost
                        heapq.heappush(pq, (new_cost, nr, nc))
        return 999 # Effectively infinity

    def get_move(self, state: State) -> Move:
        best_move = None
        best_score = -float('inf')
        
        legal_moves = [(r, c) for r in range(state.board_size) for c in range(state.board_size) if state.board[r][c] == -1]
        
        old_own_cost = self._get_path_cost(state.board, self.player_id)
        old_opp_cost = self._get_path_cost(state.board, self.opponent_id)
        
        for r, c in legal_moves:
            state.board[r][c] = self.player_id
            new_own_cost = self._get_path_cost(state.board, self.player_id)
            new_opp_cost = self._get_path_cost(state.board, self.opponent_id)
            state.board[r][c] = -1
            
            # score = 10 * own_improvement + 4 * opp_worsening
            score = (10 * (old_own_cost - new_own_cost) +
                     4 * (new_opp_cost - old_opp_cost) +
                     random.uniform(0, 0.1))
                     
            if score > best_score:
                best_score = score
                best_move = (r, c)
        
        return Move(player=self.player_id, position=list(best_move))

if __name__ == "__main__":
    DijkstraAgent().start()
