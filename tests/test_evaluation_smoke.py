import torch
from hex_ml.agents.baselines import FirstLegalAgent
from hex_ml.agents.trained_agent import TrainedHexAgent
from hex_ml.evaluation.evaluator import evaluate_trained_agent


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 9), dtype=torch.float32)


def test_evaluation_smoke_runs_without_illegal_moves_or_crashes():
    agent = TrainedHexAgent(ZeroModel())
    result = evaluate_trained_agent(agent, FirstLegalAgent(), games=2, board_size=3)
    assert result.games == 2
    assert result.illegal_moves == 0
    assert result.crashes == 0
