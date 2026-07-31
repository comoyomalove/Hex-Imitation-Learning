import torch
from hex_ml.training.model import TinyHexPolicyNet


def test_tiny_policy_net_output_shape():
    model = TinyHexPolicyNet(board_size=3)
    x = torch.zeros((2, 4, 3, 3), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (2, 9)
