import pytest
import torch
import torch.nn as nn
from torch.testing import assert_allclose


# Import the TensorTrainLayer
class TensorTrainLayer(nn.Module):
    def __init__(self, in_modes, out_modes, ranks):
        super(TensorTrainLayer, self).__init__()
        assert len(in_modes) == len(out_modes) == len(ranks) + 1  # Check dimensions
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.ranks = ranks
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(in_modes[i], out_modes[i], ranks[i], ranks[i + 1])
                )
                for i in range(len(ranks))
            ]
        )

    def forward(self, x):
        tensor_train = x.view(-1, self.in_modes[0], self.ranks[0])
        for i in range(len(self.weights)):
            tensor_train = torch.einsum("bir,iorj->borj", tensor_train, self.weights[i])
        return tensor_train.view(-1, self.out_modes[-1])


# === Test Cases ===


@pytest.mark.parametrize(
    "in_modes, out_modes, ranks, batch_size",
    [
        ([2, 3], [4, 5], [1, 2], 10),
        ([3, 2], [2, 3], [1, 1], 5),
        ([4, 2, 3], [3, 2, 4], [1, 2, 1], 8),
    ],
)
def test_tensor_train_random_input(in_modes, out_modes, ranks, batch_size):
    """Test with random input tensors of various sizes and shapes."""
    layer = TensorTrainLayer(in_modes, out_modes, ranks)
    x = torch.randn(batch_size, in_modes[0], ranks[0])
    output = layer(x)

    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == out_modes[-1], "Output shape mismatch"


def test_tensor_train_known_case():
    """Test with a known input-output pair for correctness."""
    in_modes = [2, 2]
    out_modes = [2, 2]
    ranks = [1, 1]

    layer = TensorTrainLayer(in_modes, out_modes, ranks)
    layer.weights[0].data = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    layer.weights[1].data = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])

    x = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])  # Identity input
    output = layer(x)

    expected_output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert_allclose(
        output, expected_output, atol=1e-5
    ), "Incorrect tensor contraction result!"


@pytest.mark.parametrize(
    "in_modes, out_modes, ranks",
    [
        ([2, 3, 4], [4, 3, 2], [1, 2, 1]),  # Different ranks
        ([3, 2], [2, 3], [1, 1]),  # Rank-1 factorization
        ([4, 2, 3], [3, 2, 4], [2, 1, 2]),  # Varying ranks
    ],
)
def test_tensor_train_mode_combinations(in_modes, out_modes, ranks):
    """Test with different combinations of input modes, output modes, and ranks."""
    layer = TensorTrainLayer(in_modes, out_modes, ranks)
    batch_size = 6
    x = torch.randn(batch_size, in_modes[0], ranks[0])

    output = layer(x)
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == out_modes[-1], "Output mode mismatch"


if __name__ == "__main__":
    pytest.main()
  

#test
