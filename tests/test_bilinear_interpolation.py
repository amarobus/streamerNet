import pytest
import torch
from streamernet.interpolation import BilinearInterpolation

@pytest.fixture
def sample_data():
    x = torch.tensor([1.5, 2.5])
    y = torch.tensor([1.5, 2.5])
    x_values = torch.tensor([1.0, 2.0, 3.0])
    y_values = torch.tensor([1.0, 2.0, 3.0])
    z_values = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]).unsqueeze(0).unsqueeze(1)
    return x, y, x_values, y_values, z_values


def test_bilinear_interpolation_init(sample_data):
    x, y, x_values, y_values, _ = sample_data
    interpolator = BilinearInterpolation(x, y, x_values, y_values)
    
    assert torch.allclose(interpolator.wx1, torch.tensor([[[[0.5, 0.5]]]]))
    assert torch.allclose(interpolator.wx2, torch.tensor([[[[0.5, 0.5]]]]))
    assert torch.allclose(interpolator.wy1, torch.tensor([[[[0.5, 0.5]]]]))
    assert torch.allclose(interpolator.wy2, torch.tensor([[[[0.5, 0.5]]]]))
    assert torch.allclose(interpolator.x_idx, torch.tensor([[0, 1]]).unsqueeze(-1))
    assert torch.allclose(interpolator.y_idx, torch.tensor([[0, 1]]).unsqueeze(0))
    
    
def test_bilinear_interpolation_call(sample_data):
    x, y, x_values, y_values, z_values = sample_data
    interpolator = BilinearInterpolation(x, y, x_values, y_values)
    
    result = interpolator(z_values)
    # The expected result is an average of the 4 nearest points for the sample data used
    expected = torch.tensor([[3.0, 4.0], [6.0, 7.0]]).unsqueeze(0).unsqueeze(1)

    # Check if the result has the expected shape
    assert result.shape == torch.Size([1, 1, 2, 2])
    
    assert torch.allclose(result, expected)


def test_bilinear_interpolation_edge_cases():
    x = torch.tensor([0.5, 3.5])
    y = torch.tensor([0.5, 3.5])
    x_values = torch.tensor([1.0, 2.0, 3.0])
    y_values = torch.tensor([1.0, 2.0, 3.0])
    z_values = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]).unsqueeze(0).unsqueeze(1)

    interpolator = BilinearInterpolation(x, y, x_values, y_values)
    result = interpolator(z_values)

    # Check if the result has the expected shape
    assert result.shape == torch.Size([1, 1, 2, 2])
    
    # Check interpolation for the point (0.5, 0.5)
    # This should be equal to the value at (1.0, 1.0) as it's the nearest point
    assert torch.isclose(result[0, 0, 0, 0], z_values[0, 0, 0, 0])
    
    # Check interpolation for the point (3.5, 3.5)
    # This should be equal to the value at (3.0, 3.0) as it's the nearest point
    assert torch.isclose(result[0, 0, 1, 1], z_values[0, 0, 2, 2])
    
    # Check interpolation for the point (0.5, 3.5)
    # This should be equal to the value at (1.0, 3.0) as it's the nearest point
    assert torch.isclose(result[0, 0, 0, 1], z_values[0, 0, 0, 2])
    
    # Check interpolation for the point (3.5, 0.5)
    # This should be equal to the value at (3.0, 1.0) as it's the nearest point
    assert torch.isclose(result[0, 0, 1, 0], z_values[0, 0, 2, 0])
    
    # Check if indices are calculated correctly for edge cases
    assert torch.allclose(interpolator.x_idx, torch.tensor([[0, 1]]).unsqueeze(-1))
    assert torch.allclose(interpolator.y_idx, torch.tensor([[0, 1]]).unsqueeze(0))

    # Check if the weights are calculated correctly for edge cases
    assert torch.allclose(interpolator.wx1, torch.tensor([1.0, 0.0]).view(1, 1, -1, 1))
    assert torch.allclose(interpolator.wx2, torch.tensor([0.0, 1.0]).view(1, 1, -1, 1))
    assert torch.allclose(interpolator.wy1, torch.tensor([1.0, 0.0]).view(1, 1, 1, -1))
    assert torch.allclose(interpolator.wy2, torch.tensor([0.0, 1.0]).view(1, 1, 1, -1))