import pytest
import torch
from streamernet.layers import SpectralConv1d

class TestSpectralConv1d:
    @pytest.fixture
    def conv_params(self):
        return {
            'in_channels': 3,
            'out_channels': 6,
            'modes': 4,
            'batch_size': 2,
            'nx': 16
        }

    @pytest.fixture
    def model(self, conv_params):
        return SpectralConv1d(conv_params['in_channels'], conv_params['out_channels'], conv_params['modes'])

    def test_initialization(self, model, conv_params):
        assert model.in_channels == conv_params['in_channels']
        assert model.out_channels == conv_params['out_channels']
        assert model.modes == conv_params['modes']
        assert model.weights.shape == (conv_params['in_channels'], conv_params['out_channels'], conv_params['modes'])

    def test_forward(self, model, conv_params):
        x = torch.rand(conv_params['batch_size'], conv_params['in_channels'], conv_params['nx'], dtype=torch.float32)
        y = model(x)
        assert y.shape == (conv_params['batch_size'], conv_params['out_channels'], conv_params['nx'])

    @pytest.mark.parametrize("batch_size,nx", [(1, 32), (4, 64), (8, 128)])
    def test_different_input_sizes(self, model, conv_params, batch_size, nx):
        x = torch.rand(batch_size, conv_params['in_channels'], nx, dtype=torch.float32)
        y = model(x)
        assert y.shape == (batch_size, conv_params['out_channels'], nx)

    def test_complex_multiplication(self, model):
        input_tensor = torch.randn(2, 3, 4, dtype=torch.cfloat)
        weights = torch.randn(3, 6, 4, dtype=torch.cfloat)
        result = model.compl_mul2d(input_tensor, weights)
        assert result.shape == (2, 6, 4)
        assert result.dtype == torch.cfloat