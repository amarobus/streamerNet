import pytest
import torch
from streamernet.layers import CylindricallySymmetricSpectralConv2d

class TestCylindricallySymmetricSpectralConv2d:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.batch_size = 2 # type: ignore
        self.in_channels = 3 # type: ignore
        self.out_channels = 3 # type: ignore
        self.modes1 = 8 # type: ignore
        self.modes2 = 8 # type: ignore
        self.nr = 16 # type: ignore
        self.nz = 16 # type: ignore
        self.lr = 1.0 # type: ignore
        self.lz = 1.0 # type: ignore
        self.R = 1.0 # type: ignore
        self.n_alpha = 16 # type: ignore

        self.dummy_input = torch.rand(self.batch_size, self.in_channels, self.nr, self.nz) # type: ignore
        self.layer = CylindricallySymmetricSpectralConv2d(# type: ignore
            self.in_channels, 
            self.out_channels, 
            self.modes1, 
            self.modes2, 
            self.nr, 
            self.nz, 
            self.lr, 
            self.lz, 
            self.R,
            self.n_alpha
        )

    def test_initialization(self):
        assert self.layer.in_channels == self.in_channels
        assert self.layer.out_channels == self.out_channels
        assert self.layer.modes1 == self.modes1
        assert self.layer.modes2 == self.modes2
        assert self.layer.weights1.shape == (self.in_channels, self.out_channels, self.modes1, self.modes2)

    def test_forward_pass(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dummy_input = self.dummy_input.to(device=device)
        self.layer = self.layer.to(self.dummy_input)
        output = self.layer(self.dummy_input)
        assert output.shape == self.dummy_input.shape
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"