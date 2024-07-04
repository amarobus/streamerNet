import pytest
import torch
import torch.nn.functional as F
import numpy as np

from streamernet.models import FNO1d

class TestFNO1d:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.t_input = 1
        self.step = 1 # Right now only step=1 is supported
        self.input_features = ['sigma_z', 'phi_z']
        self.output_features = ['sigma_z']
        self.step = 1
        self.in_channels = self.t_input * len(self.input_features)
        self.out_channels = self.step * len(self.output_features)
        self.modes = 4
        self.width = 20
        self.depth = 4
        self.mlp_ratio = 4
        self.activation = 'relu'
        self.batch_size = 2
        self.size_x = 16
        self.model = FNO1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=self.modes,
            width=self.width,
            depth=self.depth,
            mlp_ratio=self.mlp_ratio,
            activation=self.activation
        )

    def test_initialization(self):
        assert self.model.modes == self.modes
        assert self.model.width == self.width
        assert self.model.depth == self.depth
        assert len(self.model.conv) == self.depth
        assert len(self.model.w) == self.depth
        assert self.model.fc0.in_features == self.in_channels + 1
        assert self.model.fc0.out_features == self.width
        assert self.model.fc2.in_features == self.mlp_ratio * self.width
        assert self.model.fc2.out_features == self.out_channels

    def test_forward(self):
        x = torch.rand(self.batch_size, self.size_x, self.in_channels, dtype=torch.float32)
        y = self.model(x)
        assert y.shape == (self.batch_size, self.size_x, self.out_channels)

    def test_activation(self):
        self.model.act_fn = F.gelu
        x = torch.rand(self.batch_size, self.size_x, self.in_channels, dtype=torch.float32)
        y = self.model(x)
        assert y.shape == (self.batch_size, self.size_x, self.out_channels)

    def test_forward_4d_input(self):
        x = torch.rand(self.batch_size, self.size_x, len(self.input_features), self.t_input, dtype=torch.float32)
        y = self.model(x)
        assert y.shape == (self.batch_size, self.size_x, len(self.output_features), self.step)

    def test_get_grid(self):
        grid = self.model.get_grid((self.batch_size, self.size_x), torch.device('cpu'))
        assert grid.shape == (self.batch_size, self.size_x, 1)
        assert torch.allclose(grid[0, :, 0], torch.linspace(0, 1, self.size_x))