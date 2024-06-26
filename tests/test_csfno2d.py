import pytest
import torch
from streamernet.models import CSFNO2d

class TestCSFNO2d:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.in_channels = 1 # type: ignore
        self.out_channels = 1 # type: ignore
        self.modes1 = 12 # type: ignore
        self.modes2 = 12 # type: ignore
        self.width = 32 # type: ignore
        self.depth = 4 # type: ignore
        self.batch_size = 2 # type: ignore
        self.size_x = 64 # type: ignore
        self.size_y = 64 # type: ignore
        self.mlp_ratio = 4 # type: ignore
        self.activation = 'gelu' # type: ignore
        self.padding = None # type: ignore
        self.nr = 64 # type: ignore
        self.nz = 64 # type: ignore
        self.lr = 1e-2 # type: ignore
        self.lz = 1e-2 # type: ignore
        self.R = 1e-2 # type: ignore
        self.n_alpha = 64 # type: ignore

        self.model = CSFNO2d( # type: ignore
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            self.width,
            self.depth,
            self.mlp_ratio,
            self.activation,
            self.padding,
            self.nr,
            self.nz,
            self.lr,
            self.lz,
            self.R,
            self.n_alpha
        )

        self.dummy_input = torch.randn(self.batch_size, self.size_x, self.size_y, self.in_channels) # type: ignore

    def test_initialization(self):
        assert isinstance(self.model, CSFNO2d)
        assert self.model.modes1 == self.modes1
        assert self.model.modes2 == self.modes2
        assert self.model.width == self.width
        assert self.model.depth == self.depth
        assert self.model.padding == self.padding
        assert len(self.model.conv) == self.depth
        assert len(self.model.w) == self.depth

    def test_forward_pass(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dummy_input = self.dummy_input.to(device)
        self.model = self.model.to(device)
        output = self.model(self.dummy_input)
        assert output.shape == (self.batch_size, self.size_x, self.size_y, self.out_channels)
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"

    def test_get_grid(self):
        shape = (self.batch_size, self.size_x, self.size_y, 2)
        grid = self.model.get_grid(shape, device='cpu')
        assert grid.shape == shape
        assert grid.device.type == 'cpu'

    @pytest.mark.parametrize("activation", ['relu', 'gelu'])
    def test_activation(self, activation):
        model = CSFNO2d(
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
            activation=activation
        )
        assert model.act == (torch.nn.functional.relu if activation == 'relu' else torch.nn.functional.gelu)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unsupported activation function: invalid"):
            CSFNO2d(
                self.in_channels,
                self.out_channels,
                self.modes1,
                self.modes2,
                activation='invalid'
            )

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = self.model.to(device)
        x = self.dummy_input.to(device)
        output = model(x)
        assert output.device.type == device