import pytest
import torch
from streamernet.models.unet import ConvBlock

class TestConvBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.in_channels = 3
        self.out_channels = 64
        self.kernel_size = 3
        self.activation = 'relu'
        self.padding_type = ['same']
        self.batch_size = 2
        self.height = 32
        self.width = 32

        self.conv_block = ConvBlock(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.activation,
            self.padding_type
        )

        self.dummy_input = torch.rand(self.batch_size, self.in_channels, self.height, self.width)

    def test_initialization(self):
        assert self.conv_block.conv1.in_channels == self.in_channels
        assert self.conv_block.conv1.out_channels == self.out_channels
        assert self.conv_block.conv2.in_channels == self.out_channels
        assert self.conv_block.conv2.out_channels == self.out_channels
        assert self.conv_block.conv1.kernel_size == (self.kernel_size, self.kernel_size)
        assert isinstance(self.conv_block.activation, torch.nn.ReLU)

    def test_forward_pass(self):
        output = self.conv_block(self.dummy_input)
        assert output.shape == (self.batch_size, self.out_channels, self.height, self.width)
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"

    @pytest.mark.parametrize("activation", ['relu', 'gelu'])
    def test_activation(self, activation):
        conv_block = ConvBlock(self.in_channels, self.out_channels, self.kernel_size, activation)
        assert isinstance(conv_block.activation, torch.nn.ReLU if activation == 'relu' else torch.nn.GELU)

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        conv_block = self.conv_block.to(device)
        x = self.dummy_input.to(device)
        output = conv_block(x)
        assert output.device.type == device
