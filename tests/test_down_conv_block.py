import pytest
import torch
from streamernet.layers import DownConvBlock

class TestDownConvBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.in_channels = 64
        self.out_channels = 128
        self.kernel_size = 3
        self.activation = 'relu'
        self.padding_type = ['same']
        self.downsampling = 'max'
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, self.in_channels, 32, 32)

        self.down_conv_block = DownConvBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
            padding_type=self.padding_type,
            downsampling=self.downsampling
        )

    def test_forward_pass(self):
        output, skip = self.down_conv_block(self.input_tensor)
        assert output.shape == (self.batch_size, self.out_channels, 16, 16)
        assert skip.shape == (self.batch_size, self.out_channels, 32, 32)