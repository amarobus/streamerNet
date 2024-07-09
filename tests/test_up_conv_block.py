import pytest
import torch
from streamernet.layers import UpConvBlock

class TestUpConvBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.in_channels = 128
        self.out_channels = 64
        self.kernel_size = 3
        self.activation = 'relu'
        self.padding_type = ['same']
        self.upsampling = 'nearest'
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, self.in_channels, 16, 16)
        self.skip_tensor = torch.randn(self.batch_size, self.out_channels, 32, 32)

        self.up_conv_block = UpConvBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            activation=self.activation,
            padding_type=self.padding_type,
            upsampling=self.upsampling
        )

    def test_forward_pass(self):
        output = self.up_conv_block(self.input_tensor, self.skip_tensor)
        assert output.shape == (self.batch_size, self.out_channels, 32, 32)