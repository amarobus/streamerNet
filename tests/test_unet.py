import pytest
import torch

from streamernet.layers import ConvBlock
from streamernet.models import UNet

class TestUNet:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.in_channels = 1
        self.out_channels = 1
        self.width = 20
        self.filters = [32, 64, 128, 256, 512]
        self.kernel_size = 3
        self.activation = 'relu'
        self.padding_type = ['same']
        self.downsampling = 'max'
        self.upsampling = 'nearest'
        self.batch_size = 2
        self.nx = 64
        self.ny = 64

        self.unet = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            padding_type=self.padding_type,
            downsampling=self.downsampling,
            upsampling=self.upsampling
        )

        self.dummy_input = torch.rand(self.batch_size, self.nx, self.ny, self.in_channels)

    def test_initialization(self):
        assert isinstance(self.unet, UNet)
        assert len(self.unet.encoder) == len(self.filters) - 1
        assert len(self.unet.decoder) == len(self.filters) - 1
        assert isinstance(self.unet.bottleneck, ConvBlock)

    def test_forward_pass(self):
        output = self.unet(self.dummy_input)
        assert output.shape == (self.batch_size, self.nx, self.ny, self.out_channels)
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"

    @pytest.mark.parametrize("activation", ['relu', 'gelu'])
    def test_activation(self, activation):
        unet = UNet(self.in_channels, self.out_channels, self.width, self.filters, self.kernel_size, activation=activation)
        for encoder in unet.encoder:
            assert isinstance(encoder.conv1.activation, torch.nn.ReLU if activation == 'relu' else torch.nn.GELU)

    @pytest.mark.parametrize("downsampling", ['max', 'average'])
    def test_downsampling(self, downsampling):
        unet = UNet(self.in_channels, self.out_channels, self.width, self.filters, self.kernel_size, downsampling=downsampling)
        for encoder in unet.encoder:
            assert isinstance(encoder.pool, torch.nn.MaxPool2d if downsampling == 'max' else torch.nn.AvgPool2d)

    @pytest.mark.parametrize("upsampling", ['nearest', 'bilinear'])
    def test_upsampling(self, upsampling):
        unet = UNet(self.in_channels, self.out_channels, self.width, self.filters, self.kernel_size, upsampling=upsampling)
        for decoder in unet.decoder:
            assert isinstance(decoder.upsample, torch.nn.Upsample)
            assert decoder.upsample.mode == upsampling

    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        unet = self.unet.to(device)
        x = self.dummy_input.to(device)
        output = unet(x)
        assert output.device.type == device

    def test_output_shape(self):
        for nx, ny in [(64, 64), (128, 128), (256, 256)]:
            dummy_input = torch.rand(self.batch_size, nx, ny, self.in_channels)
            output = self.unet(dummy_input)
            assert output.shape == (self.batch_size, nx, ny, self.out_channels)
            
    
    # def test_conv_block(self):
    #     conv_block = ConvBlock(64, 128, self.kernel_size, self.activation, self.padding_type)
    #     x = torch.randn(1, 64, 32, 32)
    #     output = conv_block(x)
    #     assert output.shape == (1, 128, 32, 32)

    # def test_down_conv_block(self):
    #     down_conv_block = DownConvBlock(64, 128, self.kernel_size, self.activation, self.padding_type, self.downsampling)
    #     x = torch.randn(1, 64, 32, 32)
    #     output, skip = down_conv_block(x)
    #     assert output.shape == (1, 128, 16, 16)
    #     assert skip.shape == (1, 128, 32, 32)

    # def test_up_conv_block(self):
    #     up_conv_block = UpConvBlock(128, 64, self.kernel_size, self.activation, self.padding_type, self.upsampling)
    #     x = torch.randn(1, 128, 16, 16)
    #     output = up_conv_block(x)
    #     assert output.shape == (1, 64, 32, 32)
