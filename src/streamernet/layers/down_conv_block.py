from torch import nn

from .conv_block import ConvBlock

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, padding_type=['same'], downsampling='max'):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if downsampling == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, activation, padding_type)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, activation, padding_type)
        
    def forward(self, x):
        x = self.conv1(x)
        x_ = self.conv2(x)
        x = self.pool(x_)
        return x, x_
