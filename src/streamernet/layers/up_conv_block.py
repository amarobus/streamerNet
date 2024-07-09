import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .pad import Pad

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, padding_type=['same'], upsampling='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=upsampling)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='valid')
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, activation, padding_type)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, activation, padding_type)
        self.padding_type = padding_type
        self.pad = Pad(self.padding_type)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv0(x)
        x = torch.cat((skip, x), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x