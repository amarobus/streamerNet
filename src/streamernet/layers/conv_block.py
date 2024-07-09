import torch.nn as nn

from .pad import Pad

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, padding_type=['same']):
        super().__init__()
        self.padding_type = padding_type
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='valid')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='valid')
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.padding_type = padding_type
        self.pad = Pad(self.padding_type)

    def forward(self, x):
        x = self.pad(x)
        x = self.activation(self.conv1(x))
        x = self.pad(x)
        x = self.activation(self.conv2(x))
        return x
    