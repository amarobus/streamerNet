import torch
from torch import nn
import numpy as np

from streamernet.layers import ConvBlock, UpConvBlock, DownConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, width, filters, kernel_size, activation='relu', padding_type=['valid'], downsampling='max', upsampling='nearest'):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        blocks = len(filters) - 1

        # Encoder
        for block in range(blocks):
            self.encoder.append(DownConvBlock(width if block == 0 else filters[block-1], filters[block], kernel_size, activation, padding_type, downsampling))

        # Bottleneck
        self.bottleneck = ConvBlock(filters[-2], filters[-1], kernel_size, activation, padding_type)

        # Decoder
        for block in reversed(range(blocks)):
            self.decoder.append(UpConvBlock(filters[block+1], filters[block], kernel_size, activation, padding_type, upsampling))
            

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        # Channel dimension first
        x = x.permute(0, 3, 1, 2)
        # Encoder
        for i, encoder in enumerate(self.encoder):
            x, x_ = encoder(x)
            skip_connections.append(x_)

        x = self.bottleneck(x)

        # Decoder
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, skip_connections[-(i+1)])

        x = self.final_conv(x)

        x = x.permute(0, 2, 3, 1)

        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    


    