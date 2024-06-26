import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from streamernet.layers import CylindricallySymmetricSpectralConv2d

class CSFNO2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        width=20,
        depth=4,
        mlp_ratio=4,
        activation='relu', 
        padding=None, 
        nr=128, 
        nz=128, 
        lr=1e-2, 
        lz=1e-2, 
        R=1e-2, 
        n_alpha=128
    ):
        super().__init__()

        """
        The overall network. It contains #depth fourier layers.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. #depth layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape: (batchsize, size_x, size_y, in_channels)
        output shape: (batchsize, size_x, size_y, out_channels)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.padding = padding # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv = nn.ModuleList([CylindricallySymmetricSpectralConv2d(self.width, self.width, self.modes1, self.modes2, nr, nz, lr, lz, R, n_alpha) for _ in range(depth)])
        self.w = nn.ModuleList([nn.Conv2d(self.width, self.width, kernel_size=1, padding=0) for _ in range(depth)])

        self.fc1 = nn.Linear(self.width, mlp_ratio*self.width)
        self.fc2 = nn.Linear(mlp_ratio*self.width, out_channels)

        if activation == 'relu':
            self.act = F.relu
        elif activation == 'gelu':
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def to(self, device):
        super().to(device)
        for layer in self.conv:
            layer.to(device)
        return self

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (b, c, r, z)
        
        # Pad x, y, z dimensions if domain is non-periodic
        if self.padding:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        for conv, w in zip(self.conv[:-1], self.w[:-1]):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = self.act(x)
        
        x1 = self.conv[-1](x)
        x2 = self.w[-1](x)
        x = x1 + x2

        # Remove padding if domain is non-periodic
        if self.padding:
            x = x[..., :-self.padding,:-self.padding]

        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)