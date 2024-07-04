import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from streamernet.layers import SpectralConv1d

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width=20, depth=4, mlp_ratio=4, activation='relu', padding=None):
        super().__init__()

        """
        The network contains #depth fourier layers.
        1. Lift the input to the desire channel dimension with self.fc0.
        2. #depth layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space with self.fc1 and self.fc2 .

        input shape: (batchsize, size_x, size_y, in_channels)
        output shape: (batchsize, size_x, size_y, out_channels)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        self.padding = padding # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_channels + 1, self.width) # x grid locations are concatenated to the input channels

        self.conv = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.depth)])
        self.w = nn.ModuleList([nn.Conv1d(self.width, self.width, kernel_size=1, padding=0) for _ in range(self.depth)])

        self.fc1 = nn.Linear(self.width, self.mlp_ratio*self.width)
        self.fc2 = nn.Linear(self.mlp_ratio*self.width, self.out_channels)

        if self.activation == 'relu':
            self.act_fn = F.relu
        elif self.activation == 'gelu':
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        # Number of dimensions
        n_dims = len(x.shape)
        if n_dims == 4:
            # Reshape input
            # Input is (batchsize, size_x, num_features, num_timesteps)
            # We want to reshape it to (batchsize, size_x, num_features*num_timesteps)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Pad last dim (for non-periodic boundary conditilons)
        # if self.padding:
        #     x = F.pad(x, (0, self.padding))

        for conv, w in zip(self.conv[:-1], self.w[:-1]): # type: ignore
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = self.act_fn(x)

        x1 = self.conv[-1](x)
        x2 = self.w[-1](x)
        x = x1 + x2

        # Remove padding (for non-periodic boundary conditions)
        # if self.padding:
        #     x = x[...,:-self.padding]

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)

        # Reshape output to (batchsize, size_x, num_features, num_timesteps)
        if n_dims == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)