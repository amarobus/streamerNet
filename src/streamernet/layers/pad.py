import torch.nn.functional as F
from torch import nn

class Pad(nn.Module):
    def __init__(self, padding_type=['neumann', 'dirichlet']):
        super().__init__()
        self.padding_type = padding_type

    def forward(self, x):
        if self.padding_type == ['same']:
            x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        elif self.padding_type == ['neumann', 'dirichlet']:
            # First all Neumann boundary conditions
            x = F.pad(x, (1, 1, 1, 1), mode='replicate')
            # Then Dirichlet boundary conditions
            x[...,-1] = -x[...,-1]
            x[...,0] = -x[...,0]
            # Fill corners with zeros
            # x[...,0,0] = 0
            # x[...,-1,0] = 0
            # x[...,0,-1] = 0
            # x[...,-1,-1] = 0
        else:
            raise NotImplementedError("Padding type not implemented")
        return x