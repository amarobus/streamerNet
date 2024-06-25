import torch
from scipy import special as sp
import numpy as np

class HankelTransform:
    def __init__(self, n, R):
        self.R = R
        zeros = sp.jn_zeros(0, n+1)
        self.alpha = zeros[:-1]
        # 2 * np.pi * R * V = S
        self.S = zeros[-1]
        self.V = self.S / (2 * np.pi * self.R)
        self.r = self.alpha * self.R / self.S
        self.k = self.alpha / self.R

        j0 = sp.j0
        j1 = sp.j1
        self.T = 2 * j0(self.alpha[...,None] * self.alpha[None,...] / self.S) / (np.abs(j1(self.alpha[...,None])) * np.abs(j1(self.alpha[None,...])) * self.S)
        self.jr = np.abs(j1(self.alpha)) / self.R
        self.jv = np.abs(j1(self.alpha)) / self.V
        self.w = self.S / self.R

        self.jr = self.jr[None,None,...,None]
        self.jv = self.jv[None,None,...,None]
        
        # By default, scipy returns float64, so we need to cast to float32
        # Default dtype for torch is float32
        self.T = torch.tensor(self.T, dtype=torch.float)
        self.jr = torch.tensor(self.jr, dtype=torch.float)
        self.jv = torch.tensor(self.jv, dtype=torch.float)
        self.w = torch.tensor(self.w, dtype=torch.float)
        self.r = torch.tensor(self.r, dtype=torch.float)
        self.k = torch.tensor(self.k, dtype=torch.float)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float)
        self.S = torch.tensor(self.S, dtype=torch.float)
        self.V = torch.tensor(self.V, dtype=torch.float)
        self.R = torch.tensor(self.R, dtype=torch.float)

    def to(self, tensor):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(tensor))
        return self

    def forward(self, f):
        # Move tensors to the appropriate device and cast them to the correct type
        F1 = f / self.jr
        F2 = torch.einsum('mk,bckz->bcmz', self.T, F1)
        f2 = F2 * self.jv
        return f2
    
    def inverse(self, f):
        # Move tensors to the appropriate device and cast them to the correct type
        F2 = f / self.jv
        F1 = torch.einsum('mk,bcmz->bckz', self.T, F2)
        f1 = F1 * self.jr
        return f1