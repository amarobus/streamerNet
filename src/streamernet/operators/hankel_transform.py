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
        
        # Convert to torch tensors        
        self.T = torch.tensor(self.T)
        self.jr = torch.tensor(self.jr)
        self.jv = torch.tensor(self.jv)

        self.S = torch.tensor(self.S)
        self.V = torch.tensor(self.V)
        self.r = torch.tensor(self.r)
        self.k = torch.tensor(self.k)
        self.w = torch.tensor(self.w)
        self.alpha = torch.tensor(self.alpha)

        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        try:
            self.to(device)
        except Exception as e:
            print(f"Failed to move to device {device}: {e}")

    def to(self, device):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
        return self

    def forward(self, f):
        F1 = f / self.jr
        F2 = torch.einsum('mk,bckz->bcmz', self.T, F1)
        f2 = F2 * self.jv
        return f2
    
    def inverse(self, f):
        F2 = f / self.jv
        F1 = torch.einsum('mk,bcmz->bckz', self.T, F2)
        f1 = F1 * self.jr
        return f1