import torch
import torch.nn as nn
from streamernet.operators import HankelTransform
from streamernet.interpolation import BilinearInterpolation

class CylindricallySymmetricSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, nr, nz, lr, lz, R, n_alpha):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        dz = lz / nz
        dr = lr / nr
        z = torch.arange(dz/2., lz, dz)
        r = torch.arange(dr/2., lr, dr)
        self.HT = HankelTransform(n_alpha, R)
            
        self.interp_f = BilinearInterpolation(self.HT.r, z, r, z)
        self.interp_b = BilinearInterpolation(r, z, self.HT.r, z)
        
    def to(self, tensor):
        super().to(tensor.device)
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(tensor.device))
        self.interp_f.to(tensor.device)
        self.interp_b.to(tensor.device)
        self.HT.to(tensor.device)
        return self

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, r, z ), (in_channel, out_channel, r, z) -> (batch, out_channel, r, z)
        return torch.einsum("birz,iorz->borz", input, weights)

    def forward(self, x):
        batchsize, c, nr, nz = x.shape
        
        # Hankel Transform
        x_ht = self.HT.forward(self.interp_f(x))
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x_ht)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
  
        #Return to physical space (z)
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        # Return to physical space (r)
        x = self.HT.inverse(x)

        # Interpolate back to original grid
        x = self.interp_b(x)

        return x