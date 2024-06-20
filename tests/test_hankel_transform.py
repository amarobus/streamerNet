import pytest
import torch
from streamernet.operators.hankel_transform import HankelTransform

class TestHankelTransform:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.hankel_transform = HankelTransform(n=10, R=1.0) # type: ignore
        self.batch, self.channels, self.nr, self.nz = 2, 3, 10, 5 #type: ignore
        self.f = torch.rand(self.batch, self.channels, self.nr, self.nz).to('cuda') #type: ignore

    def test_initialization(self):
        assert isinstance(self.hankel_transform, HankelTransform)
        assert self.hankel_transform.R == 1.0
        assert self.hankel_transform.alpha.shape == (self.nr,)
        assert self.hankel_transform.T.shape == (self.nr, self.nr)

    def test_device_movement(self):
        ht = self.hankel_transform
        if torch.cuda.is_available():
            ht.to('cuda')
            assert ht.alpha.device.type == 'cuda'
        else:
            ht.to('cpu')
            assert ht.alpha.device.type == 'cpu'

    def test_forward_transform(self):
        f2 = self.hankel_transform.forward(self.f)
        assert f2.shape == (self.batch, self.channels, self.nr, self.nz)

    def test_inverse_transform(self):
        f2 = self.hankel_transform.forward(self.f)
        f_reconstructed = self.hankel_transform.inverse(f2).to(self.f.dtype)
        assert f_reconstructed.shape == (self.batch, self.channels, self.nr, self.nz)

    def test_forward_inverse_identity(self):
        f2 = self.hankel_transform.forward(self.f)
        f_reconstructed = self.hankel_transform.inverse(f2).to(self.f.dtype)
        assert torch.allclose(self.f, f_reconstructed, rtol=1e-5, atol=1e-5)

    def test_unitary_transform(self):
        T = self.hankel_transform.T
        identity = torch.eye(T.shape[0], device=T.device, dtype=T.dtype)
        product = torch.matmul(T, T.transpose(-2, -1))
        assert torch.allclose(product, identity, rtol=1e-5, atol=1e-5)