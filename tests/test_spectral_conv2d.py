import unittest
import torch
from streamernet.layers import SpectralConv2d

class TestSpectralConv2d(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 6
        self.modes1 = 4
        self.modes2 = 4
        self.batch_size = 2
        self.nx = 8
        self.ny = 8
        self.model = SpectralConv2d(self.in_channels, self.out_channels, self.modes1, self.modes2)

    def test_initialization(self):
        self.assertEqual(self.model.in_channels, self.in_channels)
        self.assertEqual(self.model.out_channels, self.out_channels)
        self.assertEqual(self.model.modes1, self.modes1)
        self.assertEqual(self.model.modes2, self.modes2)
        self.assertEqual(self.model.weights1.shape, (self.in_channels, self.out_channels, self.modes1, self.modes2))
        self.assertEqual(self.model.weights2.shape, (self.in_channels, self.out_channels, self.modes1, self.modes2))

    def test_forward(self):
        x = torch.rand(self.batch_size, self.in_channels, self.nx, self.ny, dtype=torch.float32)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_channels, self.nx, self.ny))

if __name__ == '__main__':
    unittest.main()