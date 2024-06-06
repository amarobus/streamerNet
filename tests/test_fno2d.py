import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from streamernet.layers import SpectralConv2d
from streamernet.models import FNO2d

class TestFNO2d(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 6
        self.modes1 = 4
        self.modes2 = 4
        self.width = 20
        self.depth = 4
        self.mlp_ratio = 4
        self.activation = 'relu'
        self.batch_size = 2
        self.size_x = 16
        self.size_y = 16
        self.model = FNO2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            depth=self.depth,
            mlp_ratio=self.mlp_ratio,
            activation=self.activation
        )

    def test_initialization(self):
        self.assertEqual(self.model.modes1, self.modes1)
        self.assertEqual(self.model.modes2, self.modes2)
        self.assertEqual(self.model.width, self.width)
        self.assertEqual(self.model.depth, self.depth)
        self.assertEqual(len(self.model.conv), self.depth)
        self.assertEqual(len(self.model.w), self.depth)
        self.assertIsInstance(self.model.conv[0], SpectralConv2d)
        self.assertIsInstance(self.model.w[0], nn.Conv2d)
        self.assertEqual(self.model.fc0.in_features, self.in_channels + 2)
        self.assertEqual(self.model.fc0.out_features, self.width)
        self.assertEqual(self.model.fc2.in_features, self.mlp_ratio * self.width)
        self.assertEqual(self.model.fc2.out_features, self.out_channels)

        # Check in_channels and out_channels
        self.assertEqual(self.model.fc0.in_features, self.in_channels + 2)
        self.assertEqual(self.model.fc0.out_features, self.width)
        self.assertEqual(self.model.fc2.in_features, self.mlp_ratio * self.width)
        self.assertEqual(self.model.fc2.out_features, self.out_channels)

    def test_forward(self):
        x = torch.rand(self.batch_size, self.size_x, self.size_y, self.in_channels, dtype=torch.float32)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.size_x, self.size_y, self.out_channels))

    def test_activation(self):
        self.model.act_fn = F.gelu
        x = torch.rand(self.batch_size, self.size_x, self.size_y, self.in_channels, dtype=torch.float32)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.size_x, self.size_y, self.out_channels))

if __name__ == '__main__':
    unittest.main()