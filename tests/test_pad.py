import pytest
import torch
from streamernet.layers import Pad

class TestPad:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.padding_type = ['same']
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 32, 32)

        self.pad_layer = Pad(padding_type=self.padding_type)

    def test_forward_pass(self):
        output = self.pad_layer(self.input_tensor)
        assert output.shape == (self.batch_size, 3, 34, 34)  # 2 pixels padding on each side