import pytest
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from streamernet.datasets import StreamerDataset1D

class TestStreamerDataset1D:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = {
            'data': {
                'directory': '~/projects/data/1D',
                'train_filename': 'sigma_phi_line_data.npz',
                'valid_filename': 'sigma_phi_line_data.npz'
            },
            'input': {
                'features': ['sigma_z', 'phi_z'],
                't_input': 1,
                'T': 9
            },
            'output': {
                'features': ['sigma_z']
            }
        }
        
        self.num_runs = 1000
        self.samples_per_run = 20
        self.total_samples = self.num_runs * self.samples_per_run
        
        # Create mock data
        self.mock_data = {
            'sigma_z': np.random.rand(self.total_samples, 128),
            'phi_z': np.random.rand(self.total_samples, 128),
            'run_index': np.repeat(np.arange(self.num_runs), self.samples_per_run),
            'cycles': np.tile(np.arange(self.samples_per_run), self.num_runs)
        }
        
        self.mock_npz = MagicMock()
        # Mock the behavior of accessing datasets with __getitem__
        self.mock_npz.__getitem__.side_effect = lambda key: self.mock_data[key]
        
        with patch('numpy.load', return_value=self.mock_npz):
            data_dir = os.path.expanduser(self.config['data']['directory'])
            file_path = os.path.join(data_dir, self.config['data']['train_filename'])
            input_features = self.config['input']['features']
            output_features = self.config['output']['features']
            t_input = self.config['input']['t_input']
            T = self.config['input']['T']
            
            self.dataset = StreamerDataset1D(file_path, input_features, output_features, t_input, T)

    def test_dataset_initialization(self):
        assert isinstance(self.dataset, StreamerDataset1D)

    def test_dataset_length(self):
        assert len(self.dataset) > 0

    def test_input_target_tensor_shapes(self):
        input_tensor, target_tensor = self.dataset[0]
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        
        expected_input_shape = (self.mock_data['sigma_z'].shape[1], len(self.config['input']['features']), self.config['input']['t_input'])
        expected_target_shape = (self.mock_data['sigma_z'].shape[1], len(self.config['output']['features']), self.config['input']['T'])
        
        assert input_tensor.shape == expected_input_shape
        assert target_tensor.shape == expected_target_shape

    def test_data_normalization(self):
        input_tensor, target_tensor = self.dataset[0]
        assert torch.all(input_tensor >= 0) and torch.all(input_tensor <= 1)
        assert torch.all(target_tensor >= 0) and torch.all(target_tensor <= 1)

    def test_data_loading_for_partitions(self):
        for partition in ['train', 'valid', 'test']:
            with patch('numpy.load', return_value=self.mock_npz):
                partition_dataset = StreamerDataset1D(
                    self.dataset.file_path, 
                    self.dataset.input_features, 
                    self.dataset.output_features, 
                    self.dataset.t_input, 
                    self.dataset.T, 
                    partition=partition
                )
            assert len(partition_dataset) > 0

    def test_error_handling_invalid_file_path(self):
        with pytest.raises(IOError):
            with patch('numpy.load', side_effect=IOError):
                StreamerDataset1D('invalid/file/path.npz', self.dataset.input_features, self.dataset.output_features, self.dataset.t_input, self.dataset.T)

    def test_multiple_output_features(self):
        with patch('numpy.load', return_value=self.mock_npz):
            data_dir = os.path.expanduser(self.config['data']['directory'])
            file_path = os.path.join(data_dir, self.config['data']['train_filename'])
            input_features = self.config['input']['features']
            output_features = ['sigma_z', 'phi_z']  # Both features as output
            t_input = self.config['input']['t_input']
            T = self.config['input']['T']
            
            multi_output_dataset = StreamerDataset1D(file_path, input_features, output_features, t_input, T)

        # Check if the dataset was created successfully
        assert isinstance(multi_output_dataset, StreamerDataset1D)

        # Check the shapes of input and target tensors
        input_tensor, target_tensor = multi_output_dataset[0]
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)

        expected_input_shape = (self.mock_data['sigma_z'].shape[1], len(input_features), t_input)
        expected_target_shape = (self.mock_data['sigma_z'].shape[1], len(output_features), T)

        assert input_tensor.shape == expected_input_shape
        assert target_tensor.shape == expected_target_shape

        # Check if the target tensor contains data for both output features
        assert target_tensor.shape[1] == 2  # Should have 2 channels for sigma_z and phi_z


if __name__ == '__main__':
    pytest.main([__file__])