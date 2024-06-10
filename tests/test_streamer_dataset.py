import pytest
import numpy as np
import h5py
from unittest.mock import patch, MagicMock
from streamernet.datasets.streamer_dataset import StreamerDataset

class TestStreamerDatasetBase:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.file_path = '~/path/to/file.h5' # type: ignore
        self.input_feature = 'e' # type: ignore
        self.t_input = 5 # type: ignore
        self.T = 5 # type: ignore
        self.total_time = self.t_input + self.T # type: ignore
        self.data_shape = (1000, 128, 128) # Total samples, height, width # type: ignore  
        self.num_runs = 50 # type: ignore
        self.samples_per_run = 20 # type: ignore

    @pytest.fixture
    def mock_h5file(self):
        mock_file = MagicMock(spec=h5py.File)
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        
        # Create the mock `output_index` dataset
        output_index = np.tile(np.arange(self.samples_per_run), self.num_runs)

        # Create the base `run_number` dataset
        base_run_number = np.arange(self.num_runs)
        np.random.shuffle(base_run_number)
        run_number = np.repeat(base_run_number, self.samples_per_run)

        # Mock the behavior of accessing datasets with __getitem__
        mock_file.__getitem__.side_effect = lambda key: {
            'e': np.random.rand(*self.data_shape),
            'output_index': output_index,
            'run_number': run_number,
        }[key]

        return mock_file

    @pytest.fixture
    def mock_os_path_expanduser(self):
        with patch('os.path.expanduser') as mock:
            mock.return_value = '/mocked/path/to/file.h5'
            yield mock

    @pytest.fixture
    def dataset(self, mock_h5file, mock_os_path_expanduser):
        with patch('h5py.File', return_value=mock_h5file):
            return StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='train')

class TestStreamerDataset(TestStreamerDatasetBase):
    def test_streamer_dataset_init(self, dataset):
        print(f"Input tensor shape: {dataset.input_tensor.shape}")
        print(f"Target tensor shape: {dataset.target_tensor.shape}")

        assert dataset.file_path == '/mocked/path/to/file.h5'
        assert dataset.input_feature == self.input_feature
        assert dataset.t_input == self.t_input
        assert dataset.T == self.T
        assert dataset.partition == 'train'
        assert dataset.min is not None
        assert dataset.max is not None
        assert dataset.input_tensor is not None
        assert dataset.input_tensor.shape[0] > 0
        assert dataset.target_tensor is not None
        assert dataset.target_tensor.shape[0] > 0

    def test_streamer_dataset_len(self, dataset):
        expected_len = int(np.floor(self.data_shape[0] / (self.samples_per_run/(self.total_time)) / (self.total_time) * 0.8))  # 80% of data for training (hardcoded)
        assert len(dataset) == expected_len

    def test_streamer_dataset_getitem(self, dataset):
        items = dataset[0]
        assert isinstance(items, tuple)
        assert len(items) == 2
        assert items[0].shape == (*self.data_shape[1:], self.t_input)  # input shape
        assert items[1].shape == (*self.data_shape[1:], self.T)  # target shape

    def test_streamer_dataset_different_partitions(self, mock_h5file, mock_os_path_expanduser):
        with patch('h5py.File', return_value=mock_h5file):
            train_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='train')
            train_min, train_max = train_dataset.min, train_dataset.max
            valid_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='valid', min=train_min, max=train_max)
            test_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='test', min=train_min, max=train_max)

        assert len(train_dataset) > 0
        assert len(valid_dataset) > 0
        assert len(test_dataset) > 0
        total_samples = self.data_shape[0] / (self.samples_per_run/(self.total_time)) / (self.total_time)
        assert len(train_dataset) + len(valid_dataset) + len(test_dataset) == int(total_samples*0.8) + int(total_samples*0.1) + int(total_samples*0.1) # Hardcoded 80-10-10 split

    def test_streamer_dataset_min_max_consistency(self, mock_h5file, mock_os_path_expanduser):
        with patch('h5py.File', return_value=mock_h5file):
            train_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='train')
            valid_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='valid', min=train_dataset.min, max=train_dataset.max)
            test_dataset = StreamerDataset(self.file_path, self.input_feature, t_input=self.t_input, T=self.T, partition='test', min=train_dataset.min, max=train_dataset.max)

        assert np.isclose(valid_dataset.min, train_dataset.min)
        assert np.isclose(valid_dataset.max, train_dataset.max)
        assert np.isclose(test_dataset.min, train_dataset.min)
        assert np.isclose(test_dataset.max, train_dataset.max)