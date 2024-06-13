import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

class StreamerDataset:
    def __init__(self, file_path, input_feature, t_input=10, T=10, partition='train', min=None, max=None):
        """
        Initialize the FlashDataset.

        Parameters:
        file_path (str): Path to the HDF5 file.
        input_feature (str): Input feature to load.
        t_input (int): Number of input time steps.
        T (int): Number of target time steps.
        partition (str): Partition of the data to load. Can be 'train' or 'valid'.
        """
        np.random.seed(0)
        self.file_path = os.path.expanduser(file_path)
        self.input_feature = input_feature
        self.t_input = t_input
        self.T = T
        self.partition = partition
        self.min = min # Minimum value of the data
        self.max = max # Maximum value of the data
        self.prepare_dataset()
        
    def __load_data(self):
        """
        Load data from the HDF5 file.
        
        Returns:
        data (numpy.ndarray): Data loaded from the file.
        output_index (numpy.ndarray): Output index loaded from the file.
        run_number (numpy.ndarray): Run number loaded from the file.
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                data = np.float32(f[self.input_feature])
                output_index = np.int32(f['output_index'])
                run_number = np.int32(f['run_number'])
        except Exception as e:
            raise IOError(f"Error loading data from {self.file_path}: {e}")

        return data, output_index, run_number
        
    def preprocess_data(self, data, output_index, run_number):
        """
        Preprocess data by refactoring, transforming, normalizing, and partitioning.
        
        Parameters:
        data (numpy.ndarray): Data to preprocess.
        output_index (numpy.ndarray): Output index.
        run_number (numpy.ndarray): Run number.
        
        Returns:
        input_tensor (torch.Tensor): Input tensor.
        target_tensor (torch.Tensor): Target tensor.
        
        """
        t_input = self.t_input
        T = self.T
        total_steps = t_input + T
        diff = np.diff(run_number)
        diff = np.where(diff !=0, 1, 0)

        # Indices that indicate where the array will be split
        idx = [i for i in range(len(diff)) if diff[i] == 1 and output_index[i] >= total_steps]
 
        # Split data into chunks of total_steps
        chunks = [data[i - total_steps + 1:i + 1] for i in idx]
        
        # Handle the last chunk
        if len(data) - idx[-1] >= total_steps:
            chunks.append(data[idx[-1]:idx[-1] + total_steps])
            
        data = np.stack(chunks, axis=0)

        # Shuffle
        np.random.shuffle(data)

        data = self.__get_partition_data(data, self.partition)

        # Log transformation and normalization
        data = np.log(data+1)
        
        # Set the minimum and maximum values
        self.__set_stats(data)
        # Normalize the data
        data = self.__range_normalizer(data)

        # Transpose to bring time dimension last
        data = np.transpose(data, (0, 2, 3, 1))

        # Split into input and target
        input_data = data[..., :t_input]
        target_data = data[..., t_input:t_input + T]
        
        # Convert to PyTorch tensors
        self.input_tensor = torch.tensor(input_data) # type: ignore
        self.target_tensor = torch.tensor(target_data) # type: ignore

    def __get_partition_data(self, data, partition):
        """
        Split the data into train, validation, and test partitions.
        
        Parameters:
        data (numpy.ndarray): Data to split.
        partition (str): Partition of the data to split. Can be 'train', 'valid', or 'test'.
        
        Returns:
        data (numpy.ndarray): Data for the specified partition.
        """
        if partition == 'train':
            return data[:int(0.8*len(data))]
        elif partition == 'valid':
            return data[int(0.8*len(data)):int(0.9*len(data))]
        elif partition == 'test':
            return data[int(0.9*len(data)):]
        
    def __range_normalizer(self, data):
        """
        Normalize the data to the range [0, 1].

        Parameters:
        data (numpy.ndarray): Data to normalize.

        Returns:
        numpy.ndarray: Normalized data.
        """
        if self.min is not None and self.max is not None:
            return (data - self.min) / (self.max - self.min)
        
    def __set_stats(self, data):
        """
        Set the minimum and maximum values of the data.
        
        Parameters:
        data (numpy.ndarray): Data to set the statistics.
        """
        if self.min is None and self.max is None:
            self.min = data.min()
            self.max = data.max()
        
    def prepare_dataset(self):
        """
        Prepare the dataset for training.
        """
        data, output_index, run_number = self.__load_data()
        self.preprocess_data(data, output_index, run_number)
        
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
        int: Number of samples in the dataset.
    
        """
        return len(self.input_tensor)
    
    def __getitem__(self, idx):
        """
        Get the input and target tensors for the given index.
        
        Parameters:
        idx (int): Index of the sample.
        
        Returns:
        input_tensor (torch.Tensor): Input tensor.
        target_tensor (torch.Tensor): Target tensor.
        """
        return self.input_tensor[idx], self.target_tensor[idx]
        
        
if __name__ == "__main__":
    file_path = '~/projects/data/dataset_128.h5'
    input_feature = 'e'
    dataset = StreamerDataset(file_path, input_feature, t_input=2, T=2, partition='train')
    print(dataset.input_tensor.shape, dataset.target_tensor.shape)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (input_data, target_data) in enumerate(data_loader):
        print(input_data.shape, target_data.shape)
        if i == 0:
            break