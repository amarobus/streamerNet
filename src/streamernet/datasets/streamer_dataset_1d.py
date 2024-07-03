import os
import numpy as np
import torch

class StreamerDataset1D:
    def __init__(self, file_path, input_features, output_features=None, t_input=10, T=10, partition='train', min=[], max=[]):
        """
        Initialize the FlashDataset.

        Parameters:
        file_path (str): Path to the npz file.
        input_feature (list): Input features to load.
        output_features (list): Output features to be predicted. If None, the input features are used as output features.
        t_input (int): Number of input time steps.
        T (int): Number of target time steps.
        partition (str): Partition of the data to load. Can be 'train' or 'valid'.
        """
        np.random.seed(0)
        self.file_path = os.path.expanduser(file_path)
        self.input_features = input_features
        self.output_features = input_features if output_features is None else output_features
        self.t_input = t_input
        self.T = T
        self.partition = partition
        self.min = min # Minimum value of the data
        self.max = max # Maximum value of the data
        self.prepare_dataset()
        
    def __load_data(self):
        """
        Load data from the npz file.
        
        * sigma_z[N_runs, N_z]: contains the line profiles for each output
        * phi_z[N_runs, N_z]: idem
        * cycles[N_runs]: output index of a run
        * times[N_runs]: simulation time
        * run_index[N_runs]: index of run
        * z[N_z]: z-coordinates for phi and sigma data
        
        Returns:
        data (numpy.ndarray): Data loaded from the file.
        cycles (numpy.ndarray): Output index loaded from the file.
        run_index (numpy.ndarray): Run number loaded from the file.
        """
        try:
            data_ = np.load(self.file_path)
            data =  [np.float32(data_[input_feature]) for input_feature in self.input_features]
            # (n_runs, n_features, n_z)
            data = np.stack(data, axis=0)
            cycles = np.int32(data_['cycles'])
            run_index = np.int32(data_['run_index'])
        except Exception as e:
            raise IOError(f"Error loading data from {self.file_path}: {e}")
        print(data.shape, cycles.shape, run_index.shape)
        return data, cycles, run_index
        
    def preprocess_data(self, data, cycles, run_index):
        """
        Preprocess data by refactoring, transforming, normalizing, and partitioning.
        
        Parameters:
        data (numpy.ndarray): Data to preprocess.
        cycles (numpy.ndarray): Output index.
        run_index (numpy.ndarray): Run number.
        
        Returns:
        input_tensor (torch.Tensor): Input tensor.
        target_tensor (torch.Tensor): Target tensor.
        
        """
        t_input = self.t_input
        T = self.T
        total_steps = t_input + T
        diff = np.diff(run_index)
        diff = np.where(diff !=0, 1, 0)

        # Indices that indicate where the array will be split
        idx = [i for i in range(len(diff)) if diff[i] == 1 and cycles[i] >= total_steps]
 
        # Split data into chunks of total_steps
        chunks = [data[:, i - total_steps + 1:i + 1] for i in idx]
        
        # Handle the last chunk
        if len(data) - idx[-1] >= total_steps:
            chunks.append(data[idx[-1]:idx[-1] + total_steps])
            
        # (n_sim, n_features, n_total_steps, n_z)
        data = np.stack(chunks, axis=0)

        # Shuffle simulations
        np.random.shuffle(data)

        data = self.__get_partition_data(data, self.partition)

        # Log transformation for the first feature
        data[:, 0] = np.log(data[:, 0])
        
        # Set the minimum and maximum values
        self.__set_stats(data)
        # Normalize the data
        data = self.__range_normalizer(data)

        # Transpose to bring time dimension last
        # (n_sim, n_features, n_total_steps, n_z) -> (n_sim, n_z, n_features, n_total_steps)
        data = np.transpose(data, (0, 3, 1, 2))

        # Split into input and target
        input_data = data[..., :t_input]
        target_data = data[..., :len(self.output_features), t_input:t_input + T]
        
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
        if len(self.min) !=0 and len(self.max) !=0:
            # self.min and self.max are lists of length n_features
            # We should return an array with the same shape as data, with each feature normalized by its own min and max
            return np.stack([(data[:, i] - self.min[i]) / (self.max[i] - self.min[i]) for i in range(len(self.input_features))], axis=1)
        
    def __set_stats(self, data):
        """
        Set the minimum and maximum values of the data.
        
        Parameters:
        data (numpy.ndarray): Data to set the statistics.
        """
        if len(self.min) == 0 and len(self.max) == 0:
            self.min = [data[:, i].min() for i in range(len(self.input_features))]
            self.max = [data[:, i].max() for i in range(len(self.input_features))]
        
    def prepare_dataset(self):
        """
        Prepare the dataset for training.
        """
        data, cycles, run_index = self.__load_data()
        self.preprocess_data(data, cycles, run_index)
        
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