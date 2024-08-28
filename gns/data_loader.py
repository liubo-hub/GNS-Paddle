import paddle
import numpy as np


def load_npz_data(path):
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data


class SamplesDataset(paddle.io.Dataset):
    """Dataset of samples of trajectories.

    Each sample is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    particle_type is an integer.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """

    def __init__(self, path, input_length_sequence):
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [(x.shape[0] - self._input_length_sequence) for
            x, _ in self._data]
        self._length = sum(self._data_lengths)
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in
            range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths,
            dtype=int)

    def __len__(self):
        """Return length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.

        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
        """
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1,
            idx, side='left')
        start_of_selected_trajectory = self._precompute_cumlengths[
            trajectory_idx - 1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx -
            start_of_selected_trajectory)
        positions = self._data[trajectory_idx][0][time_idx - self.
            _input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(positions.shape[0], self._data[
            trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        label = self._data[trajectory_idx][0][time_idx]
        return (positions, particle_type, n_particles_per_example), label


def collate_fn(data):
    """Collate function for SamplesDataset.

    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
    """
    position_list = []
    particle_type_list = []
    n_particles_per_example_list = []
    label_list = []
    for (positions, particle_type, n_particles_per_example), label in data:
        position_list.append(positions)
        particle_type_list.append(particle_type)
        n_particles_per_example_list.append(n_particles_per_example)
        label_list.append(label)
    return (paddle.to_tensor(data=np.vstack(position_list),dtype='float32'),
        paddle.to_tensor(data=np.concatenate(particle_type_list)), paddle.
        to_tensor(data=n_particles_per_example_list)), paddle.to_tensor(data
        =np.vstack(label_list),dtype='float32')


class TrajectoriesDataset(paddle.io.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """

    def __init__(self, path):
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)

    def __len__(self):
        """Return length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.

        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form (positions, particle_type).
        """
        positions, _particle_type = self._data[idx]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
        n_particles_per_example = positions.shape[0]
        return paddle.to_tensor(data=positions,dtype='float32'), paddle.to_tensor(data=particle_type), n_particles_per_example

class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
def get_data_loader_by_samples(path, input_length_sequence, batch_size,
    shuffle=True):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """

    dataset = SamplesDataset(path, input_length_sequence)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path)
    return DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
