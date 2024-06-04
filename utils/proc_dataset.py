from torch.utils.data import Dataset, Subset
import torch
import numpy as np
from data.proc.load_proc_data import load


def depth(group_values):
    return len(set([g for g in group_values if g is not None]))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def merge_observations(times_list, observations_list):
    n_list = np.array([len(t) for t in observations_list])
    loc = np.argmin(n_list)
    chosen_times = times_list[loc]
    aligned_observations = []
    for _, (t, obs) in enumerate(zip(times_list, observations_list)):
        locs = list(map(lambda ti: find_nearest(t, ti), chosen_times))
        aligned_observations.append(obs[:, :, locs])
    merged_observations = np.vstack(aligned_observations)
    return chosen_times, merged_observations


def onehot(i, n):
    """One-hot vector specifiying position i, with length n"""
    v = np.zeros((n))
    if i is not None:
        v[i] = 1
    return v


def scale_data(X, args):
    n_outputs = np.shape(X)[1]
    if args.data.normalize is None:
        scales = [np.max(X[:, i, :]).astype(np.float32) for i in range(n_outputs)]
    else:
        scales = args.data.normalize
    for i, scale in enumerate(scales):
        # First scale the data
        X[:, i, :] /= scale
        # Second, shift so smallest value for each time series is 0
        if args.data.subtract_background:
            mins = np.min(X[:, i, :], axis=1)[:, np.newaxis]
            X[:, i, :] -= mins
    return X, scales


def get_cassettes(devices, args):
    """
    devices: list of device indices (positions in self.device_names above)
    Returns a matrix of ones and zeros, where there are ones wherever
    the device (first index) contains the component (second index), with
    component indices taken from S components then R components.
    Each row of the matrix is a cassette.
    """
    rows = []
    for d in devices:
        device_name = args.data.device_idx_to_device_name[d]
        vs = [
            onehot(cm[device_name], depth(cm.values()))
            for p, cm in args.data.component_maps.items()
        ]
        rows.append(np.hstack(vs))
        # r_matrix[idx, r_value] = 1
    if args.data.dtype == "float32":
        return np.array(rows).astype(np.float32)
    elif args.data.dtype == "float64":
        return np.array(rows).astype(np.float64)


class TimeSeriesDataset(Dataset):
    """A class to facilitate loading batches of time-series observations"""

    def __init__(self, args, parser):
        """
        Args:
            file (string): Path to the csv file with time-series observations.
            root_dir (string): Directory with the files.
        """
        self.parser = parser
        self.args = args

    def _preprocess(self, devices, inputs, times, observations):
        self.devices = devices
        # One-hot encoding of device IDs for each of the L observations: (np.ndarray; L)
        self.dev_1hot = torch.tensor(get_cassettes(devices, self.args))
        # Transformed values of C input conditions, for each of the L observations: (np.ndarray; L x C)
        self.inputs = torch.tensor(np.log(1.0 + inputs))
        # Time-points for the observations: (np.ndarray: T)
        self.times = torch.tensor(times)
        # L observations of time-series (length T) with S observables: (np.ndarray; L x T x S):
        obs, self.scales = scale_data(observations, self.args)
        self.observations = torch.tensor(obs)

    def init_single(self, f):
        devices, inputs, times, observations = self.parser(f, self.args)
        self._preprocess(devices, inputs, times, observations)

    def init_multiple_merge(self):
        devices, inputs, times_list, observations_list = zip(
            *[self.parser(f, self.args) for f in self.args.data.files]
        )
        times, observations = merge_observations(times_list, observations_list)
        # filter_nonempty = [datasets[i] for i in range(len(datasets)) if datasets[i] is not None]
        # dataset = reduce(merge_files, filter_nonempty)
        self._preprocess(
            np.concatenate(devices), np.concatenate(inputs), times, observations
        )

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {
            "devices": self.devices[idx],
            "dev_1hot": self.dev_1hot[idx],
            "inputs": self.inputs[idx],
            "observations": self.observations[idx],
        }


class TimeSeriesDatasetPair(object):
    """A holder for a training and validation set and various associated parameters."""

    # pylint: disable=too-many-instance-attributes,too-few-public-methods

    def __init__(self, train_dataset: Subset, test_dataset: Subset, args):
        """
        :param train: a Dataset containing the training data
        :param val: a Dataset containing the validation data
        """
        # Dataset of the training data
        self.train = train_dataset
        # Dataset of the validation data
        self.test = test_dataset
        # Number of training instances (int)
        self.n_train = len(train_dataset)
        # Number of validation instances (int)
        self.n_test = len(test_dataset)

        # Number of time points and species we're training on (int)
        _, self.n_species, self.n_time = train_dataset.dataset.observations.shape
        # Number of group-level parameters (summed over all groups; int)
        self.depth = args.data.device_depth
        # Number of conditions we're training on
        self.n_conditions = len(args.data.conditions)
        # Numpy array of time-point values (floats), length self.n_time
        self.times = train_dataset.dataset.times


def split_holdout_device(args, data_dct, holdout):
    devices = data_dct.devices.astype(int)
    holdout_device_id = int(args.data.device_map[holdout])
    val_query = devices == holdout_device_id
    train_query = devices != holdout_device_id
    # import ipdb;
    # ipdb.set_trace()

    return (
        np.arange(len(val_query))[train_query],
        np.arange(len(val_query))[val_query],
    )


def build_datasets(config):
    dataset = TimeSeriesDataset(config, load)
    dataset.init_multiple_merge()
    np.random.seed(config.seed)
    if config.heldout:
        # We specified a holdout device to act as the validation set.
        train_ids, val_ids = split_holdout_device(
            args=config, data_dct=dataset, holdout=config.heldout
        )

        # A DatasetPair object: two Datasets (one train, one val) plus associated information.
        train = Subset(dataset, train_ids)
        val = Subset(dataset, val_ids)
        dataset_pair = TimeSeriesDatasetPair(train, val, config)
        # raise NotImplementedError("TODO: implement heldout device")
    else:
        loaded_data_length = len(dataset)
        indices = np.random.permutation(loaded_data_length)
        val_chunks = np.array_split(indices, config.folds)
        assert len(val_chunks) == config.folds, "Bad chunks"
        # All the ids from 0 to W-1 inclusive, in order.
        all_ids = np.arange(loaded_data_length, dtype=int)
        # split runs from 1 to args.folds, so the index we need is one less.
        # val_ids is the indices of data items to be used as validation data.
        val_ids = np.sort(val_chunks[config.split - 1])
        train_ids = np.setdiff1d(all_ids, val_ids)

        # A DatasetPair object: two Datasets (one train, one val) plus associated information.
        train = Subset(dataset, train_ids)
        val = Subset(dataset, val_ids)
        dataset_pair = TimeSeriesDatasetPair(train, val, config)
    return dataset_pair
