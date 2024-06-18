import pandas as pd
import numpy as np
from data.challenge.config_challenge import load_config
from utils.utils import find_norm_params
import os
import pickle


class TimeSeriesDatasetPair(object):
    def __init__(self, dataset, train_ids, test_ids, max_time):
        self.train = {
            "observations": dataset["observations"][train_ids],
            "shedding": dataset["shedding"][train_ids],
            "symptoms": dataset["symptoms"][train_ids],
        }
        # Dataset of the validation data
        self.test = {
            "observations": dataset["observations"][test_ids],
            "shedding": dataset["shedding"][test_ids],
            "symptoms": dataset["symptoms"][test_ids],
        }
        # Number of training instances (int)
        self.n_train = len(train_ids)
        # Number of validation instances (int)
        self.n_test = len(test_ids)
        self.max_time = max_time
        self.data_norm_params = find_norm_params(self.train["observations"])


def build_datasets(config):
    # dataset, max_time = process_data()
    with open("data/challenge/data.pkl", "rb") as pickle_file:
        dataset = pickle.load(pickle_file)
        print("Loading data from pickle file", dataset.keys())
    max_time = dataset["n_time"]

    np.random.seed(config.seed)

    loaded_data_length = dataset["observations"].shape[0]
    indices = np.random.permutation(loaded_data_length)
    val_chunks = np.array_split(indices, config.folds)
    assert len(val_chunks) == config.folds, "Bad chunks"
    # All the ids from 0 to W-1 inclusive, in order.
    all_ids = np.arange(loaded_data_length, dtype=int)
    # split runs from 1 to args.folds, so the index we need is one less.
    # val_ids is the indices of data items to be used as validation data.
    val_ids = np.sort(val_chunks[config.split - 1])
    train_ids = np.setdiff1d(all_ids, val_ids)

    dataset_pair = TimeSeriesDatasetPair(
        dataset=dataset, train_ids=train_ids, test_ids=val_ids, max_time=max_time
    )

    return dataset_pair


if __name__ == "__main__":
    config = load_config()
    build_datasets(config)
