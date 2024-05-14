import numpy as np
import torch
import random


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True


def find_norm_params(data):
    print("find_norm_params: ", data.shape)
    size_features = data.shape[2]
    mean = np.zeros(size_features)
    std = np.zeros(size_features)
    for feature in range(size_features):
        mean[feature] = data[:, :, feature].mean()
        std[feature] = data[:, :, feature].std()

    max_val = np.zeros(size_features)
    min_val = np.zeros(size_features)
    for feature in range(size_features):
        max_val[feature] = (data[:, :, feature].max())
        min_val[feature] = (data[:, :, feature].min())

    print("max_val: ", max_val, "min_val: ", min_val)

    data_norm_params = {"mean": mean,
                        "std": std,
                        "max": max_val,
                        "min": min_val}

    return data_norm_params