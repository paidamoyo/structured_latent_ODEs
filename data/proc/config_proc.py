from munch import munchify
import os
from collections import OrderedDict
from utils.proc_dataset import depth
import numpy as np
import torch


def load_config():
    args = {}
    args = munchify(args)

    # Data
    args.data_path = "data/proc/"
    args.output_dir = os.getcwd() + "/"
    args.seq_len = 86
    args.obs_dim = 4
    args.aR_dim = 3
    args.aS_dim = 4
    args.C12_dim = 1
    args.C6_dim = 1

    # Training
    args.num_epochs = 2500
    args.mini_batch_size = 36
    args.seed = 12
    args.plot_epoch = 200
    args.heldout = "R33S34_Y81C76"  # 22
    # args.heldout = 'R33S32_Y81C76'
    # args.heldout = None  # TODO: if heldout is selected then it will be used for testing otherwise cross-validation
    if args.heldout is None:
        args.folds = 4
        args.split = 1  # TODO  select from [1, 2, 3, 4]

    # CNN Parameters
    args.n_filters = 10
    args.filter_size = 10
    args.pool_size = 5
    args.cnn_hidden_dim = 50

    # Model
    args.z_aR_dim = 10
    args.z_aS_dim = 10
    args.z_C12_dim = 10
    args.z_C6_dim = 10
    args.z_epsilon_dim = 10
    args.u_hidden_dim = 25  #
    args.aux_loss_multiplier = torch.tensor(46)

    # ODE
    args.ode_state_dim = 8
    args.ode_hidden_dim = 25
    args.system_input_dim = 9
    args.learning_rate = 3e-4
    args.num_particles = 1
    args.num_samples = 200
    args.adjoint_solver = True
    args.solver = "midpoint"
    args.constant_std = 1e-2
    args.quantile_diff = 0.475  # TODO select from [0.25, 0.475]
    args.data = Config().data
    # TODO select from [Mechanistic, MechanisticGauss]
    args.model = "Mechanistic"

    return args


class Config(object):
    def __init__(self):
        data = {
            "groups": {
                "aR": [0, 1, 1, 2, 2, 2],  # LuxR RBS
                "aS": [0, 1, 2, 1, 2, 3],  # LasR RBS
            },
            "devices": [
                "Pcat_Y81C76",
                "RS100S32_Y81C76",
                "RS100S34_Y81C76",
                "R33S32_Y81C76",
                "R33S34_Y81C76",
                "R33S175_Y81C76",
            ],
            "normalize": None,
            "subtract_background": True,
            "conditions": ["C6", "C12"],
            "files": [
                "proc140916.csv",
                "proc140930.csv",
                "proc141006.csv",
                "proc141021.csv",
                "proc141023.csv",
                "proc141028.csv",
            ],
            "signals": ["OD", "mRFP1", "EYFP", "ECFP"],
            "default_devices": dict(),
            "dtype": "float32",
        }

        self.data = munchify(data)
        self.proc_data()

    def proc_data(self):
        # Group-level parameter assignments for each device
        groups_list = [[k, v] for k, v in self.data.groups.items()]
        self.data.component_maps = OrderedDict()
        for k, group in groups_list:
            self.data.component_maps[k] = OrderedDict(zip(self.data.devices, group))
            # Total number of group-level parameters
        self.data.device_depth = sum(
            [depth(cm.values()) for k, cm in self.data.component_maps.items()]
        )
        # Relevance vectors for decoding multi-hot vector into multiple one-hot vectors
        self.data.relevance_vectors = OrderedDict()
        k1 = 0
        for k, group in groups_list:
            k2 = depth(group) + k1
            rv = np.zeros(self.data.device_depth)
            rv[k1:k2] = 1.0
            # print("Relevance for %s: "%k + str(rv))
            if k in self.data.default_devices:
                rv[k1 + self.data.default_devices[k]] = 0.0
            self.data.relevance_vectors[k] = rv.astype(np.float32)
            k1 = k2
        # Manually curated device list: map from device names to 0.0, 1.0, ...
        self.data.device_map = dict(
            zip(self.data.devices, (float(v) for v in range(len(self.data.devices))))
        )
        # Map from device indices (as ints) to device names
        self.data.device_idx_to_device_name = dict(enumerate(self.data.devices))
        # Map from device indices (as floats) to device names
        self.data.device_lookup = {v: k for k, v in self.data.device_map.items()}
