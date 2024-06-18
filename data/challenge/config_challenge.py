from munch import munchify
import os
import torch


def load_config():
    args = {}
    args = munchify(args)

    # Data
    args.data_path = os.getcwd() + "/data/challenge/"
    args.norm = "zero_to_one"  # normalize data (0, 1)
    args.obs_dim = 4
    args.shedding_dim = 1
    args.symptoms_dim = 1

    # Model
    args.z_shedding_dim = 5
    args.z_symptoms_dim = 5
    args.z_epsilon_dim = 5
    args.u_hidden_dim = 25  #
    args.aux_loss_multiplier = torch.tensor(46)

    # Training
    args.seed = 12
    args.num_epochs = 500
    args.plot_epoch = 250
    args.mini_batch_size = 100
    args.folds = 5
    args.split = 5  # TODO select from [1, 2, 3, 4, 5]

    # CNN Parameters
    args.n_filters = 10
    args.filter_size = 10
    args.pool_size = 5
    args.cnn_hidden_dim = 50

    # ODE
    args.ode_state_dim = 5
    args.ode_hidden_dim = 25
    args.system_input_dim = 2
    args.learning_rate = 0.001
    args.num_particles = 1
    args.num_samples = 200
    args.adjoint_solver = True
    args.solver = "midpoint"
    args.constant_std = 1e-2
    args.quantile_diff = 0.475  # select from [0.25, 0.475]
    # # select from [Mechanistic, MechanisticGauss]
    args.model = "Mechanistic"
    return args
