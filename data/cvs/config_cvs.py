from munch import munchify
import os
import torch


def load_config():
    args = {}
    args = munchify(args)

    # Data
    args.data_path = os.getcwd() + "/data/cvs/"
    args.seq_len = 86  #
    args.data_size = 1000
    args.delta_t = 1.0
    args.norm = "zero_to_one"  # normalize data (0, 1)
    args.obs_dim = 3
    args.iext_dim = 1
    args.rtpr_dim = 1

    # Model
    args.z_iext_dim = 5
    args.z_rtpr_dim = 5
    args.z_epsilon_dim = 5
    args.u_hidden_dim = 25  #
    args.aux_loss_multiplier = torch.tensor(46)

    # Training
    args.seed = 12
    args.num_epochs = 1000
    args.plot_epoch = 100
    args.mini_batch_size = 128

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
    args.adjoint_solver = True
    args.solver = "midpoint"
    args.constant_std = 1e-2
    args.quantile_diff = 0.475  # select from [0.25, 0.475]
    # args.solver = 'rk4'
    # select from  [Mechanistic, MechanisticGauss] # MechanisticGauss is the ablation model
    args.model = "Mechanistic"
    return args
