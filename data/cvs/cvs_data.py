from munch import munchify
import numpy as np
import torch
import os
from scipy import integrate
from utils.utils import set_seed, find_norm_params


def load_data_config():
    args = {}
    args = munchify(args)
    args.output_dir = os.getcwd() + "/data/cvs/"
    args.seq_len = 86
    args.data_size = 1000
    args.delta_t = 1.0
    args.noise_std = 0.05
    args.seed = 12
    args.num_epochs = 400
    args.norm = "zero_to_one"  # normalize data (0, 1)
    args.mini_batch_size = 128
    return args


def get_random_params():
    i_ext = 0.0 if np.random.rand() > 0.5 else -2.0
    r_tpr_mod = 0.0 if np.random.rand() > 0.5 else 0.5

    return {
        "i_ext": i_ext,
        "r_tpr_mod": r_tpr_mod,
        "f_hr_max": 3.0,
        "f_hr_min": 2.0 / 3.0,
        "r_tpr_max": 2.134,
        "r_tpr_min": 0.5335,
        "sv_mod": 0.0001,
        "ca": 4.0,
        "cv": 111.0,
        # dS/dt parameters
        "k_width": 0.1838,
        "p_aset": 70,
        "tau": 20,
        "p_0lv": 2.03,
        "r_valve": 0.0025,
        "k_elv": 0.066,
        "v_ed0": 7.14,
        "T_sys": 4.0 / 15.0,
        "cprsw_max": 103.8,
        "cprsw_min": 25.9,
    }


def dx_dt(state, t, params):
    # Shared Parameters:
    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    r_tpr_max = params["r_tpr_max"]
    r_tpr_min = params["r_tpr_min"]
    ca = params["ca"]
    cv = params["cv"]
    k_width = params["k_width"]
    p_aset = params["p_aset"]
    tau = params["tau"]
    sv_mod = params["sv_mod"]

    # Observation specific parameters:
    i_ext = params["i_ext"]
    r_tpr_mod = params["r_tpr_mod"]

    # State variables
    p_a = 100.0 * state[0]
    p_v = 10.0 * state[1]
    s = state[2]
    sv = 100.0 * state[3]

    # Building f_hr and r_tpr:
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

    # Building dp_a/dt and dp_v/dt:
    dva_dt = -1.0 * (p_a - p_v) / r_tpr + sv * f_hr
    dvv_dt = -1.0 * dva_dt + i_ext
    dpa_dt = dva_dt / (ca * 100.0)
    dpv_dt = dvv_dt / (cv * 10.0)

    # Building dS/dt:
    ds_dt = (1.0 / tau) * (1.0 - 1.0 / (1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)

    dsv_dt = i_ext * sv_mod  ## TODO verify this

    # State derivative
    return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])


def states_trajectory_to_sample(states, params):
    p_a = states[:, 0]
    p_v = states[:, 1]
    s = states[:, 2]

    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    # init_state = np.array([init_pa, init_pv, init_s, init_sv])
    return np.stack((p_a, p_v, f_hr), axis=1)


def init_random_state():
    init_state = np.ones(4) * 1
    return init_state


def create_cvs_data(args):
    t = np.arange(
        0.0, stop=(args.seq_len) * args.delta_t, step=args.delta_t
    )  # from time 0 t0 86

    sample_size = 3
    state_size = 4

    raw_data = np.zeros(tuple([args.data_size, args.seq_len, sample_size]))
    latent_data = np.zeros((args.data_size, args.seq_len, state_size))
    params_data = []

    for i in range(args.data_size):
        # initial state
        init_state = init_random_state()
        params = get_random_params()
        params_data.append(params)

        states_trajectory = integrate.odeint(dx_dt, init_state, t, args=tuple([params]))

        raw_data[i] = states_trajectory_to_sample(states_trajectory, params)
        latent_data[i] = states_trajectory

    return raw_data, latent_data, params_data


def add_noise(args, data):
    noisy_data = data + args.noise_std * np.random.normal(size=data.shape)
    return noisy_data


def make_dataset(args):
    raw_data, latent_data, params_data = create_cvs_data(args)
    print("raw_data: ", raw_data.shape)
    print("latent_data: ", latent_data.shape)
    print("params_data: ", len(params_data))

    buffer = int(round(raw_data.shape[0] * (1 - 0.1)))

    train_data = raw_data[:buffer]
    test_data = raw_data[buffer:]

    noisy_train_data = add_noise(args, train_data)
    noisy_test_data = add_noise(args, test_data)

    train_latent_data = latent_data[:buffer]
    test_latent_data = latent_data[buffer:]

    train_params_data = params_data[:buffer]
    train_params_data = {
        key: np.array([sample[key] for sample in train_params_data])
        for key in train_params_data[0]
    }

    test_params_data = params_data[buffer:]
    test_params_data = {
        key: np.array([sample[key] for sample in test_params_data])
        for key in test_params_data[0]
    }

    torch.save(train_params_data, args.output_dir + "train_params_data.pkl")
    torch.save(test_params_data, args.output_dir + "test_params_data.pkl")

    torch.save(train_latent_data, args.output_dir + "train_latent_data.pkl")
    torch.save(test_latent_data, args.output_dir + "test_latent_data.pkl")

    data_norm_params = find_norm_params(noisy_train_data)
    torch.save(data_norm_params, args.output_dir + "data_norm_params.pkl")

    dataset_dict = {"train": noisy_train_data, "test": noisy_test_data}

    torch.save(dataset_dict, args.output_dir + "processed_data.pkl")
    torch.save(test_data, args.output_dir + "gt_test_data.pkl")


if __name__ == "__main__":
    args = load_data_config()
    print(args)
    set_seed(args.seed)

    make_dataset(args)
