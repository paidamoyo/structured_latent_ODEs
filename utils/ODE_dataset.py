from torch.utils.data import Dataset
import torch
import random


class ODEDataCSV(Dataset):
    def __init__(self, data_dir, ds_type, seq_len, random_start, transforms=None):
        self.transforms = transforms if transforms is not None else {}
        self.random_start = random_start
        # self.train = train
        self.ds_type = ds_type
        self.seq_len = seq_len

        obs_dict = torch.load(data_dir + 'processed_data.pkl')
        train_params = torch.load(data_dir + 'train_params_data.pkl')
        test_params = torch.load(data_dir + 'test_params_data.pkl')
        # print("data_dict: ", data_dict)

        buffer = int(round(obs_dict["train"].shape[0] * (1 - 0.1)))
        if ds_type == 'train':
            self.obs = torch.FloatTensor(obs_dict["train"])[:buffer]
            self.iext = torch.FloatTensor(train_params['i_ext'])[:buffer]
            self.rtpr = torch.FloatTensor(train_params['r_tpr_mod'])[:buffer]
            print("TRAIN: ", "obs=", self.obs.shape, "rtpr: ", torch.unique(self.rtpr), "iext:",
                  torch.unique(self.iext))

        elif ds_type == 'val':
            self.obs = torch.FloatTensor(obs_dict["train"])[buffer:]
            self.iext = torch.FloatTensor(train_params['i_ext'])[buffer:]
            self.rtpr = torch.FloatTensor(train_params['r_tpr_mod'])[buffer:]
            print("VAL: ", "obs=", self.obs.shape, "rtpr: ", torch.unique(self.rtpr), "iext:", torch.unique(self.iext))

        elif ds_type == 'test':
            self.obs = torch.FloatTensor(obs_dict["test"])
            self.iext = torch.FloatTensor(test_params['i_ext'])
            self.rtpr = torch.FloatTensor(test_params['r_tpr_mod'])
            print("TEST: ", "obs=", self.obs.shape, "rtpr: ", torch.unique(self.rtpr), "iext:",
                  torch.unique(self.iext))

    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        if self.random_start:
            start_time = random.randint(0, self.obs.size(1) - self.seq_len)
        else:
            start_time = 0

        obs = self.obs[idx, start_time:start_time + self.seq_len]
        iext = (self.iext[idx] >= 0).float()  # 0 or -0.2
        rtpr = (self.rtpr[idx] > 0).float()  # 0 or 0.5

        for transform in self.transforms:
            obs = self.transforms[transform](obs)

        return {'observations': obs, 'iext': iext, 'rtpr': rtpr}


class ODEDataChallenge(Dataset):
    def __init__(self, data, ds_type, seq_len, random_start, transforms=None):
        self.transforms = transforms if transforms is not None else {}
        self.random_start = random_start
        # self.train = train
        self.ds_type = ds_type
        self.seq_len = seq_len

        self.obs = torch.FloatTensor(data["observations"])
        self.shedding = torch.FloatTensor(data['shedding'])
        self.symptom = torch.FloatTensor(data['symptoms'])
        print("obs=", self.obs.shape, "symptom: ", torch.sum(self.symptom) / len(self.symptom), "shedding:",
              torch.sum(self.shedding) / len(self.shedding))

    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        shedding = self.shedding[idx].float()
        symptoms = self.symptom[idx].float()

        for transform in self.transforms:
            obs = self.transforms[transform](obs)

        return {'observations': obs, 'shedding': shedding, 'symptoms': symptoms}


class ODEDataSynBio(Dataset):
    def __init__(self, data_dir, ds_type, seq_len, random_start, transforms=None):
        self.transforms = transforms if transforms is not None else {}
        self.random_start = random_start
        # self.train = train
        self.ds_type = ds_type
        self.seq_len = seq_len

        obs_dict = torch.load(data_dir + 'processed_data.pkl')
        train_params = torch.load(data_dir + 'train_params_data.pkl')
        test_params = torch.load(data_dir + 'test_params_data.pkl')
        # print("data_dict: ", data_dict)
        devices = ['Pcat-Pcat', 'R100-S32', 'R100-S34', 'R33-S32', 'R33-S34', 'R33-S175']  # 3, 4
        self.aR_to_num = {
            # Pcat, R100, R33
            0: 0,  # Pcat
            10: 1,  # R100
            20: 2,  # R33
        }

        self.aS_to_num = {
            # Pcat, S32, S34, S175
            0: 0,  # Pcat
            30: 1,  # S32
            50: 2,  # S175
            80: 3  # S34
        }

        # device_to_num = {
        #     "Pcat": 0,
        #     "R100": 10,
        #     "R33": 20,
        #     "S32": 30,
        #     "S175": 50,
        #     "S34": 80,
        # }

        buffer = int(round(obs_dict["train"].shape[0] * (1 - 0.1)))
        if ds_type == 'train':
            self.obs = torch.FloatTensor(obs_dict["train"])[:buffer]
            self.aR = torch.FloatTensor(train_params['aR'])[:buffer]
            self.aS = torch.FloatTensor(train_params['aS'])[:buffer]
            print("TRAIN: ", "obs=", self.obs.shape, "aR: ", torch.unique(self.aR), "aS:",
                  torch.unique(self.aS))

        elif ds_type == 'val':
            self.obs = torch.FloatTensor(obs_dict["train"])[buffer:]
            self.aR = torch.FloatTensor(train_params['aR'])[buffer:]
            self.aS = torch.FloatTensor(train_params['aS'])[buffer:]
            print("VAL: ", "obs=", self.obs.shape, "aR: ", torch.unique(self.aR), "aS:",
                  torch.unique(self.aS))

        elif ds_type == 'test':
            self.obs = torch.FloatTensor(obs_dict["test"])
            self.aR = torch.FloatTensor(test_params['aR'])
            self.aS = torch.FloatTensor(test_params['aS'])
            print("TEST: ", "obs=", self.obs.shape, "aR: ", torch.unique(self.aR), "aS:",
                  torch.unique(self.aS))

    def __encode__(self, val, lookup):
        idx = lookup[int(val)]
        zeros = torch.zeros(len(lookup))
        zeros[idx] = 1
        return zeros

    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        if self.random_start:
            start_time = random.randint(0, self.obs.size(1) - self.seq_len)
        else:
            start_time = 0

        obs = self.obs[idx, start_time:start_time + self.seq_len]
        aR = self.__encode__(val=self.aR[idx], lookup=self.aR_to_num)
        aS = self.__encode__(self.aS[idx], lookup=self.aS_to_num)

        for transform in self.transforms:
            obs = self.transforms[transform](obs)

        return {'observations': obs, 'aS': aS, 'aR': aR}


class NormalizeZScore(object):
    """Normalize sample by mean and std."""

    def __init__(self, data_norm_params):
        self.mean = torch.FloatTensor(data_norm_params["mean"])
        self.std = torch.FloatTensor(data_norm_params["std"])

    def __call__(self, sample):
        new_sample = torch.zeros_like(sample, dtype=torch.float)
        for feature in range(self.mean.size(0)):
            if self.std[feature] > 0:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature]) / self.std[feature]
            else:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature])

        return new_sample

    def denormalize(self, batch):
        denormed_batch = torch.zeros_like(batch)
        for feature in range(batch.size(2)):
            denormed_batch[:, :, feature] = (batch[:, :, feature] * self.std[feature]) + self.mean[feature]

        return denormed_batch


class NormalizeToUnitSegment(object):
    """Normalize sample to the segment [0, 1] by max and min"""

    def __init__(self, data_norm_params):
        self.min_val = data_norm_params["min"]
        self.max_val = data_norm_params["max"]
        print("INIT min_val: ", self.min_val, "max_val:", self.max_val)

    def __call__(self, sample):
        new_sample = torch.zeros_like(sample, dtype=torch.float)
        for feature in range(self.min_val.shape[0]):
            new_sample[:, feature] = (sample[:, feature] - self.min_val[feature]) / (
                    self.max_val[feature] - self.min_val[feature])
        return new_sample

    def denormalize(self, batch):
        denormed_batch = torch.zeros_like(batch)
        for feature in range(batch.size(2)):
            denormed_batch[:, :, feature] = (batch[:, :, feature] * (self.max_val[feature] - self.min_val[feature])) + \
                                            self.min_val[feature]
        return denormed_batch


def create_transforms(args, data_norm_params=None):
    if data_norm_params is None:
        data_norm_params = torch.load(args.data_path + 'data_norm_params.pkl')

    data_transforms = {}
    # Normalization transformation
    if args.norm is not None:
        if args.norm == "zscore":
            normalize_transform = NormalizeZScore(data_norm_params)
        elif args.norm == "zero_to_one":
            normalize_transform = NormalizeToUnitSegment(data_norm_params)
        else:
            raise Exception("Choose valid normalization function: zscore or zero_to_one")
        data_transforms["normalize"] = normalize_transform
    return data_transforms
