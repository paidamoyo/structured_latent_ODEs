import torch.nn as nn
import torch


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)


class EncoderCONV(nn.Module):
    def __init__(self, n_channels, n_filters, filter_size, pool_size, n_time, latent_dim, hidden_dim):
        super(EncoderCONV, self).__init__()
        n_filters = n_filters
        filter_size = filter_size
        pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        n_conv = n_time - (filter_size - 1)
        n_pool = n_conv - (pool_size - 1)
        self.n_hidden_layer = n_pool * n_filters

        # Define layers
        self.conv = nn.Conv1d(n_channels, n_filters, filter_size)
        nn.init.orthogonal_(self.conv.weight)
        self.pool = nn.AvgPool1d(pool_size, stride=1)
        self.lin = nn.Linear(self.n_hidden_layer, self.hidden_dim)
        nn.init.orthogonal_(self.lin.weight)
        self.act = nn.Tanh()
        self.z_loc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_scale = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            Exp()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        x = self.act(x)
        z_loc = self.z_loc(x)
        z_scale = self.z_scale(x)
        return z_loc, z_scale
