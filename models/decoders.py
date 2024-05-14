import torch
from torch import nn
from models.blackbox_ode import OdeModel


class Decoder(nn.Module):
    '''
    Decoder network
    '''

    def __init__(self, config, times, latent_dim, device):
        super(Decoder, self).__init__()
        print("Initialising decoder")
        self.ode_model = OdeModel()
        self.times = times
        self.ode_state_dim = config.ode_state_dim
        self.obs_dim = config.obs_dim
        self.latent_dim = latent_dim
        self.ode_hidden_dim = config.ode_hidden_dim
        system_input_dim = config.system_input_dim

        self.ode_model.init_with_params(times=self.times, ode_state_dim=self.ode_state_dim, latent_dim=self.latent_dim,
                                        ode_hidden_dim=self.ode_hidden_dim,
                                        adjoint_solver=config.adjoint_solver, solver=config.solver,
                                        device=device)

        self.output_q50 = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.output_q75 = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.output_q25 = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.constant_std = torch.nn.Parameter(torch.ones(self.obs_dim, len(self.times)) * config.constant_std,
                                               requires_grad=True)

    def forward(self, z):
        solution = self.ode_model.solve_ODE(z=z)

        mu_50 = torch.squeeze(self.output_q50(solution)).permute(0, 2, 1)  # obs * K * T
        mu_75 = torch.squeeze(self.output_q75(solution)).permute(0, 2, 1)
        mu_25 = torch.squeeze(self.output_q25(solution)).permute(0, 2, 1)

        # import ipdb;
        # ipdb.set_trace()

        softplus = torch.nn.Softplus()
        std = torch.ones_like(mu_50) * softplus(self.constant_std)
        return solution, mu_75, mu_50, mu_25, std


class GaussianDecoder(nn.Module):
    '''
      Decoder network
      '''

    def __init__(self, config, times, latent_dim, device):
        super(GaussianDecoder, self).__init__()
        print("Initialising decoder")
        self.ode_model = OdeModel()
        self.times = times
        self.ode_state_dim = config.ode_state_dim
        self.obs_dim = config.obs_dim
        self.latent_dim = latent_dim
        self.ode_hidden_dim = config.ode_hidden_dim

        self.ode_model.init_with_params(times=self.times, ode_state_dim=self.ode_state_dim, latent_dim=self.latent_dim,
                                        ode_hidden_dim=self.ode_hidden_dim,
                                        adjoint_solver=config.adjoint_solver, solver=config.solver,
                                        device=device)

        self.output_mean = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.constant_std = torch.nn.Parameter(torch.ones(self.obs_dim, len(self.times)) * config.constant_std,
                                               requires_grad=True)

    def forward(self, z):
        solution = self.ode_model.solve_ODE(z=z)
        mean = torch.squeeze(self.output_mean(solution)).permute(0, 2, 1)  # obs * K * T
        softplus = torch.nn.Softplus()
        std = torch.ones_like(mean) * softplus(self.constant_std)
        # import ipdb;
        # ipdb.set_trace()
        return solution, mean, std


class VarianceGaussianDecoder(nn.Module):
    '''
      Decoder network
      '''

    def __init__(self, config, times, latent_dim, device):
        super(VarianceGaussianDecoder, self).__init__()
        print("Initialising decoder")
        self.ode_model = OdeModel()
        self.std_ode_model = OdeModel()
        self.times = times
        self.ode_state_dim = config.ode_state_dim
        self.obs_dim = config.obs_dim
        self.latent_dim = latent_dim
        self.ode_hidden_dim = config.ode_hidden_dim
        system_input_dim = config.system_input_dim

        self.ode_model.init_with_params(times=self.times, ode_state_dim=self.ode_state_dim, latent_dim=self.latent_dim,
                                        ode_hidden_dim=self.ode_hidden_dim,
                                        adjoint_solver=config.adjoint_solver, solver=config.solver,
                                        device=device)

        self.std_ode_model.init_with_params(times=self.times, ode_state_dim=self.ode_state_dim,
                                            latent_dim=self.latent_dim,
                                            ode_hidden_dim=self.ode_hidden_dim,
                                            adjoint_solver=config.adjoint_solver, solver=config.solver,
                                            device=device)

        self.output_mean = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.output_std = nn.Sequential(
            nn.Linear(self.ode_state_dim, self.obs_dim, bias=False),
        )

        self.constant_std = torch.nn.Parameter(torch.ones(self.obs_dim, len(self.times)) * config.constant_std,
                                               requires_grad=True)

    def forward(self, z):
        solution = self.ode_model.solve_ODE(z=z)
        mean = torch.squeeze(self.output_mean(solution)).permute(0, 2, 1)  # obs * K * T

        solution_std = self.std_ode_model.solve_ODE(z=z)
        std = torch.squeeze(self.output_std(solution_std)).permute(0, 2, 1)
        # import ipdb;
        # ipdb.set_trace()
        return solution, mean, std
