import torch.nn as nn
import torch
import torchdiffeq


class OdeModel(nn.Module):
    def init_with_params(self, times, ode_state_dim, latent_dim, ode_hidden_dim, adjoint_solver,
                         solver,
                         device):
        super(OdeModel, self).__init__()
        self.times = times
        self.ode_state_dim = ode_state_dim
        self.latent_dim = latent_dim
        self.ode_hidden_dim = ode_hidden_dim
        self.device = device
        self.adjoint_solver = adjoint_solver
        self.solver = solver

        self.latent_to_ode_net = nn.Sequential(nn.Linear(self.latent_dim, self.ode_hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.ode_hidden_dim, self.ode_state_dim),
                                               nn.Sigmoid())

        n_inputs = self.latent_dim

        self.dynamics = Dynamics(n_inputs=n_inputs, hidden_dim=self.ode_hidden_dim, n_outputs=ode_state_dim,
                                 hidden_activation=nn.ReLU)  # n_outputs can be arbitary as well

    def gen_dynamics(self, z):
        return OdeFunc(z=z, dynamics=self.dynamics)

    def initialize_state(self, z):
        x_0 = self.latent_to_ode_net(z)
        return x_0

    def solve_ODE(self, z):
        init_state = self.initialize_state(z).to(self.device)
        d_states_d_t = self.gen_dynamics(z=z)

        if self.adjoint_solver:
            sol = torchdiffeq.odeint_adjoint(func=d_states_d_t, y0=init_state, t=self.times,
                                             method=self.solver)
        else:
            sol = torchdiffeq.odeint(func=d_states_d_t, y0=init_state, t=self.times,
                                     method=self.solver)

        return sol.permute(1, 0, 2)


class OdeFunc(nn.Module):
    def __init__(self, z, dynamics):
        super().__init__()
        self.dynamics = dynamics
        self.n_batch = z.shape[0]
        self.constants = torch.cat([z], dim=1)

    def forward(self, t, state):
        # import ipdb;
        # ipdb.set_trace()
        dx_dt = self.dynamics.forward(t=t, state=state, constants=self.constants, n_batch=self.n_batch)
        return dx_dt


class Dynamics(nn.Module):
    '''Initialize neural dynamics layers'''

    def __init__(self, n_inputs, hidden_dim, n_outputs, hidden_activation=nn.Tanh):
        super(Dynamics, self).__init__()
        print('- Initialising neural dynamics with %d hidden layers' % hidden_dim)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        n_inputs = self.n_inputs + 1  # z + time add time!

        self.dynamics_hidden = nn.Linear(n_inputs, hidden_dim)
        nn.init.xavier_uniform_(self.dynamics_hidden.weight)
        inp_act = hidden_activation()

        self.dyanamics_growth = nn.Linear(hidden_dim, self.n_outputs)
        nn.init.xavier_uniform_(self.dyanamics_growth.weight, gain=0.5)

        self.dyanmics_degradation = nn.Linear(hidden_dim, self.n_outputs)
        nn.init.xavier_uniform_(self.dyanmics_degradation.weight, gain=1)

        self.prod = nn.Sequential(
            self.dynamics_hidden,
            inp_act,
            self.dyanamics_growth,
            nn.Sigmoid()
        )
        self.degr = nn.Sequential(
            self.dynamics_hidden,
            inp_act,
            self.dyanmics_degradation,
            nn.Sigmoid()
        )

    def forward(self, t, state, constants, n_batch):

        t_expanded = t.repeat([n_batch, 1])
        if constants is not None:
            x = torch.cat([t_expanded, constants], dim=1)
        else:
            x = torch.cat([t_expanded], dim=1)
        # import ipdb;
        # ipdb.set_trace()
        xa = self.prod(x)
        xd = self.degr(x)
        dynamics = xa - xd * state
        return dynamics
