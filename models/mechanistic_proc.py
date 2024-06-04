import pyro
import torch.nn as nn
import pyro.distributions as dist
import torch

from models.encoder_conv import EncoderCONV
from models.encoder_mlp import EncoderMLP
from models.decoders import Decoder
from utils.exp import Exp


class MechanisticModel(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    disentangled variational auto-encoder
    """

    def __init__(self, config, device, times):
        super().__init__()

        self.config = config
        ## Data description
        self.obs_dim = config.obs_dim
        self.times = times

        ## CNN parameters
        self.n_time = len(self.times)
        self.n_filters = config.n_filters
        self.filter_size = config.filter_size
        self.pool_size = config.pool_size
        self.cnn_hidden_dim = config.cnn_hidden_dim
        self.n_channels = self.obs_dim

        ## System input Dimensions
        self.aR_dim = config.aR_dim
        self.aS_dim = config.aS_dim
        self.C12_dim = config.C12_dim
        self.C6_dim = config.C6_dim

        self.aux_loss_multiplier = config.aux_loss_multiplier
        self.condition_on_device = False

        ## Latent Dim
        self.z_aR_dim = config.z_aR_dim
        self.z_aS_dim = config.z_aS_dim
        self.z_C12_dim = config.z_C12_dim
        self.z_C6_dim = config.z_C6_dim
        self.z_epsilon_dim = config.z_epsilon_dim
        self.latent_dim = self.z_aR_dim + self.z_aS_dim + self.z_C12_dim + self.z_C6_dim + self.z_epsilon_dim

        self.device = device
        self.use_cuda = device != 'cpu'
        self.epsilon = 1e-8

        self.u_hidden_dim = config.u_hidden_dim
        self.allow_broadcast = False
        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()
        self.l1_func = nn.L1Loss()

    def setup_networks(self):
        # qi(encoded_data, data.inputs)
        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter

        # Posterior over aR (classifier)
        self.q_aR_given_z_aR = EncoderMLP(
            mlp_sizes=[self.z_aR_dim] + [self.u_hidden_dim] + [self.aR_dim],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over aS (classifier)
        self.q_aS_given_z_aS = EncoderMLP(
            mlp_sizes=[self.z_aS_dim] + [self.u_hidden_dim] + [self.aS_dim],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over C12
        self.q_C12_given_z_C12 = EncoderMLP(
            mlp_sizes=[self.z_C12_dim] + [self.u_hidden_dim] + [[self.C12_dim, self.C12_dim]],
            activation=nn.Softplus,
            output_activation=[Exp, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over C6
        self.q_C6_given_z_C6 = EncoderMLP(
            mlp_sizes=[self.z_C6_dim] + [self.u_hidden_dim] + [[self.C6_dim, self.C6_dim]],
            activation=nn.Softplus,
            output_activation=[Exp, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over z
        self.encoder = EncoderCONV(n_channels=self.n_channels, n_time=self.n_time, n_filters=self.n_filters,
                                   filter_size=self.filter_size, pool_size=self.pool_size,
                                   latent_dim=self.latent_dim,
                                   hidden_dim=self.cnn_hidden_dim)

        system_input_dim = self.C12_dim + self.C6_dim + self.aR_dim + self.aS_dim
        self.z_u_dim = self.z_C12_dim * 4
        self.p_z_u_given_u = EncoderMLP(
            mlp_sizes=[system_input_dim] + [[self.z_u_dim, self.z_u_dim]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Likelihood over observations
        self.decoder = Decoder(config=self.config, times=self.times, latent_dim=self.latent_dim, device=self.device)

        self.constant_std_C_12 = torch.nn.Parameter(torch.ones(1) * self.config.constant_std,
                                                    requires_grad=True)
        self.constant_std_C_6 = torch.nn.Parameter(torch.ones(1) * self.config.constant_std,
                                                   requires_grad=True)

        self.softplus = torch.nn.Softplus()

    def model(self, observations, aR, aS, C12, C6):
        """
        The model corresponds to the following generative process:
        p(x, u, z) = p(u)p(z|u)p(x|z) since x || u |z # note u = [aR, aS]
        p(u) = Cat(alpha)
        p(epsilon) = normal(0,I)              # (latent)
        p(z_u|u) = normal (mu(u), \sigma(u))    # which digit (semi-supervised)
        p(x|z) = ALD(m_gamma(z))   #
        loc is given by a neural network  `decoder`
        :param data: a batch
        :return: X_states, z, Y_observations
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("mechanistic", self)

        with pyro.plate("data"):
            # sample from the constant prior distribution
            z = self.get_prior_z(C12=C12, C6=C6, aR=aR, aS=aS)
            z_split = self.split_z_prior(z)
            self.q_label(aR=aR, aS=aS, z_aR=z_split['z_aR'], z_aS=z_split['z_aS'], name='p')
            self.q_continous(C12=C12, C6=C6, z_C12=z_split['z_C12'], z_C6=z_split['z_C6'], name='p')
            # import ipdb;
            # ipdb.set_trace()
            solution_xt, mu_75, mu_50, mu_25, std = self.decoder.forward(z=z)

        def get_series(id, mu_pred, std_pred, x_greater_mu):
            # P(X < μ) = \tau
            target = observations[:, id, :]
            pred = mu_pred[:, id, :]
            scale = std_pred[:, id, :]

            ge = target.ge(pred)  # X >= mu (scale with tau)
            if x_greater_mu == 1:
                mask = ge
            else:
                mask = torch.logical_not(ge)

            pred = torch.masked_select(input=pred, mask=mask)
            target = torch.masked_select(input=target, mask=mask)
            scale = torch.masked_select(input=scale, mask=mask)

            return pred, scale, target, self.l1_func(target, pred)

        ## Compute Quantile (\tau = 0.75, 0.5)
        median = 0.5
        # diff = 0.475
        diff = self.config.quantile_diff
        lower = median - diff
        upper = median + diff

        x0_data, x1_data, x2_data = self.compute_likelihood(get_series=get_series, mu=mu_50, std=std,
                                                            tau=median)

        _, _, _ = self.compute_likelihood(get_series=get_series, mu=mu_75, std=std,
                                          tau=upper)

        _, _, _, = self.compute_likelihood(get_series=get_series, mu=mu_25, std=std,
                                           tau=lower)
     
         
        # import ipdb;
        # ipdb.set_trace()
        l1_loss = x0_data[3] + x1_data[3] + x2_data[3]
        return l1_loss

    def get_prior_z(self, C12, C6, aR, aS):
        batch_size = C12.shape[0]
        options = dict(dtype=C12.dtype, device=self.device)
        z_epsilon_loc = torch.zeros(batch_size, self.z_epsilon_dim, **options)
        z_epsilon_scale = torch.ones(batch_size, self.z_epsilon_dim, **options)

        system_inputs = torch.cat((aR, aS), dim=1)
        system_inputs = torch.cat((system_inputs, C12), dim=1)
        system_inputs = torch.cat((system_inputs, C6), dim=1)

        z_u_loc, z_u_scale = self.p_z_u_given_u.forward(system_inputs)  # p(z_u|u)
        z_u = pyro.sample("z_u", dist.Normal(z_u_loc, z_u_scale).to_event(1))
        z_epsilon = pyro.sample("z_epsilon",
                                dist.Normal(z_epsilon_loc, z_epsilon_scale).to_event(1))  # p(z_epsilon)
        # pass as tensor z
        z = torch.cat((z_u, z_epsilon), dim=1)
        return z

    def compute_likelihood(self, get_series, mu, std, tau):
        with pyro.poutine.scale(scale=1 - tau):
            # std_tau = std / tau
            std_tau = std
            x_greater_mu = 0
            x0_data = get_series(id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x1_data = get_series(id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x2_data = get_series(id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x3_data = get_series(id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)

            pyro.sample("x_0_{}_g".format(tau), dist.Laplace(x0_data[0], x0_data[1]).to_event(1), obs=x0_data[2])
            pyro.sample("x_1_{}_g".format(tau), dist.Laplace(x1_data[0], x1_data[1]).to_event(1), obs=x1_data[2])
            pyro.sample("x_2_{}_g".format(tau), dist.Laplace(x2_data[0], x2_data[1]).to_event(1), obs=x2_data[2])
            pyro.sample("x_3_{}_g".format(tau), dist.Laplace(x3_data[0], x3_data[1]).to_event(1), obs=x3_data[2])

        with pyro.poutine.scale(scale=tau):
            # actual >= pred then scale with tau
            # P(actual < pred) = tau, where pred is the t-th quantile
            # std_tau = std / (1 - tau)
            std_tau = std
            x_greater_mu = 1
            x0_data = get_series(id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x1_data = get_series(id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x2_data = get_series(id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x3_data = get_series(id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)

            pyro.sample("x_0_{}_l".format(tau), dist.Laplace(x0_data[0], x0_data[1]).to_event(1), obs=x0_data[2])
            pyro.sample("x_1_{}_l".format(tau), dist.Laplace(x1_data[0], x1_data[1]).to_event(1), obs=x1_data[2])
            pyro.sample("x_2_{}_l".format(tau), dist.Laplace(x2_data[0], x2_data[1]).to_event(1), obs=x2_data[2])
            pyro.sample("x_3_{}_l".format(tau), dist.Laplace(x3_data[0], x3_data[1]).to_event(1), obs=x3_data[2])

        return x0_data, x1_data, x2_data

    def guide(self, observations, aR, aS, C12, C6):
        """
        The guide corresponds to the following:
        q(z|x, y) = \frac {q(y|z_c) q(z|x)}{q(y|x)}
        q(z|x) = normal(loc(x),scale(x))       # infer handwriting z_c, z_\c from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y(z_c)` to get q(y|z_c)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # sample (and score) the latent handwriting-style with the variational
            loc_z, scale_z = self.encoder.forward(observations)  # q(z_not_c|x)
            z = self.split_z(loc_z=loc_z, scale_z=scale_z)
            z_aR, z_aS, z_C12, z_C6, z_epsilon, z_u = z['z_aR'], z['z_aS'], z['z_C12'], z['z_C6'], z['z_epsilon'], z[
                'z_u']

            z_u = pyro.sample("z_u", dist.Normal(z_u[0], z_u[1]).to_event(1))
            z_epsilon = pyro.sample("z_epsilon", dist.Normal(z_epsilon[0], z_epsilon[1]).to_event(1))
            return z_aR, z_aS, z_C12, z_C6, z_epsilon

    def split_z_prior(self, z):
        z_aR = z[:, 0:self.z_aR_dim]

        z_group_dim = self.z_aR_dim + self.z_aS_dim
        z_aS = z[:, self.z_aR_dim: z_group_dim]

        z_group_C12_dim = z_group_dim + self.z_C12_dim
        z_C12 = z[:, z_group_dim: z_group_C12_dim]

        z_group_cond_dim = z_group_C12_dim + self.z_C6_dim
        z_C6 = z[:, z_group_C12_dim: z_group_cond_dim]

        return {"z_aR": z_aR,
                "z_aS": z_aS,
                "z_C12": z_C12,
                "z_C6": z_C6
                }

    def split_z(self, loc_z, scale_z):
        loc_z_aR, scale_z_aR = loc_z[:, 0:self.z_aR_dim], scale_z[:, 0:self.z_aR_dim]

        z_group_dim = self.z_aR_dim + self.z_aS_dim
        loc_z_aS, scale_z_aS = loc_z[:, self.z_aR_dim: z_group_dim], scale_z[:, self.z_aR_dim: z_group_dim]

        z_group_C12_dim = z_group_dim + self.z_C12_dim
        loc_z_C12, scale_z_C12 = loc_z[:, z_group_dim: z_group_C12_dim], scale_z[:, z_group_dim: z_group_C12_dim]

        z_group_cond_dim = z_group_C12_dim + self.z_C6_dim
        loc_z_C6, scale_z_6 = loc_z[:, z_group_C12_dim: z_group_cond_dim], scale_z[:,
                                                                           z_group_C12_dim: z_group_cond_dim]

        loc_z_epsilon, scale_z_epsilon = loc_z[:, -self.z_epsilon_dim:], scale_z[:, -self.z_epsilon_dim:]

        z_u_loc = torch.cat((loc_z_aR, loc_z_aS), dim=1)
        z_u_loc = torch.cat((z_u_loc, loc_z_C12), dim=1)
        z_u_loc = torch.cat((z_u_loc, loc_z_C6), dim=1)

        z_u_scale = torch.cat((scale_z_aR, scale_z_aS), dim=1)
        z_u_scale = torch.cat((z_u_scale, scale_z_C12), dim=1)
        z_u_scale = torch.cat((z_u_scale, scale_z_6), dim=1)

        return {"z_aR": [loc_z_aR, scale_z_aR],
                "z_aS": [loc_z_aS, scale_z_aS],
                "z_C12": [loc_z_C12, scale_z_C12],
                "z_C6": [loc_z_C6, scale_z_6],
                'z_epsilon': [loc_z_epsilon, scale_z_epsilon],
                'z_u': [z_u_loc, z_u_scale]
                }

    def model_meta(self, observations, aR, aS, C12, C6):
        """
        q(t|x) = \int q(t|z_t) q(z|x) dz
        q(d|x) = \int q(d|z_d) q(z|x) dz
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("dis_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
            z = self.split_z(loc_z=loc_z, scale_z=scale_z)
            z_aR, z_aS, z_C12, z_C6, z_epsilon = z['z_aR'], z['z_aS'], z['z_C12'], z['z_C6'], z['z_epsilon']

            z_aR = pyro.sample("z_aR_u", dist.Normal(z_aR[0], z_aR[1]).to_event(1))
            z_aS = pyro.sample("z_aS_u", dist.Normal(z_aS[0], z_aS[1]).to_event(1))
            z_C12 = pyro.sample("z_C12_u", dist.Normal(z_C12[0], z_C12[1]).to_event(1))
            z_C6 = pyro.sample("z_C6_u", dist.Normal(z_C6[0], z_C6[1]).to_event(1))

            self.q_label(aR=aR, aS=aS, z_aS=z_aS, z_aR=z_aR, name='q')
            self.q_continous(C12=C12, C6=C6, z_C12=z_C12, z_C6=z_C6, name='q')

    def q_label(self, aR, aS, z_aR, z_aS, name):
        alpha_aR = self.q_aR_given_z_aR.forward(z_aR)  # q(aR|z_aR)
        alpha_aS = self.q_aS_given_z_aS.forward(z_aS)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample("aR_u_" + name, dist.OneHotCategorical(alpha_aR).to_event(1), obs=aR)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample("aS_u_" + name, dist.OneHotCategorical(alpha_aS).to_event(1), obs=aS)

    def q_continous(self, C12, C6, z_C12, z_C6, name):
        loc_C12, _ = self.q_C12_given_z_C12.forward(z_C12)  # q(C12|z_C12)
        loc_C6, _ = self.q_C6_given_z_C6.forward(z_C6)  # q(C6|z_C6)
        std_C12 = self.softplus(self.constant_std_C_12)
        std_C6 = self.softplus(self.constant_std_C_6)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample("C12_u_" + name, dist.Laplace(loc_C12, std_C12).to_event(1), obs=C12)
            pyro.sample("C6_u_" + name, dist.Laplace(loc_C6, std_C6).to_event(1), obs=C6)

    def guide_meta(self, observations, aR, aS, C12, C6):
        """
        dummy meta function to accompany model_meta
        """
        pass

    def pred_inputs(self, observations):
        """
        "Predict prob of aR, aS, C12, C6"
        """
        loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
        z = self.split_z(loc_z=loc_z, scale_z=scale_z)
        z_aR, z_aS, z_C12, z_C6, z_epsilon = z['z_aR'], z['z_aS'], z['z_C12'], z['z_C6'], z['z_epsilon']

        z_aR_sample = torch.normal(z_aR[0], z_aR[1])
        alpha_aR = self.q_aR_given_z_aR(z_aR_sample)

        z_aS_sample = torch.normal(z_aS[0], z_aS[1])
        alpha_aS = self.q_aS_given_z_aS(z_aS_sample)

        z_C12_sample = torch.normal(z_C12[0], z_C12[1])
        pred_C12, _ = self.q_C12_given_z_C12.forward(z_C12_sample)  # q(C12|z_C12)

        z_C6_sample = torch.normal(z_C6[0], z_C6[1])
        pred_C6, _ = self.q_C6_given_z_C6.forward(z_C6_sample)  # q(C6|z_C6)

        # get the index (device) that corresponds to
        # the maximum predicted class probability
        _, ind_aR = torch.topk(alpha_aR, 1)
        _, ind_aS = torch.topk(alpha_aS, 1)

        # convert the digit(s) to one-hot tensor(s)
        pred_aR = torch.zeros_like(alpha_aR).scatter_(1, ind_aR, 1.0)
        pred_aS = torch.zeros_like(alpha_aS).scatter_(1, ind_aS, 1.0)

        return {'aR': pred_aR, 'aS': pred_aS, 'C12': pred_C12, 'C6': pred_C6}

    def recon(self, observations, aR, aS, C12, C6, is_post):
        if is_post:
            loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
            z = torch.normal(loc_z, scale_z)
        else:
            z = self.get_prior_z(C12=C12, C6=C6, aR=aR, aS=aS)
            # import ipdb;
            # ipdb.set_trace()

        solution_xt, mu_75, mu_50, mu_25, std = self.decoder.forward(z=z)
        l1 = self.l1_func(mu_50, observations)
        results = {"l1": l1, "solution_xt": solution_xt, "mu_75": mu_75, "mu_50": mu_50, "mu_25": mu_25, "std": std,
                   "z": z}
        return results
