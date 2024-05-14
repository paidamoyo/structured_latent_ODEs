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
        self.iext_dim = config.iext_dim
        self.rtpr_dim = config.rtpr_dim

        self.aux_loss_multiplier = torch.tensor(46)
        self.condition_on_device = False

        ## Latent Dim
        self.z_iext_dim = config.z_iext_dim
        self.z_rtpr_dim = config.z_rtpr_dim
        self.z_epsilon_dim = config.z_epsilon_dim
        self.latent_dim = self.z_iext_dim + self.z_rtpr_dim + self.z_epsilon_dim

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

        # Posterior over iext (classifier)
        self.q_iext_given_z_iext = EncoderMLP(
            mlp_sizes=[self.z_iext_dim] + [self.u_hidden_dim] + [self.iext_dim],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over rtpr (classifier)
        self.q_rtpr_given_z_rtpr = EncoderMLP(
            mlp_sizes=[self.z_rtpr_dim] + [self.u_hidden_dim] + [self.rtpr_dim],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda)

        # Posterior over z
        self.encoder = EncoderCONV(n_channels=self.n_channels, n_time=self.n_time, n_filters=self.n_filters,
                                   filter_size=self.filter_size, pool_size=self.pool_size,
                                   latent_dim=self.latent_dim,
                                   hidden_dim=self.cnn_hidden_dim)

        # Prior over z_iext and z_rtpr
        self.p_z_iext_given_iext = EncoderMLP(mlp_sizes=[self.iext_dim] + [
            [self.z_iext_dim, self.z_iext_dim]],
                                              activation=nn.Softplus,
                                              output_activation=[None, Exp],
                                              allow_broadcast=self.allow_broadcast,
                                              use_cuda=self.use_cuda)

        self.p_z_rtprs_given_rtprs = EncoderMLP(mlp_sizes=[self.rtpr_dim] + [
            [self.z_rtpr_dim, self.z_rtpr_dim]],
                                                activation=nn.Softplus,
                                                output_activation=[None, Exp],
                                                allow_broadcast=self.allow_broadcast,
                                                use_cuda=self.use_cuda)

        # Likelihood over observations
        self.decoder = Decoder(config=self.config, times=self.times, latent_dim=self.latent_dim, device=self.device)

    def model(self, observations, iext, rtpr):
        """
        The model corresponds to the following generative process:
        p(x, u, z) = p(u)p(z|u)p(x|z) since x || u |z # note u = [iext, rtprs]
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

        batch_size = observations.shape[0]
        options = dict(dtype=observations.dtype, device=self.device)

        with pyro.plate("data"):
            # sample from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_epsilon_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_epsilon_dim, **options)

            z_epsilon = pyro.sample("z_epsilon", dist.Normal(prior_loc, prior_scale).to_event(1))  # p(z_not_c)

            # print("iext: ", iext.shape)
            z_iext_loc, z_iext_scale = self.p_z_iext_given_iext.forward(iext)  # p(z_iext|iext)
            z_iext = pyro.sample("z_iext", dist.Normal(z_iext_loc, z_iext_scale).to_event(1))

            z_rtpr_loc, z_rtpr_scale = self.p_z_rtprs_given_rtprs.forward(rtpr)  # p(z_rtpr|rtpr)
            z_rtpr = pyro.sample("z_rtpr", dist.Normal(z_rtpr_loc, z_rtpr_scale).to_event(1))
            # pass as tensor z
            z = torch.cat((z_iext, z_rtpr), dim=1)
            z = torch.cat((z, z_epsilon), dim=1)
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
        diff = 0.475
        lower = median - diff
        upper = median + diff

        x0_data, x1_data, x2_data = self.compute_likelihood(get_series=get_series, mu=mu_50, std=std,
                                                            tau=median)
        _, _, _ = self.compute_likelihood(get_series=get_series, mu=mu_75, std=std,
                                          tau=upper)

        _, _, _, = self.compute_likelihood(get_series=get_series, mu=mu_25, std=std,
                                           tau=lower)

        # self.q_label(iext=iext, rtpr=rtpr, z_iext=z_iext, z_rtpr=z_rtpr)
        # import ipdb;
        # ipdb.set_trace()
        l1_loss = x0_data[3] + x1_data[3] + x2_data[3]
        return l1_loss

    def compute_likelihood(self, get_series, mu, std, tau):
        with pyro.poutine.scale(scale=1 - tau):
            # std_tau = std / tau
            std_tau = std
            x_greater_mu = 0
            x0_data = get_series(id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x1_data = get_series(id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x2_data = get_series(id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            # x3_data = get_series(id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)

            pyro.sample("x_0_{}_g".format(tau), dist.Laplace(x0_data[0], x0_data[1]).to_event(1), obs=x0_data[2])
            pyro.sample("x_1_{}_g".format(tau), dist.Laplace(x1_data[0], x1_data[1]).to_event(1), obs=x1_data[2])
            pyro.sample("x_2_{}_g".format(tau), dist.Laplace(x2_data[0], x2_data[1]).to_event(1), obs=x2_data[2])
            # pyro.sample("x_3_{}_g".format(tau), dist.Laplace(x3_data[0], x3_data[1]).to_event(1), obs=x3_data[2])

        with pyro.poutine.scale(scale=tau):
            # actual >= pred then scale with tau
            # P(actual < pred) = tau, where pred is the t-th quantile
            # std_tau = std / (1 - tau)
            std_tau = std
            x_greater_mu = 1
            x0_data = get_series(id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x1_data = get_series(id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            x2_data = get_series(id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)
            # x3_data = get_series(id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu)

            pyro.sample("x_0_{}_l".format(tau), dist.Laplace(x0_data[0], x0_data[1]).to_event(1), obs=x0_data[2])
            pyro.sample("x_1_{}_l".format(tau), dist.Laplace(x1_data[0], x1_data[1]).to_event(1), obs=x1_data[2])
            pyro.sample("x_2_{}_l".format(tau), dist.Laplace(x2_data[0], x2_data[1]).to_event(1), obs=x2_data[2])
            # pyro.sample("x_3_{}_l".format(tau), dist.Laplace(x3_data[0], x3_data[1]).to_event(1), obs=x3_data[2])

        return x0_data, x1_data, x2_data

    def guide(self, observations, iext, rtpr):
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
            loc_z, scale_z = self.encoder.forward(observations)  # q(z_not_c|x)

            z_iext = pyro.sample("z_iext",
                                 dist.Normal(loc_z[:, 0:self.z_iext_dim],
                                             scale_z[:, 0:self.z_iext_dim]).to_event(
                                     1))
            z_rtpr = pyro.sample("z_rtpr", dist.Normal(
                loc_z[:, self.z_iext_dim: self.z_iext_dim + self.z_rtpr_dim],
                scale_z[:,
                self.z_iext_dim: self.z_iext_dim + self.z_rtpr_dim]).to_event(1))
            z_epsilon = pyro.sample("z_epsilon",
                                    dist.Normal(loc_z[:, -self.z_epsilon_dim:],
                                                scale_z[:, -self.z_epsilon_dim:]).to_event(1))
            return z_iext, z_rtpr, z_epsilon

    def model_meta(self, observations, iext, rtpr):
        """
        q(t|x) = \int q(t|z_t) q(z|x) dz
        q(d|x) = \int q(d|z_d) q(z|x) dz
        """
        pyro.module("dis_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)

            loc_z_iext, scale_z_iext = loc_z[:, 0:self.z_iext_dim], scale_z[:,
                                                                    0:self.z_iext_dim]
            z_iext = pyro.sample("z_iext_cls", dist.Normal(loc_z_iext, scale_z_iext).to_event(1))

            z_group_dim = self.z_iext_dim + self.z_rtpr_dim
            loc_z_rtpr, scale_z_rtpr = loc_z[:, self.z_iext_dim: z_group_dim], scale_z[:,
                                                                               self.z_iext_dim: z_group_dim]
            z_rtpr = pyro.sample("z_rtpr_cls", dist.Normal(loc_z_rtpr, scale_z_rtpr).to_event(1))

            self.q_label(iext=iext, rtpr=rtpr, z_rtpr=z_rtpr, z_iext=z_iext)

    def q_label(self, iext, rtpr, z_iext, z_rtpr):
        alpha_iext = self.q_iext_given_z_iext.forward(z_iext)  # q(iext|z_iext)
        alpha_rtpr = self.q_rtpr_given_z_rtpr(z_rtpr)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample("iext_cls", dist.Bernoulli(alpha_iext).to_event(1), obs=iext)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample("rtpr_cls", dist.Bernoulli(alpha_rtpr).to_event(1), obs=rtpr)

    def guide_meta(self, observations, iext, rtpr):
        """
        dummy guide function to accompany model_meta in inference
        """
        pass

    def classifier(self, observations):
        """
        "Predict prob of iext"
        """
        loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)

        loc_z_iext, scale_z_iext = loc_z[:, 0:self.z_iext_dim], scale_z[:, 0:self.z_iext_dim]
        z_iext = torch.normal(loc_z_iext, scale_z_iext)
        alpha_iext = self.q_iext_given_z_iext(z_iext)
        pred_iext = (alpha_iext > 0.5).float()

        z_group_dim = self.z_iext_dim + self.z_rtpr_dim
        loc_z_rtpr, scale_z_rtpr = loc_z[:, self.z_iext_dim: z_group_dim
                                   ], scale_z[:, self.z_iext_dim: z_group_dim]
        z_rtpr = torch.normal(loc_z_rtpr, scale_z_rtpr)
        alpha_rtpr = self.q_rtpr_given_z_rtpr(z_rtpr)
        pred_rtpr = (alpha_rtpr > 0.5).float()

        return {'iext': pred_iext, 'rtpr': pred_rtpr}

    def recon(self, observations, iext, rtpr, is_post):
        if is_post:
            loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
            z = torch.normal(loc_z, scale_z)
        else:
            batch_size = observations.shape[0]
            z_epsilon_loc = torch.zeros((batch_size, self.z_epsilon_dim))
            z_epsilon_scale = torch.ones((batch_size, self.z_epsilon_dim))
            z_epsilon = torch.normal(z_epsilon_loc, z_epsilon_scale)

            z_iex_loc, z_iex_scale = self.p_z_iext_given_iext(iext)
            z_iext = torch.normal(z_iex_loc, z_iex_scale)

            z_rtpr_loc, z_rtpr_scale = self.p_z_rtprs_given_rtprs(rtpr)
            z_rtpr = torch.normal(z_rtpr_loc, z_rtpr_scale)

            z = torch.cat((z_iext, z_rtpr), dim=1)
            z = torch.cat((z, z_epsilon), dim=1)
            # import ipdb;
            # ipdb.set_trace()

        solution_xt, mu_75, mu_50, mu_25, std = self.decoder.forward(z=z)
        l1 = self.l1_func(mu_50, observations)
        results = {"l1": l1, "solution_xt": solution_xt, "mu_75": mu_75, "mu_50": mu_50, "mu_25": mu_25, "std": std,
                   "z": z}
        return results
