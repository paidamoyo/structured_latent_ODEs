import pyro
import torch.nn as nn
import pyro.distributions as dist
import torch

from models.encoder_conv import EncoderCONV
from models.encoder_mlp import EncoderMLP
from models.decoders import GaussianDecoder
from utils.exp import Exp


class MechanisticModelGauss(nn.Module):
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
        print("n_time: ", self.n_time)
        self.n_filters = config.n_filters
        self.filter_size = config.filter_size
        self.pool_size = config.pool_size
        self.cnn_hidden_dim = config.cnn_hidden_dim
        self.n_channels = self.obs_dim

        ## System input Dimensions
        self.shedding_dim = config.shedding_dim
        self.symptoms_dim = config.symptoms_dim

        self.aux_loss_multiplier = config.aux_loss_multiplier
        self.condition_on_device = False

        ## Latent Dim
        self.z_shedding_dim = config.z_shedding_dim
        self.z_symptoms_dim = config.z_symptoms_dim
        self.z_epsilon_dim = config.z_epsilon_dim
        self.latent_dim = self.z_shedding_dim + self.z_symptoms_dim + self.z_epsilon_dim

        self.device = device
        self.use_cuda = device != "cpu"
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

        # Posterior over (classifier)
        self.q_shedding_given_z_shedding = EncoderMLP(
            mlp_sizes=[self.z_shedding_dim] + [self.u_hidden_dim] + [self.shedding_dim],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # Posterior over (classifier)
        self.q_symptom_given_z_symptom = EncoderMLP(
            mlp_sizes=[self.z_symptoms_dim] + [self.u_hidden_dim] + [self.symptoms_dim],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # Posterior over z
        self.encoder = EncoderCONV(
            n_channels=self.n_channels,
            n_time=self.n_time,
            n_filters=self.n_filters,
            filter_size=self.filter_size,
            pool_size=self.pool_size,
            latent_dim=self.latent_dim,
            hidden_dim=self.cnn_hidden_dim,
        )

        system_input_dim = self.shedding_dim + self.symptoms_dim
        self.z_u_dim = self.z_shedding_dim * 2
        self.p_z_u_given_u = EncoderMLP(
            mlp_sizes=[system_input_dim] + [[self.z_u_dim, self.z_u_dim]],
            activation=nn.Softplus,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # Likelihood over observations
        self.decoder = GaussianDecoder(
            config=self.config,
            times=self.times,
            latent_dim=self.latent_dim,
            device=self.device,
        )

        self.softplus = torch.nn.Softplus()

    def model(self, observations, symptoms, shedding):
        """
        The model corresponds to the following generative process:
        p(x, u, z) = p(u)p(z|u)p(x|z) since x || u |z #
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
            z = self.get_prior_z(symptoms=symptoms, shedding=shedding)
            # import ipdb;
            # ipdb.set_trace()
            solution_xt, mean, std = self.decoder.forward(z=z)
            # pyro.sample("y", dist.Normal(mean, std).to_event(1), obs=observations)
            for id in range(observations.shape[1]):
                pyro.sample(
                    f"y_{id}",
                    dist.Normal(mean[:, id, :], std[:, id, :]).to_event(1),
                    obs=observations[:, id, :],
                )

        z = self.split_z_prior(z)
        # self.q_label(shedding=shedding, symptoms=symptoms, z_shedding=z['z_shedding'], z_symptoms=z['z_symptoms'],
        #              name='p')
        # import ipdb;
        # ipdb.set_trace()
        l1_loss = self.l1_func(observations, mean)
        return l1_loss

    def get_prior_z(self, symptoms, shedding):
        batch_size = symptoms.shape[0]
        options = dict(dtype=symptoms.dtype, device=self.device)
        z_epsilon_loc = torch.zeros(batch_size, self.z_epsilon_dim, **options)
        z_epsilon_scale = torch.ones(batch_size, self.z_epsilon_dim, **options)

        system_inputs = torch.cat((symptoms, shedding), dim=1)

        z_u_loc, z_u_scale = self.p_z_u_given_u.forward(system_inputs)  # p(z_u|u)
        z_u = pyro.sample("z_u", dist.Normal(z_u_loc, z_u_scale).to_event(1))
        z_epsilon = pyro.sample(
            "z_epsilon", dist.Normal(z_epsilon_loc, z_epsilon_scale).to_event(1)
        )  # p(z_epsilon)
        # pass as tensor z
        z = torch.cat((z_u, z_epsilon), dim=1)
        return z

    def compute_likelihood(self, get_series, mu, std, tau):
        with pyro.poutine.scale(scale=1 - tau):
            # std_tau = std / tau
            std_tau = std
            x_greater_mu = 0
            x0_data = get_series(
                id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x1_data = get_series(
                id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x2_data = get_series(
                id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x3_data = get_series(
                id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )

            pyro.sample(
                "x_0_{}_g".format(tau),
                dist.Laplace(x0_data[0], x0_data[1]).to_event(1),
                obs=x0_data[2],
            )
            pyro.sample(
                "x_1_{}_g".format(tau),
                dist.Laplace(x1_data[0], x1_data[1]).to_event(1),
                obs=x1_data[2],
            )
            pyro.sample(
                "x_2_{}_g".format(tau),
                dist.Laplace(x2_data[0], x2_data[1]).to_event(1),
                obs=x2_data[2],
            )
            pyro.sample(
                "x_3_{}_g".format(tau),
                dist.Laplace(x3_data[0], x3_data[1]).to_event(1),
                obs=x3_data[2],
            )

        with pyro.poutine.scale(scale=tau):
            # actual >= pred then scale with tau
            # P(actual < pred) = tau, where pred is the t-th quantile
            # std_tau = std / (1 - tau)
            std_tau = std
            x_greater_mu = 1
            x0_data = get_series(
                id=0, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x1_data = get_series(
                id=1, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x2_data = get_series(
                id=2, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )
            x3_data = get_series(
                id=3, mu_pred=mu, std_pred=std_tau, x_greater_mu=x_greater_mu
            )

            pyro.sample(
                "x_0_{}_l".format(tau),
                dist.Laplace(x0_data[0], x0_data[1]).to_event(1),
                obs=x0_data[2],
            )
            pyro.sample(
                "x_1_{}_l".format(tau),
                dist.Laplace(x1_data[0], x1_data[1]).to_event(1),
                obs=x1_data[2],
            )
            pyro.sample(
                "x_2_{}_l".format(tau),
                dist.Laplace(x2_data[0], x2_data[1]).to_event(1),
                obs=x2_data[2],
            )
            pyro.sample(
                "x_3_{}_l".format(tau),
                dist.Laplace(x3_data[0], x3_data[1]).to_event(1),
                obs=x3_data[2],
            )

        return x0_data, x1_data, x2_data

    def guide(self, observations, symptoms, shedding):
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
            z_shedding, z_symptoms, z_epsilon, z_u = (
                z["z_shedding"],
                z["z_symptoms"],
                z["z_epsilon"],
                z["z_u"],
            )

            z_u = pyro.sample("z_u", dist.Normal(z_u[0], z_u[1]).to_event(1))
            z_epsilon = pyro.sample(
                "z_epsilon", dist.Normal(z_epsilon[0], z_epsilon[1]).to_event(1)
            )
            return z_shedding, z_symptoms, z_epsilon

    def split_z_prior(self, z):
        z_shedding = z[:, 0 : self.z_shedding_dim]

        z_group_dim = self.z_shedding_dim + self.z_symptoms_dim
        z_symptoms = z[:, self.z_shedding_dim : z_group_dim]
        return {
            "z_shedding": z_shedding,
            "z_symptoms": z_symptoms,
        }

    def split_z(self, loc_z, scale_z):
        # z_shedding, z_symptoms, z_epsilon, z_u
        loc_z_shedding, scale_z_shedding = (
            loc_z[:, 0 : self.z_shedding_dim],
            scale_z[:, 0 : self.z_shedding_dim],
        )

        z_group_dim = self.z_shedding_dim + self.z_symptoms_dim
        loc_z_symptoms, scale_z_symptoms = (
            loc_z[:, self.z_shedding_dim : z_group_dim],
            scale_z[:, self.z_shedding_dim : z_group_dim],
        )

        loc_z_epsilon, scale_z_epsilon = (
            loc_z[:, -self.z_epsilon_dim :],
            scale_z[:, -self.z_epsilon_dim :],
        )

        z_u_loc = torch.cat((loc_z_shedding, loc_z_symptoms), dim=1)

        z_u_scale = torch.cat((scale_z_shedding, scale_z_symptoms), dim=1)

        return {
            "z_shedding": [loc_z_shedding, scale_z_shedding],
            "z_symptoms": [loc_z_symptoms, scale_z_symptoms],
            "z_epsilon": [loc_z_epsilon, scale_z_epsilon],
            "z_u": [z_u_loc, z_u_scale],
        }

    def model_meta(self, observations, shedding, symptoms):
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
            z_shedding, z_symptoms, z_epsilon, z_u = (
                z["z_shedding"],
                z["z_symptoms"],
                z["z_epsilon"],
                z["z_u"],
            )

            z_shedding = pyro.sample(
                "z_shedding_u", dist.Normal(z_shedding[0], z_shedding[1]).to_event(1)
            )
            z_symptoms = pyro.sample(
                "z_symptoms_u", dist.Normal(z_symptoms[0], z_symptoms[1]).to_event(1)
            )

            self.q_label(
                shedding=shedding,
                symptoms=symptoms,
                z_symptoms=z_symptoms,
                z_shedding=z_shedding,
                name="q",
            )

    def q_label(self, shedding, symptoms, z_shedding, z_symptoms, name):
        alpha_shedding = self.q_shedding_given_z_shedding.forward(
            z_shedding
        )  # q(shedding|z_shedding)
        alpha_symptoms = self.q_symptom_given_z_symptom.forward(z_symptoms)

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample(
                "shedding_u_" + name,
                dist.Bernoulli(alpha_shedding).to_event(1),
                obs=shedding,
            )

        with pyro.poutine.scale(scale=self.aux_loss_multiplier):
            pyro.sample(
                "symptoms _u_" + name,
                dist.Bernoulli(alpha_symptoms).to_event(1),
                obs=symptoms,
            )

    def guide_meta(self, observations, shedding, symptoms):
        """
        dummy guide function to accompany model_meta in inference
        """
        pass

    def pred_inputs(self, observations):
        loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
        z = self.split_z(loc_z=loc_z, scale_z=scale_z)
        z_shedding, z_symptoms, z_epsilon, z_u = (
            z["z_shedding"],
            z["z_symptoms"],
            z["z_epsilon"],
            z["z_u"],
        )

        z_shedding_sample = torch.normal(z_shedding[0], z_shedding[1])
        alpha_shedding = self.q_shedding_given_z_shedding(z_shedding_sample)
        pred_shedding = (alpha_shedding > 0.5).float()

        z_symptoms_sample = torch.normal(z_symptoms[0], z_symptoms[1])
        alpha_symptoms = self.q_symptom_given_z_symptom(z_symptoms_sample)
        pred_symptoms = (alpha_symptoms > 0.5).float()

        return {"shedding": pred_shedding, "symptoms": pred_symptoms}

    def recon(self, observations, shedding, symptoms, is_post):
        if is_post:
            loc_z, scale_z = self.encoder.forward(observations)  # q(z|x)
            z = torch.normal(loc_z, scale_z)
        else:
            z = self.get_prior_z(symptoms=symptoms, shedding=shedding)
            # import ipdb;
            # ipdb.set_trace()

        solution_xt, mean, std = self.decoder.forward(z=z)
        mu_75 = mean + 2 * std
        mu_25 = mean - 2 * std
        l1 = self.l1_func(mean, observations)
        results = {
            "l1": l1,
            "solution_xt": solution_xt,
            "mu_75": mu_75,
            "mu_50": mean,
            "mu_25": mu_25,
            "std": std,
            "z": z,
        }
        return results
