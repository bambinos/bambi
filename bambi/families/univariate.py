# pylint: disable=unused-argument
import numpy as np
from scipy import stats

from .family import Family


class UnivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        """Predict mean response"""
        mean = self.link.linkinv(linear_predictor)
        obs_n = mean.shape[-1]
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        dims = ("chain", "draw", response_dim)
        posterior[response_var] = (dims, mean)
        posterior = posterior.assign_coords({response_dim: list(range(obs_n))})
        return posterior


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = self.link.linkinv(linear_predictor)
        return np.random.binomial(1, mean)


class Beta(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        kappa = posterior[model.response.name + "_kappa"].values
        kappa = kappa[:, :, np.newaxis]
        alpha = mean * kappa
        beta = (1 - mean) * kappa
        return np.random.beta(alpha, beta)


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, trials=None):
        if trials is None:
            trials = model.response.data[:, 1]
        mean = self.link.linkinv(linear_predictor)
        return np.random.binomial(trials.squeeze(), mean)


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        alpha = posterior[model.response.name + "_alpha"].values
        alpha = alpha[:, :, np.newaxis]
        beta = alpha / mean
        return np.random.gamma(alpha, 1 / beta)


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = self.link.linkinv(linear_predictor)
        sigma = posterior[model.response.name + "_sigma"].values
        sigma = sigma[:, :, np.newaxis]
        return np.random.normal(mean, sigma)


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        n = posterior[model.response.name + "_alpha"].values
        n = n[:, :, np.newaxis]
        p = n / (mean + n)
        return np.random.negative_binomial(n, p)


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        return np.random.poisson(mean)


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        sigma = posterior[model.response.name + "_sigma"].values
        sigma = sigma[:, :, np.newaxis]

        if isinstance(self.likelihood.priors["nu"], (int, float)):
            nu = self.likelihood.priors["nu"]
        else:
            nu = posterior[model.response.name + "_nu"].values[:, :, np.newaxis]

        return stats.t.rvs(nu, mean, sigma)


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "tan_2"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        kappa = posterior[model.response.name + "_kappa"].values
        kappa = kappa[:, :, np.newaxis]
        return np.random.vonmises(mean, kappa)


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = ["inverse", "inverse_squared", "identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = self.link.linkinv(linear_predictor)
        lam = posterior[model.response.name + "_lam"].values
        lam = lam[:, :, np.newaxis]
        return np.random.wald(mean, lam)
