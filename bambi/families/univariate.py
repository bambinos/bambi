# pylint: disable=unused-argument
import numpy as np
from scipy import stats

from .family import Family


class UnivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        """Predict mean response"""
        mean = self.link.linkinv(linear_predictor)
        obs_n = mean.shape[-1]
        name = model.response.name + "_mean"
        coord_name = name + "_obs"

        # Drop var/dim if already present
        if name in posterior.data_vars:
            posterior = posterior.drop_vars(name).drop_dims(coord_name)

        coords = ("chain", "draw", coord_name)
        posterior[name] = (coords, mean)
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        "Sample from posterior predictive distribution"
        mean = self.link.linkinv(linear_predictor)
        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        return np.random.binomial(1, mean)


class Beta(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        kappa = posterior[model.response.name + "_kappa"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        kappa = kappa[:, idxs, np.newaxis]

        alpha = mean * kappa
        beta = (1 - mean) * kappa
        return np.random.beta(alpha, beta)


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "logit", "probit", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n, trials=None):
        if trials is None:
            trials = model.response.data[:, 1]

        mean = self.link.linkinv(linear_predictor)
        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        np.random.binomial(trials.squeeze(), mean)


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        alpha = posterior[model.response.name + "_alpha"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        alpha = alpha[:, idxs, np.newaxis]
        beta = alpha / mean
        return np.random.gamma(alpha, 1 / beta)


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        "Sample from posterior predictive distribution"
        mean = self.link.linkinv(linear_predictor)
        sigma = posterior[model.response.name + "_sigma"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        sigma = sigma[:, idxs, np.newaxis]

        return np.random.normal(mean, sigma)


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "cloglog"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        n = posterior[model.response.name + "_alpha"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        n = n[:, idxs, np.newaxis]
        p = n / (mean + n)

        return np.random.negative_binomial(n, p)


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        return np.random.poisson(mean)


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = ["identity", "log", "inverse"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        sigma = posterior[model.response.name + "_sigma"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        if isinstance(self.likelihood.priors["nu"], (int, float)):
            nu = self.likelihood.priors["nu"]
        else:
            nu = posterior[model.response.name + "_nu"].values[:, idxs, np.newaxis]

        mean = mean[:, idxs, :]
        sigma = sigma[:, idxs, np.newaxis]

        return stats.t.rvs(nu, mean, sigma)


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = ["inverse", "inverse_squared", "identity", "log"]

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        mean = self.link.linkinv(linear_predictor)
        lam = posterior[model.response.name + "_lam"].values

        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :]
        lam = lam[:, idxs, np.newaxis]

        return np.random.wald(mean, lam)
