# pylint: disable=unused-argument
import numpy as np
import xarray as xr
from scipy import stats

from bambi.families.family import Family


class UnivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        return NotImplemented


class AsymmetricLaplace(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "b": ["log"],
        "q": ["logit", "probit", "cloglog"],
    }

    def posterior_predictive(self, model, posterior):
        "Sample from posterior predictive distribution"
        mean = posterior[model.response_name + "_mean"]
        b = posterior[model.response_name + "_b"]
        kappa = posterior[model.response_name + "_kappa"]
        return xr.apply_ufunc(stats.laplace_asymmetric, kappa, mean, b)


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}

    def posterior_predictive(self, model, posterior):
        "Sample from posterior predictive distribution"
        mean = posterior[model.response_name + "_mean"]
        return xr.apply_ufunc(np.random.binomial, 1, mean)

    def get_data(self, response):
        if response.term.data.ndim == 1:
            return response.term.data
        idx = response.levels.index(response.success)
        return response.term.data[:, idx]

    def get_success_level(self, response):
        if response.categorical:
            return get_success_level(response.term)
        return 1


class Beta(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["logit", "probit", "cloglog"], "kappa": ["log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        kappa = posterior[model.response_name + "_kappa"]
        alpha = mean * kappa
        beta = (1 - mean) * kappa
        return xr.apply_ufunc(np.random.beta, alpha, beta)

    @staticmethod
    def transform_backend_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        return kwargs


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}

    def posterior_predictive(self, model, posterior, trials=None):
        if trials is None:
            trials = model.response_component.response_term.data[:, 1]
        mean = posterior[model.response_name + "_mean"]
        return xr.apply_ufunc(np.random.binomial, trials.squeeze(), mean)

    @staticmethod
    def transform_backend_kwargs(kwargs):
        observed = kwargs.pop("observed")
        kwargs["observed"] = observed[:, 0].squeeze()
        kwargs["n"] = observed[:, 1].squeeze()
        return kwargs


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "alpha": ["log"]}

    def posterior_predictive(self, model, posterior, linear_predictor):
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor)
        alpha = posterior[model.response_name + "_alpha"]
        beta = alpha / mean
        return xr.apply_ufunc(np.random.gamma, alpha, 1 / beta)

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
        # We build sigma from mu and alpha.
        alpha = kwargs.pop("alpha")
        kwargs["sigma"] = kwargs["mu"] / (alpha**0.5)
        return kwargs


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"]}

    def posterior_predictive(self, model, posterior):
        "Sample from posterior predictive distribution"
        mean = posterior[model.response_name + "_mean"]
        sigma = posterior[model.response_name + "_sigma"]
        return xr.apply_ufunc(np.random.normal, mean, sigma)


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "cloglog"], "alpha": ["log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        n = posterior[model.response_name + "_alpha"]
        p = n / (mean + n)
        return xr.apply_ufunc(np.random.negative_binomial, n, p)


class Laplace(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "b": ["log"]}

    def posterior_predictive(self, model, posterior, linear_predictor):
        "Sample from posterior predictive distribution"
        mean = posterior[model.response_name + "_mean"]
        b = posterior[model.response_name + "_b"]
        return xr.apply_ufunc(np.random.laplace, mean, b)


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        return xr.apply_ufunc(np.random.poisson, mean)


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"], "nu": ["log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        sigma = posterior[model.response_name + "_sigma"]
        nu_component = model.components["nu"]

        # Constant component with fixed value
        if hasattr(nu_component, "prior") and isinstance(nu_component.prior, (int, float)):
            nu = nu_component.prior
        # Either constant or distributional, but non-constant value
        else:
            nu = posterior[model.response_name + "_nu"]

        return xr.apply_ufunc(stats.t.rvs, nu, mean, sigma)


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "tan_2"], "kappa": ["log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        kappa = posterior[model.response_name + "_kappa"]
        return xr.apply_ufunc(np.random.vonmises, mean, kappa)


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["inverse", "inverse_squared", "identity", "log"], "lam": ["log"]}

    def posterior_predictive(self, model, posterior):
        mean = posterior[model.response_name + "_mean"]
        lam = posterior[model.response_name + "_lam"]
        return xr.apply_ufunc(np.random.wald, mean, lam)


# pylint: disable = protected-access
def get_success_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return term.components[0].reference

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
