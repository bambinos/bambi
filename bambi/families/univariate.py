import numpy as np
import xarray as xr

from bambi.families.family import Family
from bambi.utils import get_aliased_name


class UnivariateFamily(Family):
    pass


class AsymmetricLaplace(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "b": ["log"],
        "kappa": ["log"],
        "q": ["logit", "probit", "cloglog"],
    }


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}

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

    @staticmethod
    def transform_backend_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        return kwargs


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}

    def posterior_predictive(self, model, posterior, **kwargs):
        data = kwargs["data"]

        if data is None:
            trials = model.response_component.response_term.data[:, 1]
        else:
            trials = model.response_component.design.response.evaluate_new_data(data)

        response_name = get_aliased_name(model.response_component.response_term)
        mean = posterior[response_name + "_mean"]
        return xr.apply_ufunc(np.random.binomial, trials.squeeze(), mean)

    @staticmethod
    def transform_backend_kwargs(kwargs):
        observed = kwargs.pop("observed")
        kwargs["observed"] = observed[:, 0].squeeze()
        kwargs["n"] = observed[:, 1].squeeze()
        return kwargs


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "alpha": ["log"]}

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
        # We build sigma from mu and alpha.
        alpha = kwargs.pop("alpha")
        kwargs["sigma"] = kwargs["mu"] / (alpha**0.5)
        return kwargs


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"]}


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "cloglog"], "alpha": ["log"]}


class Laplace(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "b": ["log"]}


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"]}


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"], "nu": ["log"]}


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "tan_2"], "kappa": ["log"]}


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["inverse", "inverse_squared", "identity", "log"], "lam": ["log"]}


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
