import numpy as np
import xarray as xr
import pytensor.tensor as pt

from bambi.families.family import Family
from bambi.utils import get_aliased_name


class UnivariateFamily(Family):
    KIND = "Univariate"


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


class Categorical(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    UFUNC_KWARGS = {"axis": -1}

    def transform_linear_predictor(self, model, linear_predictor):
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_dim"
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        return linear_predictor

    def transform_coords(self, model, mean):
        # The mean has the reference level in the dimension, a new name is needed
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_dim"
        response_levels_dim_complete = response_name + "_mean_dim"
        levels_complete = model.response_component.response_term.levels
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: levels_complete})
        return mean

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    def get_coords(self, response):
        name = response.name + "_dim"
        return {name: [level for level in response.levels if level != response.reference]}

    def get_reference(self, response):
        return get_reference_level(response.term)

    @staticmethod
    def transform_backend_nu(nu, data):
        # Add column of zeros to the linear predictor for the reference level (the first one)
        shape = (data.shape[0], 1)

        # The first line makes sure the intercept-only models work
        nu = np.ones(shape) * nu  # (response_levels, ) -> (n, response_levels)
        nu = pt.concatenate([np.zeros(shape), nu], axis=1)
        return nu


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


# pylint: disable = protected-access
def get_reference_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
