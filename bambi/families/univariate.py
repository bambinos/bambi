import numpy as np
import pytensor.tensor as pt
import scipy.special as sp
import xarray as xr

from bambi.families.family import Family
from bambi.utils import get_aliased_name


class UnivariateFamily(Family):
    KIND = "Univariate"


class BinomialBaseFamily(UnivariateFamily):
    def posterior_predictive(self, model, posterior, **kwargs):
        data = kwargs["data"]
        if data is None:
            trials = model.response_component.response_term.data[:, 1]
        else:
            trials = model.response_component.design.response.evaluate_new_data(data).astype(int)
        # Prepend 'draw' and 'chain' dimensions
        trials = trials[np.newaxis, np.newaxis, :]
        return super().posterior_predictive(model, posterior, n=trials)

    @staticmethod
    def transform_backend_kwargs(kwargs):
        observed = kwargs.pop("observed")
        kwargs["observed"] = observed[:, 0].squeeze()
        kwargs["n"] = observed[:, 1].squeeze()
        return kwargs


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
    """Beta Family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    SUPPORTED_LINKS = {"mu": ["logit", "probit", "cloglog"], "kappa": ["log"]}

    @staticmethod
    def transform_backend_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        return kwargs


class BetaBinomial(BinomialBaseFamily):
    """BetaBinomial family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    SUPPORTED_LINKS = {"mu": ["logit", "probit", "cloglog"], "kappa": ["log"]}

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # First, transform the parameters of the beta component
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        # Then transform the parameters of the binomial component
        return BinomialBaseFamily.transform_backend_kwargs(kwargs)

    @staticmethod
    def transform_kwargs(kwargs):
        # First, transform the parameters of the beta component
        mu = kwargs.pop("mu")
        kappa = kwargs.pop("kappa")
        kwargs["alpha"] = mu * kappa
        kwargs["beta"] = (1 - mu) * kappa
        return kwargs


class Binomial(BinomialBaseFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}


class Categorical(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    INVLINK_KWARGS = {"axis": -1}

    # pylint: disable = unused-argument
    @staticmethod
    def transform_linear_predictor(
        model, linear_predictor: xr.DataArray, posterior: xr.DataArray
    ) -> xr.DataArray:
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_reduced_dim"
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        return linear_predictor

    def transform_coords(self, model, mean):
        # The mean has the reference level in the dimension, a new name is needed
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_reduced_dim"
        response_levels_dim_complete = response_name + "_dim"
        levels_complete = model.response_component.response_term.levels
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: levels_complete})
        return mean

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    def get_coords(self, response):
        name = get_aliased_name(response) + "_reduced_dim"
        return {name: [level for level in response.levels if level != response.reference]}

    def get_reference(self, response):
        return get_reference_level(response.term)

    @staticmethod
    def transform_backend_eta(eta, kwargs):
        data = kwargs["observed"]

        # Add column of zeros to the linear predictor for the reference level (the first one)
        shape = (data.shape[0], 1)

        # The first line makes sure the intercept-only models work
        eta = np.ones(shape) * eta  # (response_levels, ) -> (n, response_levels)
        eta = pt.concatenate([np.zeros(shape), eta], axis=1)
        return eta


class Cumulative(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["logit", "probit", "cloglog"], "threshold": ["identity"]}

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    @staticmethod
    def transform_linear_predictor(
        model, linear_predictor: xr.DataArray, posterior: xr.DataArray
    ) -> xr.DataArray:
        """Computes threshold_k - eta"""
        threshold_component = model.components["threshold"]
        response_name = get_aliased_name(model.response_component.response_term)
        if threshold_component.alias:
            threshold_name = threshold_component.alias
        else:
            threshold_name = f"{response_name}_threshold"
        threshold = posterior[threshold_name]
        return threshold - linear_predictor

    @staticmethod
    def transform_mean(model, mean: xr.DataArray) -> xr.DataArray:
        """Computes P(Y = k) = F(threshold_k - eta) - F(threshold_{k - 1} - eta)"""
        threshold_component = model.components["threshold"]
        response_name = get_aliased_name(model.response_component.response_term)
        if threshold_component.alias:
            threshold_name = threshold_component.alias
        else:
            threshold_name = response_name + "_threshold"
        threshold_dim = threshold_name + "_dim"
        response_dim = response_name + "_dim"
        mean = xr.concat(
            [
                mean.isel({threshold_dim: 0}),
                mean.diff(threshold_dim),
                1 - mean.isel({threshold_dim: -1}),
            ],
            dim=threshold_dim,
        )
        mean = mean.rename({threshold_dim: response_dim})
        mean = mean.assign_coords({response_dim: model.response_component.response_term.levels})
        mean = mean.transpose(..., response_dim)  # make sure response levels is the last dim
        return mean

    @staticmethod
    def transform_backend_eta(eta, kwargs):
        # shape(threshold) = (K, )
        # shape(eta) = (n, )
        # shape(threshold - shape_padright(eta)) = (n, K)
        threshold = kwargs["threshold"]
        eta_shifted = threshold - pt.shape_padright(eta)
        return eta_shifted

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # P(Y = k) = F(threshold_k - eta) - F(threshold_{k - 1} - eta)
        p = kwargs.pop("p")
        p = pt.concatenate(
            [
                pt.shape_padright(p[..., 0]),
                p[..., 1:] - p[..., :-1],
                pt.shape_padright(1 - p[..., -1]),
            ],
            axis=-1,
        )
        kwargs["p"] = p
        kwargs.pop("threshold", None)  # this is not passed to the likelihood function
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        kwargs.pop("threshold", None)  # this is not passed to the likelihood function
        return kwargs


class Exponential(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"]}

    @staticmethod
    def transform_backend_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kwargs["lam"] = 1 / mu
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        mu = kwargs.pop("mu")
        kwargs["lam"] = 1 / mu
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

    @staticmethod
    def transform_kwargs(kwargs):
        alpha = kwargs.pop("alpha")
        kwargs["sigma"] = kwargs["mu"] / (alpha**0.5)
        return kwargs


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"]}


class HurdleGamma(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }

    @staticmethod
    def transform_backend_kwargs(kwargs):
        alpha = kwargs.pop("alpha")
        kwargs["sigma"] = kwargs["mu"] / (alpha**0.5)
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        alpha = kwargs.pop("alpha")
        kwargs["sigma"] = kwargs["mu"] / (alpha**0.5)
        return kwargs


class HurdleLogNormal(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "sigma": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class HurdleNegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "cloglog"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class HurdlePoisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"], "psi": ["logit", "probit", "cloglog"]}


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "cloglog"], "alpha": ["log"]}


class Laplace(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "b": ["log"]}


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"]}


class StoppingRatio(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["logit", "probit", "cloglog"], "threshold": ["identity"]}

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    @staticmethod
    def transform_linear_predictor(
        model, linear_predictor: xr.DataArray, posterior: xr.DataArray
    ) -> xr.DataArray:
        """Computes threshold_k - eta"""
        threshold_component = model.components["threshold"]
        response_name = get_aliased_name(model.response_component.response_term)
        if threshold_component.alias:
            threshold_name = threshold_component.alias
        else:
            threshold_name = f"{response_name}_threshold"
        threshold = posterior[threshold_name]
        return threshold - linear_predictor

    @staticmethod
    def transform_mean(model, mean: xr.DataArray) -> xr.DataArray:
        """Computes P(Y = k) = F(threshold_k - eta) - F(threshold_{k - 1} - eta)"""
        threshold_component = model.components["threshold"]
        response_name = get_aliased_name(model.response_component.response_term)
        if threshold_component.alias:
            threshold_name = threshold_component.alias
        else:
            threshold_name = response_name + "_threshold"
        threshold_dim = threshold_name + "_dim"
        response_dim = response_name + "_dim"
        threshold_n = len(mean[threshold_dim])

        # the `.assign_coords`` is needed for the concat to work
        mean = xr.concat(
            [
                mean.isel({threshold_dim: 0}),
                *[
                    (
                        mean.isel({threshold_dim: j})
                        * (1 - mean).isel({threshold_dim: slice(None, j)}).prod(threshold_dim)
                    )
                    for j in range(1, threshold_n)
                ],
                (1 - mean).prod(threshold_dim).assign_coords({threshold_dim: threshold_n + 1}),
            ],
            dim=threshold_dim,
        )
        mean = mean.rename({threshold_dim: response_dim})
        mean = mean.assign_coords({response_dim: model.response_component.response_term.levels})
        mean = mean.transpose(..., response_dim)  # make sure response levels is the last dim
        return mean

    @staticmethod
    def transform_backend_eta(eta, kwargs):
        # shape(threshold) = (K, )
        # shape(eta) = (n, )
        # shape(threshold - shape_padright(eta)) = (n, K)
        threshold = kwargs["threshold"]
        eta_shifted = threshold - pt.shape_padright(eta)
        return eta_shifted

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # P(Y = k) = F(threshold_k - eta) * \prod_{j=1}^{k-1}{1 - F(threshold_j - eta)}
        p = kwargs.pop("p")
        n_columns = p.type.shape[-1]
        p = pt.concatenate(
            [
                pt.shape_padright(p[..., 0]),
                *[
                    pt.shape_padright(p[..., j] * pt.prod(1 - p[..., :j], axis=-1))
                    for j in range(1, n_columns)
                ],
                pt.shape_padright(pt.prod(1 - p, axis=-1)),
            ],
            axis=-1,
        )
        kwargs["p"] = p
        kwargs.pop("threshold", None)  # this is not passed to the likelihood function
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        kwargs.pop("threshold", None)  # this is not passed to the likelihood function
        return kwargs


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"], "nu": ["log"]}


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "tan_2"], "kappa": ["log"]}


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["inverse", "inverse_squared", "identity", "log"], "lam": ["log"]}


class Weibull(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["log", "identity", "inverse"], "alpha": ["log"]}

    @staticmethod
    def transform_backend_kwargs(kwargs):
        # The Weibull distribution is specified using alpha (shape) and beta (scale).
        # We request a prior for alpha and we model 'mu' as a function of the linear predictor.
        # Here we determine 'beta' out of the value of 'mu' and 'alpha'
        mu = kwargs.pop("mu")
        alpha = kwargs.get("alpha")
        kwargs["beta"] = mu / pt.gamma(1 + 1 / alpha)
        return kwargs

    @staticmethod
    def transform_kwargs(kwargs):
        mu = kwargs.pop("mu")
        alpha = kwargs.get("alpha")
        kwargs["beta"] = mu / sp.gamma(1 + 1 / alpha)
        return kwargs


class ZeroInflatedBinomial(BinomialBaseFamily):
    SUPPORTED_LINKS = {
        "p": ["identity", "logit", "probit", "cloglog"],
        "psi": ["logit", "probit", "cloglog"],
    }


class ZeroInflatedNegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "cloglog"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class ZeroInflatedPoisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"], "psi": ["logit", "probit", "cloglog"]}


# pylint: disable = protected-access
def get_success_level(term):
    """Returns the success level of a categorical term.

    Whenever the concept of "success level" does not apply, it returns None.
    """
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
    """Returns the reference level of a categorical term.

    Whenever the concept of "reference level" does not apply, it returns None.
    """
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
