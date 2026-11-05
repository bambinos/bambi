import warnings

import numpy as np
import scipy.special as sp
import xarray as xr

from bambi.families.family import Family
from bambi.families.types import DimType, ResponseType, ParamSpec
from bambi.transformations import transformations_namespace
from bambi.utils import extract_argument_names


class AdjacentCategory(Family):
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit", "probit", "cloglog"], ndim=1),
        "threshold": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS),
    }


class AsymmetricLaplace(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "b": ParamSpec(links=["log", "logexpm1"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
        "q": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mu, b, kappa = params["mu"], params["b"], params["kappa"]
        mean = mu - (kappa - 1 / kappa) / b
        variance = (1 + kappa**4) / (b**2 * kappa**2)
        return mean, variance


class Bernoulli(Family):
    DATA_TYPE = ResponseType.BINARY
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        p = params["p"]
        return p, p * (1 - p)


class Beta(Family):
    """Beta family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit", "probit", "cloglog"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        mu, kappa = params["mu"], params["kappa"]
        return mu, _beta_variance(mu, kappa)


class BetaBinomial(Family):
    """BetaBinomial family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit", "probit", "cloglog"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        mu, kappa, n = params["mu"], params["kappa"], params["n"]
        return n * mu, n * mu * (1 - mu) * (n + kappa) / (1 + kappa)


class Binomial(Family):
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        return _binomial_moments(params["p"], params["n"])


class Categorical(Family):
    DATA_TYPE = ResponseType.CATEGORICAL
    PARAMETERS = {
        "p": ParamSpec(links=["softmax"], ndim=1, coefs_dim=DimType.RESPONSE_REDUCED),
    }


class ContinuationRatio(Family):
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit", "probit", "cloglog"], ndim=1),
        "threshold": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS),
    }


class Cumulative(Family):
    # There's a single linear predictor, not as many linear predictors as response levels.
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit", "probit", "cloglog"], ndim=1),
        "threshold": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS),
    }


class ExGaussian(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "sigma": ParamSpec(links=["log", "logexpm1"]),
        "nu": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        mu, sigma, nu = params["mu"], params["sigma"], params["nu"]
        return mu + nu, sigma**2 + nu**2


class Exponential(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
    }

    def moments(self, params):
        mu = params["mu"]
        return mu, mu**2


class Gamma(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        return _gamma_moments(params["mu"], params["alpha"])


class Gaussian(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "sigma": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        return params["mu"], params["sigma"] ** 2


class HurdleGamma(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mean, variance = _gamma_moments(params["mu"], params["alpha"])
        return _hurdle_moments(mean, variance, params["psi"])


class HurdleLogNormal(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "sigma": ParamSpec(links=["log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mean, variance = _lognormal_moments(params["mu"], params["sigma"])
        return _hurdle_moments(mean, variance, params["psi"])


class HurdleNegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "cloglog"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mu, alpha = params["mu"], params["alpha"]
        mean, variance = _negativebinomial_moments(mu, alpha)
        p_zero = (alpha / (alpha + mu)) ** alpha
        return _hurdle_moments(mean, variance, params["psi"], p_zero)


class HurdlePoisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mu = params["mu"]
        return _hurdle_moments(mu, mu, params["psi"], np.exp(-mu))


class NegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "cloglog"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        return _negativebinomial_moments(params["mu"], params["alpha"])


class Laplace(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "b": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        return params["mu"], 2 * params["b"] ** 2


class LogNormal(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "sigma": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        return _lognormal_moments(params["mu"], params["sigma"])


class LogLogistic(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        # log(Y) is logistic with location 'mu' and scale 'alpha'. The mean is finite only when
        # alpha < 1 and the variance only when alpha < 1/2.
        mu, alpha = params["mu"], params["alpha"]
        _warn_infinite_variance(alpha >= 0.5, "alpha", ">= 0.5", "log-logistic")
        b = np.pi * alpha
        scale = np.exp(mu)
        mean = xr.where(alpha < 1, scale * b / np.sin(b), np.inf)
        variance = xr.where(
            alpha < 0.5, scale**2 * (2 * b / np.sin(2 * b) - (b / np.sin(b)) ** 2), np.inf
        )
        return mean, variance


class OrderedBeta(Family):
    """Ordered beta family

    Beta interior (of `mu` and `kappa`), with additional boundary cutpoints
    of `cut0` and `cut0 + exp(logdelta)`, as in ordbetareg (Kubinec, 2023).
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
        "cut0": ParamSpec(links=["identity"]),
        "logdelta": ParamSpec(links=["identity"]),
    }

    def moments(self, params):
        mu, kappa = params["mu"], params["kappa"]
        cut0 = params["cut0"]
        cut1 = cut0 + np.exp(params["logdelta"])
        eta = np.log(mu) - np.log1p(-mu)
        p_zero = _sigmoid(cut0 - eta)
        p_one = _sigmoid(eta - cut1)
        p_interior = 1 - p_zero - p_one
        mean = p_one + p_interior * mu
        second_moment = p_one + p_interior * (_beta_variance(mu, kappa) + mu**2)
        return mean, second_moment - mean**2


class OrderedStereotype(Family):
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit"], ndim=1),
        "alpha": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_REDUCED),
        "delta": ParamSpec(links=["softmax"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS_REDUCED),
    }


class Poisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1"]),
    }

    def moments(self, params):
        return params["mu"], params["mu"]


class StoppingRatio(Family):
    # There's a single linear predictor, not as many linear predictors as response levels.
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit", "probit", "cloglog"], ndim=1),
        "threshold": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS),
    }


class StudentT(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "inverse"]),
        "sigma": ParamSpec(links=["log", "logexpm1"]),
        "nu": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        mu, sigma, nu = params["mu"], params["sigma"], params["nu"]
        # The variance is infinite when nu <= 2 (and the mean is undefined when nu <= 1).
        # Those draws get an infinite variance (`Model.r2_score` then uses the residual R^2).
        _warn_infinite_variance(nu <= 2, "nu", "<= 2", "Student's t")
        return mu, xr.where(nu > 2, sigma**2 * nu / (nu - 2), np.inf)


class VonMises(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
    }

    CIRCULAR = True

    def moments(self, params):
        # The circular variance of the von Mises distribution is 1 - I1(kappa) / I0(kappa).
        # The exponentially scaled Bessel functions avoid overflow for large kappa.
        mu, kappa = params["mu"], params["kappa"]
        bessel_ratio = xr.apply_ufunc(sp.i1e, kappa) / xr.apply_ufunc(  # pylint: disable=no-member
            sp.i0e, kappa  # pylint: disable=no-member
        )
        return mu, 1 - bessel_ratio


class Wald(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["inverse", "inverse_squared", "identity", "log", "logexpm1"]),
        "lam": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        mu, lam = params["mu"], params["lam"]
        return mu, mu**3 / lam


class Weibull(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["log", "logexpm1", "identity", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        # 'mu' is the mean, the scale is 'beta = mu / gamma(1 + 1 / alpha)'
        mu, alpha = params["mu"], params["alpha"]
        return _weibull_moments(mu / sp.gamma(1 + 1 / alpha), alpha)


class WeibullPH(Family):
    """Weibull family with a proportional-hazards parametrization.

    The survival function is `S(t) = exp(-lam * t**alpha)` and the hazard is
    `h(t) = alpha * lam * t**(alpha - 1)`.
    With a log link for `lam` and a shared shape `alpha`,
    exponentiated regression coefficients are hazard ratios.
    """

    PARAMETERS = {
        "lam": ParamSpec(links=["log", "logexpm1", "identity", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
    }

    def moments(self, params):
        alpha = params["alpha"]
        return _weibull_moments(params["lam"] ** (-1 / alpha), alpha)


class CureExponential(Family):
    PARAMETERS = {
        **Exponential.PARAMETERS,
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class CureGamma(Family):
    PARAMETERS = {
        **Gamma.PARAMETERS,
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class CureLogNormal(Family):
    PARAMETERS = {
        **LogNormal.PARAMETERS,
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class CureLogLogistic(Family):
    PARAMETERS = {
        **LogLogistic.PARAMETERS,
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class CureWeibullPH(Family):
    PARAMETERS = {
        **WeibullPH.PARAMETERS,
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class CureWeibull(Family):
    """Weibull mixture-cure family.

    `mu` is the mean event time among susceptible subjects, `alpha` is their Weibull shape,
    and `cure` is the probability of never experiencing the event.
    The overall survival is `cure + (1 - cure) * S_u(t)`, where S_u is the survival of the uncured.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["log", "logexpm1", "identity", "inverse"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
        "cure": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):  # pylint: disable=unused-argument
        raise _cure_moments_error(self.name)


class ZeroInflatedBinomial(Family):
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mean, variance = _binomial_moments(params["p"], params["n"])
        return _zero_inflated_moments(mean, variance, params["psi"])


class ZeroInflatedNegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1", "cloglog"]),
        "alpha": ParamSpec(links=["log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mean, variance = _negativebinomial_moments(params["mu"], params["alpha"])
        return _zero_inflated_moments(mean, variance, params["psi"])


class ZeroInflatedPoisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "logexpm1"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mu = params["mu"]
        return _zero_inflated_moments(mu, mu, params["psi"])


class ZeroOneInflatedBeta(Family):
    """Zero-one-inflated beta family

    Beta interior (of `mu` and `kappa`), with additional inflation parameters:
    `zoi` is the probability of a 0 or 1 and `coi` the probability that it is 1.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit", "probit", "cloglog"]),
        "kappa": ParamSpec(links=["log", "logexpm1"]),
        "zoi": ParamSpec(links=["logit", "probit", "cloglog"]),
        "coi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }

    def moments(self, params):
        mu, kappa, zoi, coi = params["mu"], params["kappa"], params["zoi"], params["coi"]
        mean = zoi * coi + (1 - zoi) * mu
        second_moment = zoi * coi + (1 - zoi) * (_beta_variance(mu, kappa) + mu**2)
        return mean, second_moment - mean**2


class Multinomial(Family):
    RESPONSE_NDIM = 1
    PARAMETERS = {
        "p": ParamSpec(links=["softmax"], ndim=1, coefs_dim=DimType.RESPONSE_REDUCED),
    }

    def get_levels(self, response):
        labels = extract_argument_names(response.full_name, list(transformations_namespace))
        if labels:
            return labels
        return [str(level) for level in range(response.data.shape[1])]


class DirichletMultinomial(Family):
    RESPONSE_NDIM = 1
    PARAMETERS = {
        "a": ParamSpec(links=["log", "logexpm1"], ndim=1, coefs_dim=DimType.RESPONSE),
    }

    def get_levels(self, response):
        levels = extract_argument_names(response.full_name, list(transformations_namespace))
        if levels:
            return levels
        return [str(level) for level in range(response.data.shape[1])]


def _cure_moments_error(name):
    return NotImplementedError(
        f"The '{name}' family is a mixture cure model. Cured subjects never experience the "
        "event, so the mean and variance of the response are not finite."
    )


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _beta_variance(mu, kappa):
    return mu * (1 - mu) / (1 + kappa)


def _binomial_moments(p, n):
    return n * p, n * p * (1 - p)


def _gamma_moments(mu, alpha):
    return mu, mu**2 / alpha


def _lognormal_moments(mu, sigma):
    # 'mu' and 'sigma' are the mean and standard deviation of log(Y)
    mean = np.exp(mu + sigma**2 / 2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    return mean, variance


def _negativebinomial_moments(mu, alpha):
    return mu, mu + mu**2 / alpha


def _weibull_moments(beta, alpha):
    # 'beta' is the scale and 'alpha' the shape
    gamma_1 = sp.gamma(1 + 1 / alpha)
    return beta * gamma_1, beta**2 * (sp.gamma(1 + 2 / alpha) - gamma_1**2)


def _zero_inflated_moments(mean, variance, psi):
    """Moments of a zero-inflated distribution

    `mean` and `variance` are the moments of the base distribution, which is used with
    probability `psi`. The remaining `1 - psi` of the mass is a point mass at zero.
    """
    return psi * mean, psi * variance + psi * (1 - psi) * mean**2


def _hurdle_moments(mean, variance, psi, p_zero=0):
    """Moments of a hurdle distribution

    `mean` and `variance` are the moments of the base distribution, and `p_zero` is the
    probability it assigns to zero (0 for continuous distributions). With probability `psi` the
    value comes from the base distribution truncated to exclude zero, otherwise it is zero.
    """
    mean_nonzero = mean / (1 - p_zero)
    second_moment_nonzero = (variance + mean**2) / (1 - p_zero)
    hurdle_mean = psi * mean_nonzero
    return hurdle_mean, psi * second_moment_nonzero - hurdle_mean**2


def _warn_infinite_variance(condition, name, bound, distribution):
    if condition.any():
        warnings.warn(
            f"Some posterior draws of '{name}' are {bound}. The variance of the {distribution} "
            "distribution is not finite for them.",
            UserWarning,
            stacklevel=3,
        )
