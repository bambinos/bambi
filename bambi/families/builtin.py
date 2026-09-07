from bambi.families.family import Family
from bambi.families.types import DimType, ResponseType, ParamSpec
from bambi.transformations import transformations_namespace
from bambi.utils import extract_argument_names


class AsymmetricLaplace(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "b": ParamSpec(links=["log"]),
        "kappa": ParamSpec(links=["log"]),
        "q": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class Bernoulli(Family):
    DATA_TYPE = ResponseType.BINARY
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
    }


class Beta(Family):
    """Beta family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit", "probit", "cloglog"]),
        "kappa": ParamSpec(links=["log"]),
    }


class BetaBinomial(Family):
    """BetaBinomial family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    PARAMETERS = {
        "mu": ParamSpec(links=["logit", "probit", "cloglog"]),
        "kappa": ParamSpec(links=["log"]),
    }


class Binomial(Family):
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
    }


class Categorical(Family):
    DATA_TYPE = ResponseType.CATEGORICAL
    PARAMETERS = {
        "p": ParamSpec(links=["softmax"], ndim=1, coefs_dim=DimType.RESPONSE_REDUCED),
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
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "sigma": ParamSpec(links=["log"]),
        "nu": ParamSpec(links=["log"]),
    }


class Exponential(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
    }


class Gamma(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "alpha": ParamSpec(links=["log"]),
    }


class Gaussian(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "sigma": ParamSpec(links=["log"]),
    }


class HurdleGamma(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "alpha": ParamSpec(links=["log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class HurdleLogNormal(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "sigma": ParamSpec(links=["log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class HurdleNegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "cloglog"]),
        "alpha": ParamSpec(links=["log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class HurdlePoisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class NegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "cloglog"]),
        "alpha": ParamSpec(links=["log"]),
    }


class Laplace(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "b": ParamSpec(links=["log"]),
    }


class LogNormal(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "sigma": ParamSpec(links=["log"]),
    }


class LogLogistic(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "alpha": ParamSpec(links=["log"]),
    }


class Poisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log"]),
    }


class StoppingRatio(Family):
    # There's a single linear predictor, not as many linear predictors as response levels.
    DATA_TYPE = ResponseType.ORDINAL
    PARAMETERS = {
        "p": ParamSpec(links=["logit", "probit", "cloglog"], ndim=1),
        "threshold": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE_CUTPOINTS),
    }


class StudentT(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "inverse"]),
        "sigma": ParamSpec(links=["log"]),
        "nu": ParamSpec(links=["log"]),
    }


class VonMises(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity"]),
        "kappa": ParamSpec(links=["log"]),
    }


class Wald(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["inverse", "inverse_squared", "identity", "log"]),
        "lam": ParamSpec(links=["log"]),
    }


class Weibull(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["log", "identity", "inverse"]),
        "alpha": ParamSpec(links=["log"]),
    }


class ZeroInflatedBinomial(Family):
    PARAMETERS = {
        "p": ParamSpec(links=["identity", "logit", "probit", "cloglog"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class ZeroInflatedNegativeBinomial(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log", "cloglog"]),
        "alpha": ParamSpec(links=["log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


class ZeroInflatedPoisson(Family):
    PARAMETERS = {
        "mu": ParamSpec(links=["identity", "log"]),
        "psi": ParamSpec(links=["logit", "probit", "cloglog"]),
    }


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
        "a": ParamSpec(links=["log"], ndim=1, coefs_dim=DimType.RESPONSE),
    }

    def get_levels(self, response):
        levels = extract_argument_names(response.full_name, list(transformations_namespace))
        if levels:
            return levels
        return [str(level) for level in range(response.data.shape[1])]
