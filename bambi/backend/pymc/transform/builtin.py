import numpy as np
import pytensor.tensor as pt

from bambi.backend.pymc.transform.register import transforms_registry
from bambi.families.builtin import (
    Beta,
    BetaBinomial,
    Bernoulli,
    Categorical,
    Cumulative,
    Exponential,
    Gamma,
    HurdleGamma,
    Multinomial,
    StoppingRatio,
    Weibull,
)


@transforms_registry.transform_data(Bernoulli)
def _(data):
    if not np.all(np.isin(data, (0, 1))):
        raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family")
    return {"observed": data}


@transforms_registry.transform_parameters(Beta)
def _(parameters):
    mu = parameters["mu"]
    kappa = parameters["kappa"]
    return {"alpha": mu * kappa, "beta": (1 - mu) * kappa}


@transforms_registry.transform_parameters(BetaBinomial)
def _(parameters):
    mu = parameters["mu"]
    kappa = parameters["kappa"]
    return {"alpha": mu * kappa, "beta": (1 - mu) * kappa}


@transforms_registry.transform_predictor(Categorical, "p")
def _(predictor, _parameters, inverse_link):
    if predictor.ndim == 1:
        zeros = pt.zeros(shape=(1,))
    else:
        zeros = pt.zeros(shape=(predictor.shape[0], 1))
    return inverse_link(pt.concatenate((zeros, predictor), axis=-1))


@transforms_registry.transform_predictor(Cumulative, "p")
def _(predictor, parameters, inverse_link):
    # P(Y = k) = F(threshold_k - predictor) - F(threshold_{k - 1} - predictor)
    threshold = parameters["threshold"]

    if predictor == 0:
        # When the model does not have any predictors. PyMC will reshape accordingly.
        predictor = threshold
    else:
        predictor = threshold - pt.shape_padright(predictor)

    probability = inverse_link(predictor)
    probability = pt.concatenate(
        [
            pt.shape_padright(probability[..., 0]),
            probability[..., 1:] - probability[..., :-1],
            pt.shape_padright(1 - probability[..., -1]),
        ],
        axis=-1,
    )

    return probability


@transforms_registry.transform_parameters(Cumulative)
def _(parameters):
    return {"p": parameters["p"]}


@transforms_registry.transform_parameters(Exponential)
def _(parameters):
    return {"lam": 1 / parameters["mu"]}


@transforms_registry.transform_parameters(Gamma)
def _(parameters):
    return {
        "mu": parameters["mu"],
        "sigma": parameters["mu"] / (parameters["alpha"] ** 0.5),
    }


@transforms_registry.transform_parameters(HurdleGamma)
def _(parameters):
    return {
        "mu": parameters["mu"],
        "sigma": parameters["mu"] / (parameters["alpha"] ** 0.5),
        "psi": parameters["psi"],
    }


@transforms_registry.transform_predictor(Multinomial, "p")
def _(predictor, _parameters, inverse_link):
    if predictor.ndim == 1:
        zeros = pt.zeros(shape=(1,))
    else:
        zeros = pt.zeros(shape=(predictor.shape[0], 1))
    return inverse_link(pt.concatenate((zeros, predictor), axis=-1))


@transforms_registry.transform_predictor(StoppingRatio, "p")
def _(predictor, parameters, inverse_link):
    # P(Y = k) = F(threshold_k - predictor) * prod_(j=1)^(k-1)(1 - F(threshold_j - predictor))
    threshold = parameters["threshold"]

    if predictor == 0:
        # An additive predictor with no predictors, e.g. p ~ 0.
        # shape: (K, )
        predictor = threshold
    else:
        # shape: (n, K)
        predictor = threshold - pt.shape_padright(predictor)

    probability = inverse_link(predictor)
    survival = pt.cumprod(1 - probability, axis=-1)
    return pt.concatenate(
        [
            probability[..., :1],
            probability[..., 1:] * survival[..., :-1],
            survival[..., -1:],
        ],
        axis=-1,
    )


@transforms_registry.transform_parameters(StoppingRatio)
def _(parameters):
    return {"p": parameters["p"]}


@transforms_registry.transform_parameters(Weibull)
def _(parameters):
    mu = parameters["mu"]
    alpha = parameters["alpha"]
    return {
        "alpha": alpha,
        "beta": mu / pt.gamma(1 + 1 / alpha),
    }
