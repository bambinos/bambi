import numpy as np
import pymc3 as pm
import theano.tensor as tt


def probit(x):
    """Probit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 0.5 + 0.5 * tt.erf(x / tt.sqrt(2))
    result = tt.switch(tt.eq(result, 0), eps, result)
    result = tt.switch(tt.eq(result, 1), 1 - eps, result)

    return result


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 1 - tt.exp(-tt.exp(x))
    result = tt.switch(tt.eq(result, 0), eps, result)
    result = tt.switch(tt.eq(result, 1), 1 - eps, result)

    return result


def get_pymc_distribution(dist):
    """Return a PyMC3 distribution."""
    if isinstance(dist, str):
        if hasattr(pm, dist):
            dist = getattr(pm, dist)
        else:
            raise ValueError(f"The Distribution '{dist}' was not found in PyMC3")
    return dist


def has_hyperprior(kwargs):
    """Determines if a Prior has an hyperprior"""
    return (
        "sigma" in kwargs
        and "observed" not in kwargs
        and isinstance(kwargs["sigma"], pm.model.TransformedRV)
    )
