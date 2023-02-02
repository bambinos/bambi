import pytensor.tensor as pt
import pymc as pm


def get_distribution(dist):
    """Return a PyMC distribution."""
    if isinstance(dist, str):
        if hasattr(pm, dist):
            dist = getattr(pm, dist)
        else:
            raise ValueError(f"The Distribution '{dist}' was not found in PyMC")
    return dist


def has_hyperprior(kwargs):
    """Determines if a Prior has an hyperprior"""
    return (
        "sigma" in kwargs
        and "observed" not in kwargs
        and isinstance(kwargs["sigma"], pt.TensorVariable)
    )


def get_distribution_from_prior(prior):
    if prior.dist is not None:
        distribution = prior.dist
    else:
        distribution = get_distribution(prior.name)
    return distribution


def get_distribution_from_likelihood(likelihood):
    """
    It works because both `Prior` and `Likelihood` instances have a `name` and a `dist` argument.
    """
    return get_distribution_from_prior(likelihood)
