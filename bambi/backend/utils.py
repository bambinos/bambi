import aesara.tensor as at
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
        and isinstance(kwargs["sigma"], at.TensorVariable)
    )
