"""Default prior configuration for monotonic ``mo()`` terms."""

import numpy as np

from bambi.priors.prior import Prior


def generate_prior_monotonic(D):
    """Build the default prior dict for a monotonic effect with simplex length ``D``.

    The simplex is given a symmetric ``Dirichlet(1, ..., 1)`` prior (the brms default).
    The slope is given a unit Normal; auto-scaling will adjust it like any other
    common-term Normal prior at build time.
    """
    return {
        "simplex": Prior("Dirichlet", a=np.ones(D)),
        "slope": Prior("Normal", mu=0, sigma=1),
    }


def generate_prior_monotonic_group_specific(D):
    """Default prior dict for a group-specific ``(mo(x) | g)`` term.

    The slope ``r_g ~ Normal(0, sigma_g)`` with ``sigma_g ~ HalfNormal`` (matching
    bambi's existing group-specific convention). The simplex is shared across
    groups with the same default Dirichlet(1, ..., 1) prior.
    """
    return {
        "simplex": Prior("Dirichlet", a=np.ones(D)),
        "slope": Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=1)),
    }
