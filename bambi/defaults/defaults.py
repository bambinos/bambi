from . import pps
from ..priors import Family, Likelihood, Prior


# Default parameters for PyMC3 distributions
SETTINGS_DISTRIBUTIONS = {
    "Bernoulli": {"p": 0.5},
    "Beta": {"alpha": 1, "beta": 1},
    "Binomial": {"n": 1, "p": 0.5},
    "Cauchy": {"alpha": 0, "beta": 1},
    "Flat": {},
    "Gamma": {"alpha": 2, "beta": 2},
    "HalfCauchy": {"beta": 1},
    "HalfFlat": {},
    "HalfNormal": {"sigma": 1},
    "Normal": {"mu": 0, "sigma": 1},
    "NegativeBinomial": {"alpha": 1, "mu": 1},
    "Poisson": {"mu": 1},
    "StudentT": {"lam": 1, "nu": 1},
    "Wald": {"mu": 1, "lam": 1},
}

# fmt: off
# Beta: it asks for kappa, then we do alpha = mu*kappa, beta= (1-mu)*kappa
SETTINGS_FAMILIES = {
    "bernoulli": {
        "likelihood": {
            "name": "Bernoulli",
            "args": {},
            "parent": "p",
            "pps": pps.pps_bernoulli
        },
        "link": "logit"
    },
    "beta": {
        "likelihood": {
            "name": "Beta",
            "args": {
                "kappa": "HalfCauchy"
            },
            "parent": "mu",
            "pps": pps.pps_beta
        },
        "link": "logit"
    },
    "binomial": {
        "likelihood": {
            "name": "Binomial",
            "args": {},
            "parent": "p",
            "pps": pps.pps_binomial
        },
        "link": "logit"
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu",
            "pps": pps.pps_gamma
        },
        "link": "inverse",
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "args": {
                "sigma": "HalfNormal"
            },
            "parent": "mu",
            "pps": pps.pps_gaussian
        },
        "link": "identity",
    },
    "negativebinomial": {
        "likelihood": {
            "name": "NegativeBinomial",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu",
            "pps": pps.pps_negativebinomial
        },
        "link": "log",
    },
    "poisson": {
        "likelihood": {
            "name": "Poisson",
            "args": {},
            "parent": "mu",
            "pps": pps.pps_poisson
        },
        "link": "log"
    },
    "t": {
        "likelihood": {
            "name": "StudentT",
            "args": {
                "lam": "HalfCauchy",
                "nu": 2
            },
            "parent": "mu",
            "pps": pps.pps_t
        },
        "link": "identity",
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "args": {
                "lam": "HalfCauchy"
            },
            "parent": "mu",
            "pps": pps.pps_wald
        },
        "link": "inverse_squared",
    },
}
# fmt: on


def generate_prior(dist, **kwargs):
    if isinstance(dist, str):
        prior = Prior(dist, **SETTINGS_DISTRIBUTIONS[dist])
        if kwargs:
            prior.update(**{k: generate_prior(v) for k, v in kwargs.items()})
    elif isinstance(dist, (int, float)):
        prior = dist
    else:
        raise ValueError("'dist' must be the name of a distribution or a numeric value.")
    return prior


def generate_likelihood(name, args, parent, pps):  # pylint: disable=redefined-outer-name
    priors = {k: generate_prior(v) for k, v in args.items()}
    return Likelihood(name, parent, pps, **priors)


def generate_family(name, likelihood, link):
    likelihood = generate_likelihood(**likelihood)
    return Family(name, likelihood, link)


def get_default_prior(term_type):
    if term_type in ["intercept", "common"]:
        prior = generate_prior("Normal")
    elif term_type in ["intercept_flat", "common_flat"]:
        prior = generate_prior("Flat")
    elif term_type == "group_specific":
        prior = generate_prior("Normal", sigma="HalfNormal")
    elif term_type == "group_specific_flat":
        prior = generate_prior("Normal", sigma="HalfFlat")
    else:
        raise ValueError("Unrecognized term type.")
    return prior


def get_builtin_family(name):
    """Generate a built-in ``Family`` instance

    Given the name of a built-in family, this function returns a ``Family`` instance that is
    constructed by calling other utility functions that construct the ``Likelihood`` and the
    ``Prior``s that are needed to build the family.
    """
    return generate_family(name, **SETTINGS_FAMILIES[name])
