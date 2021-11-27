from bambi.families import Likelihood
from bambi.priors import Prior

from bambi.families.univariate import (
    Bernoulli,
    Beta,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    StudentT,
    Wald,
)

from bambi.families.multivariate import Categorical

## NOTE: Check docs/api_reference.rst links the right lines from this document
# Default parameters for PyMC3 distributions
SETTINGS_DISTRIBUTIONS = {
    "Bernoulli": {"p": 0.5},
    "Beta": {"alpha": 1, "beta": 1},
    "Binomial": {"n": 1, "p": 0.5},
    "Cauchy": {"alpha": 0, "beta": 1},
    "Flat": {},
    "Gamma": {"alpha": 2, "beta": 0.1},
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
SETTINGS_FAMILIES = {
    "bernoulli": {
        "likelihood": {
            "name": "Bernoulli",
            "args": {},
            "parent": "p",
        },
        "link": "logit",
        "family": Bernoulli,
    },
    "beta": {
        "likelihood": {
            "name": "Beta",
            "args": {
                "kappa": "HalfCauchy"
            },
            "parent": "mu",
        },
        "link": "logit",
        "family": Beta,
    },
    "binomial": {
        "likelihood": {
            "name": "Binomial",
            "args": {},
            "parent": "p",
        },
        "link": "logit",
        "family": Binomial,
    },
    "categorical": {
        "likelihood": {
            "name": "Categorical",
            "args": {},
            "parent": "p",
        },
        "link": "softmax",
        "family": Categorical,
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu",
        },
        "link": "inverse",
        "family": Gamma,
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "args": {
                "sigma": "HalfNormal"
            },
            "parent": "mu",
        },
        "link": "identity",
        "family": Gaussian,
    },
    "negativebinomial": {
        "likelihood": {
            "name": "NegativeBinomial",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu",
        },
        "link": "log",
        "family": NegativeBinomial,
    },
    "poisson": {
        "likelihood": {
            "name": "Poisson",
            "args": {},
            "parent": "mu",
        },
        "link": "log",
        "family": Poisson,
    },
    "t": {
        "likelihood": {
            "name": "StudentT",
            "args": {
                "sigma": "HalfNormal",
                "nu": "Gamma"
            },
            "parent": "mu",
        },
        "link": "identity",
        "family": StudentT,
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "args": {
                "lam": "HalfCauchy"
            },
            "parent": "mu",
        },
        "link": "inverse_squared",
        "family": Wald,
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


def generate_likelihood(name, args, parent):  # pylint: disable=redefined-outer-name
    priors = {k: generate_prior(v) for k, v in args.items()}
    return Likelihood(name, parent, **priors)


def generate_family(name, likelihood, link, family):
    likelihood = generate_likelihood(**likelihood)
    return family(name, likelihood, link)


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
    """Generate a built-in ``bambi.families.Family`` instance

    Given the name of a built-in family, this function returns a ``bambi.families.Family`` instance
    that is constructed by calling other utility functions that construct the
    ``bambi.families.Likelihood`` and the ``bambi.priors.Prior`` instances that are needed to build
    the family.
    """
    if name in SETTINGS_FAMILIES:
        return generate_family(name, **SETTINGS_FAMILIES[name])
    raise ValueError("f'{name}' is not a valid built-in family name")
