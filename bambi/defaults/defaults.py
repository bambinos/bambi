from bambi.families import Likelihood
from bambi.priors import Prior

from bambi.families.univariate import (
    Bernoulli,
    Beta,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Laplace,
    Poisson,
    StudentT,
    VonMises,
    Wald,
)

from bambi.families.multivariate import Categorical, Multinomial

# Default parameters for PyMC distributions
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
    "Laplace": {"mu": 0, "b": 1},
    "Poisson": {"mu": 1},
    "StudentT": {"lam": 1, "nu": 1},
    "VonMises": {"mu": 0, "kappa": 1},
    "Wald": {"mu": 1, "lam": 1},
}

# fmt: off
BUILTIN_FAMILIES = {
    "bernoulli": {
        "likelihood": {
            "name": "Bernoulli",
            "params": ["p"],
            "parent": "p",
        },
        "link": {"p": "logit"},
        "family": Bernoulli,
    },
    "beta": {
        "likelihood": {
            "name": "Beta",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "logit", "kappa": "log"},
        "family": Beta,
    },
    "binomial": {
        "likelihood": {
            "name": "Binomial",
            "params": ["p"],
            "parent": "p",
        },
        "link": {"p": "logit"},
        "family": Binomial,
    },
    "categorical": {
        "likelihood": {
            "name": "Categorical",
            "params": ["p"],
            "parent": "p",
        },
        "link": {"p": "softmax"},
        "family": Categorical,
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "inverse", "alpha": "log"},
        "family": Gamma,
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "params": ["mu", "sigma"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log"},
        "family": Gaussian,
    },
    "multinomial": {
        "likelihood": {
            "name": "Multinomial",
            "params": {},
            "parent": "p"
        },
        "link": {"p": "softmax"},
        "family": Multinomial,
    },
    "negativebinomial": {
        "likelihood": {
            "name": "NegativeBinomial",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "log", "alpha": "log"},
        "family": NegativeBinomial,
    },
    "laplace": {
        "likelihood": {
            "name": "Laplace",
            "params": ["mu", "b"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "b": "log"},
        "family": Laplace,
    },
    "poisson": {
        "likelihood": {
            "name": "Poisson",
            "params": ["mu"],
            "parent": "mu",
        },
        "link": {"mu": "log"},
        "family": Poisson,
    },
    "t": {
        "likelihood": {
            "name": "StudentT",
            "params": ["mu", "sigma", "nu"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log", "nu": "log"},
        "family": StudentT,
    },
    "vonmises": {
        "likelihood": {
            "name": "VonMises",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "tan_2", "kappa": "log"},
        "family": VonMises,
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "params": ["mu", "lam"],
            "parent": "mu",
        },
        "link": {"mu": "inverse_squared", "lam": "log"},
        "family": Wald,
    },
}
# fmt: on


def generate_prior(dist, **kwargs):
    """Generate a Prior distribution.

    The parameter ``kwargs`` is used to pass hyperpriors that are assigned to the parameters of
    the prior to be built.

    Parameters
    ----------
    dist: str, int, float
        If a string, it is the name of the prior distribution with default values taken from
        ``SETTINGS_DISTRIBUTIONS``. If a number, it is a factor used to scale the standard deviation
        of the priors generated automatically by Bambi.

    Raises
    ------
    ValueError
        If ``dist`` is not a string or a number.

    Returns
    -------
    Prior
        The Prior instance.
    """
    if isinstance(dist, str):
        prior = Prior(dist, **SETTINGS_DISTRIBUTIONS[dist])
        if kwargs:
            prior.update(**{k: generate_prior(v) for k, v in kwargs.items()})
    elif isinstance(dist, (int, float)):
        prior = dist
    else:
        raise ValueError("'dist' must be the name of a distribution or a numeric value.")
    return prior


def generate_likelihood(name, params, parent):
    """Generate a Likelihood instance.

    FIXME
    Parameters
    ----------
    name: str
        The name of the likelihood function.
    args: dict
        Indicates the auxiliary parameters and the values for their default priors. The keys are the
        names of the parameters and the values are passed to ``generate_prior()`` to obtain the
        actual instance of ``bambi.Prior``.
    parent: str
        The name of the parent parameter. In other words, the name of the mean parameter in the
        likelihood function.

    Returns
    -------
    bambi.Likelihood
        The likelihood instance.
    """
    #priors = {k: generate_prior(v) for k, v in args.items()}
    return Likelihood(name, params, parent)


def generate_family(name, likelihood, link, family):
    """Generate a Bambi family.

    Parameters
    ----------
    name: str
        The name of the family.
    likelihood: bambi.Likelihood
        A representation of the likelihood function that corresponds to the family being created.
    link: bambi.Link
        A representation of the link function that corresponds to the family being created.
    family: subclass of bambi.Family
        A subclass of bambi.Family that generates the instance of the desired family.

    Returns
    -------
    bambi.Family
        The family instance.
    """
    likelihood = generate_likelihood(**likelihood)
    return family(name, likelihood, link)


def get_default_prior(term_type):
    """Generate a Prior based on the default settings

    The following summarises default priors for each type of term:

    * intercept: Normal prior.
    * common: Normal prior.
    * intercept_flat: Uniform prior.
    * common_flat: Uniform prior.
    * group_specific: Normal prior where its sigma has a HalfNormal hyperprior.
    * group_specific_flat: Normal prior where its sigma has a HalfFlat hyperprior.

    Parameters
    ----------
    term_type: str
        The type of the term for which the default prior is wanted.

    Raises
    ------
    ValueError
        If ``term_type`` is not within the values listed above.

    Returns
    -------
    prior: Prior
        The instance of Prior according to the ``term_type``.
    """
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
    """Generate a built-in ``bambi.families.Family`` instance.

    Given the name of a built-in family, this function returns a ``bambi.families.Family`` instance
    that is constructed by calling other utility functions that construct the
    ``bambi.families.Likelihood`` and the ``bambi.priors.Prior`` instances that are needed to build
    the family.

    The available built-in families are found in ``SETTINGS_FAMILIES``.

    Parameters
    ----------
    name: str
        The name of the built-in family.

    Raises
    ------
    ValueError
        If ``name`` is not the name of a built-in family.

    Returns
    -------
    bambi.families.Family
        The family instance.
    """
    if name in BUILTIN_FAMILIES:
        return generate_family(name, **BUILTIN_FAMILIES[name])
    raise ValueError(f"'{name}' is not a valid built-in family name.")
