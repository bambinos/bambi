from bambi.defaults.utils import generate_family
from bambi.families.univariate import (
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Categorical,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Laplace,
    Poisson,
    StudentT,
    VonMises,
    Wald,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from bambi.families.multivariate import Multinomial, DirichletMultinomial


# fmt: off
BUILTIN_FAMILIES = {
    "asymmetriclaplace": {
        "likelihood": {
            "name": "AsymmetricLaplace",
            "params": ["mu", "b", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "b": "log", "kappa": "log"},
        "family": AsymmetricLaplace,
        "default_priors": {"b": "HalfNormal", "kappa": "HalfNormal"}
    },

    "bernoulli": {
        "likelihood": {
            "name": "Bernoulli",
            "params": ["p"],
            "parent": "p",
        },
        "link": {"p": "logit"},
        "family": Bernoulli
    },
    "beta": {
        "likelihood": {
            "name": "Beta",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "logit", "kappa": "log"},
        "family": Beta,
        "default_priors": {"kappa": "HalfCauchy"},
    },
    "beta_binomial": {
        "likelihood": {
            "name": "BetaBinomial",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "logit", "kappa": "log"},
        "family": BetaBinomial,
        "default_priors": {"kappa": "HalfCauchy"},
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
    "dirichlet_multinomial": {
        "likelihood": {
            "name": "DirichletMultinomial",
            "params": ["a"],
            "parent": "a",
        },
        "link": {"a": "log"},
        "family": DirichletMultinomial,
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "inverse", "alpha": "log"},
        "family": Gamma,
        "default_priors": {"alpha": "HalfCauchy"},
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "params": ["mu", "sigma"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log"},
        "family": Gaussian,
        "default_priors": {"sigma": "HalfNormal"}
    },
    "multinomial": {
        "likelihood": {
            "name": "Multinomial",
            "params": ["p"],
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
        "default_priors": {"alpha": "HalfCauchy"},
    },
    "laplace": {
        "likelihood": {
            "name": "Laplace",
            "params": ["mu", "b"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "b": "log"},
        "family": Laplace,
        "default_priors": {"b": "HalfNormal"},
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
        "default_priors": {"sigma": "HalfNormal", "nu": "Gamma"},
    },
    "vonmises": {
        "likelihood": {
            "name": "VonMises",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "tan_2", "kappa": "log"},
        "family": VonMises,
        "default_priors": {"kappa": "HalfNormal"},
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "params": ["mu", "lam"],
            "parent": "mu",
        },
        "link": {"mu": "inverse_squared", "lam": "log"},
        "family": Wald,
        "default_priors": {"lam": "HalfCauchy"},
    },
    "zero_inflated_binomial": {
        "likelihood": {
            "name": "ZeroInflatedBinomial",
            "params": ["p", "psi"],
            "parent": "p"
        },
        "link": {"p": "logit", "psi": "logit"},
        "family": ZeroInflatedBinomial,
        "default_priors": {"psi": "Beta"},
    },
    "zero_inflated_negativebinomial": {
        "likelihood": {
            "name": "ZeroInflatedNegativeBinomial",
            "params": ["mu", "alpha", "psi"],
            "parent": "mu",
        },
        "link": {"mu": "log", "alpha": "log", "psi": "logit"},
        "family": ZeroInflatedNegativeBinomial,
        "default_priors": {"alpha": "HalfCauchy", "psi": "Beta"},
    },
    "zero_inflated_poisson": {
        "likelihood": {
            "name": "ZeroInflatedPoisson",
            "params": ["mu", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "log", "psi": "logit"},
        "family": ZeroInflatedPoisson,
        "default_priors": {"psi": "Beta"},
    }
}
# fmt: on


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
