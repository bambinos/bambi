from bambi.defaults.utils import generate_family
from bambi.families.builtin import (
    AsymmetricLaplace,
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Categorical,
    Cumulative,
    ExGaussian,
    DirichletMultinomial,
    Exponential,
    Gamma,
    Gaussian,
    HurdleGamma,
    HurdleLogNormal,
    HurdleNegativeBinomial,
    HurdlePoisson,
    Laplace,
    LogLogistic,
    LogNormal,
    Multinomial,
    NegativeBinomial,
    Poisson,
    StoppingRatio,
    StudentT,
    VonMises,
    Wald,
    Weibull,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)

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
        "default_priors": {
            "b": {"name": "HalfNormal", "sigma": 1},
            "kappa": {"name": "HalfNormal", "sigma": 1},
        },
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
        "default_priors": {"kappa": {"name": "HalfCauchy", "beta": 1}},
    },
    "beta_binomial": {
        "likelihood": {
            "name": "BetaBinomial",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "logit", "kappa": "log"},
        "family": BetaBinomial,
        "default_priors": {"kappa": {"name": "HalfCauchy", "beta": 1}},
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
    "cumulative": {
        "likelihood": {
            "name": "Cumulative",
            "params": ["p", "threshold"],
            "parent": "p",
        },
        "link": {"p": "logit", "threshold": "identity"},
        "family": Cumulative,
        "default_priors": {
            "threshold": {"name": "Normal", "mu": 0, "sigma": 1, "transform": "ordered"}
        },
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
    "exgaussian": {
        "likelihood": {
            "name": "ExGaussian",
            "params": ["mu", "sigma", "nu"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log", "nu": "log"},
        "family": ExGaussian,
        "default_priors": {
            "sigma": {"name": "HalfNormal", "sigma": 1},
            "nu": {"name": "HalfNormal", "sigma": 1},
        },
    },
    "exponential": {
        "likelihood": {
            "name": "Exponential",
            "params": ["mu"],
            "parent": "mu",
        },
        "link": {"mu": "log"},
        "family": Exponential,
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "inverse", "alpha": "log"},
        "family": Gamma,
        "default_priors": {"alpha": {"name": "HalfCauchy", "beta": 1}},
    },
    "lognormal": {
        "likelihood": {
            "name": "LogNormal",
            "params": ["mu", "sigma"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log"},
        "family": LogNormal,
        "default_priors": {"sigma": {"name": "HalfNormal", "sigma": 1}},
    },
    "loglogistic": {
        "likelihood": {
            "name": "LogLogistic",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "alpha": "log"},
        "family": LogLogistic,
        "default_priors": {"alpha": {"name": "HalfNormal", "sigma": 1}},
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "params": ["mu", "sigma"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log"},
        "family": Gaussian,
        "default_priors": {"sigma": {"name": "HalfNormal", "sigma": 1}}
    },
    "hurdle_gamma": {
        "likelihood": {
            "name": "HurdleGamma",
            "params": ["mu", "alpha", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "log", "alpha": "log", "psi": "logit"},
        "family": HurdleGamma,
        "default_priors": {
            "alpha": {"name": "HalfCauchy", "beta": 1},
            "psi": {"name": "Beta", "alpha": 2, "beta": 2}
        }
    },
    "hurdle_lognormal": {
        "likelihood": {
            "name": "HurdleLogNormal",
            "params": ["mu", "sigma", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "identity", "sigma": "log", "psi": "logit"},
        "family": HurdleLogNormal,
        "default_priors": {
            "sigma": {"name": "HalfNormal", "sigma": 1},
            "psi": {"name": "Beta", "alpha": 2, "beta": 2}
        }
    },
    "hurdle_negativebinomial": {
        "likelihood": {
            "name": "HurdleNegativeBinomial",
            "params": ["mu", "alpha", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "log", "alpha": "log", "psi": "logit"},
        "family": HurdleNegativeBinomial,
        "default_priors": {
            "alpha": {"name": "HalfCauchy", "beta": 1}, 
            "psi": {"name": "Beta", "alpha": 2, "beta": 2}
        }
    },
    "hurdle_poisson": {
        "likelihood": {
            "name": "HurdlePoisson",
            "params": ["mu", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "log", "psi": "logit"},
        "family": HurdlePoisson,
        "default_priors": {"psi": {"name": "Beta", "alpha": 2, "beta": 2}},
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
        "default_priors": {"alpha": {"name": "HalfCauchy", "beta": 1}},
    },
    "laplace": {
        "likelihood": {
            "name": "Laplace",
            "params": ["mu", "b"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "b": "log"},
        "family": Laplace,
        "default_priors": {"b": {"name": "HalfNormal", "sigma": 1}},
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
    "sratio": {
        "likelihood": {
            "name": "StoppingRatio",
            "params": ["p", "threshold"],
            "parent": "p",
        },
        "link": {"p": "logit", "threshold": "identity"},
        "family": StoppingRatio,
        "default_priors": {"threshold": {"name": "Normal", "mu": 0, "sigma": 1}},
    },
    "t": {
        "likelihood": {
            "name": "StudentT",
            "params": ["mu", "sigma", "nu"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "sigma": "log", "nu": "log"},
        "family": StudentT,
        "default_priors": {
            "sigma": {"name": "HalfNormal", "sigma": 1}, 
            "nu": {"name": "Gamma", "alpha": 2, "beta": 0.1}
        },
    },
    "vonmises": {
        "likelihood": {
            "name": "VonMises",
            "params": ["mu", "kappa"],
            "parent": "mu",
        },
        "link": {"mu": "identity", "kappa": "log"},
        "family": VonMises,
        "default_priors": {"kappa": {"name": "HalfNormal", "sigma": 1}},
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "params": ["mu", "lam"],
            "parent": "mu",
        },
        "link": {"mu": "inverse_squared", "lam": "log"},
        "family": Wald,
        "default_priors": {"lam": {"name": "HalfCauchy", "beta": 1}},
    },
    "weibull": {
        "likelihood": {
            "name": "Weibull",
            "params": ["mu", "alpha"],
            "parent": "mu",
        },
        "link": {"mu": "log", "alpha": "log"},
        "family": Weibull,
        "default_priors": {"alpha": {"name": "HalfCauchy", "beta": 1}},
    },
    "zero_inflated_binomial": {
        "likelihood": {
            "name": "ZeroInflatedBinomial",
            "params": ["p", "psi"],
            "parent": "p"
        },
        "link": {"p": "logit", "psi": "logit"},
        "family": ZeroInflatedBinomial,
        "default_priors": {"psi": {"name": "Beta", "alpha": 2, "beta": 2}},
    },
    "zero_inflated_negativebinomial": {
        "likelihood": {
            "name": "ZeroInflatedNegativeBinomial",
            "params": ["mu", "alpha", "psi"],
            "parent": "mu",
        },
        "link": {"mu": "log", "alpha": "log", "psi": "logit"},
        "family": ZeroInflatedNegativeBinomial,
        "default_priors": {
            "alpha": {"name": "HalfCauchy", "beta": 1}, 
            "psi": {"name": "Beta", "alpha": 2, "beta": 2}
        },
    },
    "zero_inflated_poisson": {
        "likelihood": {
            "name": "ZeroInflatedPoisson",
            "params": ["mu", "psi"],
            "parent": "mu"
        },
        "link": {"mu": "log", "psi": "logit"},
        "family": ZeroInflatedPoisson,
        "default_priors": {"psi": {"name": "Beta", "alpha": 2, "beta": 2}},
    }
}
# fmt: on


def get_builtin_family(name):
    """Generate a built-in `bambi.families.Family` instance.

    Given the name of a built-in family, this function returns a `bambi.families.Family` instance
    that is constructed by calling other utility functions that construct the
    `bambi.families.Likelihood` and the `bambi.priors.Prior` instances that are needed to build
    the family.

    The available built-in families are found in `BUILTIN_FAMILIES`.

    Parameters
    ----------
    name: str
        The name of the built-in family.

    Raises
    ------
    ValueError
        If `name` is not the name of a built-in family.

    Returns
    -------
    bambi.families.Family
        The family instance.
    """
    if name in BUILTIN_FAMILIES:
        return generate_family(name, **BUILTIN_FAMILIES[name])
    raise ValueError(f"'{name}' is not a valid built-in family name.")
