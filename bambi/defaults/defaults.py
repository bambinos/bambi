from ..priors import Family, Likelihood, Prior


# Default parameters for PyMC3 distributions
SETTINGS_DISTRIBUTIONS = {
    "Bernoulli": {"p": 0.5},
    "Beta": {"alpha": 1, "beta": 1},
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
SETTINGS_FAMILIES = {
    "bernoulli": {
        "likelihood": {
            "name": "Bernoulli",
            "args": {},
            "parent": "p"
        },
        "link": "logit"
    },
    "gamma": {
        "likelihood": {
            "name": "Gamma",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu"
        },
        "link": "inverse",
    },
    "gaussian": {
        "likelihood": {
            "name": "Normal",
            "args": {
                "sigma": "HalfNormal"
            },
            "parent": "mu"
        },
        "link": "identity",
    },
    "negativebinomial": {
        "likelihood": {
            "name": "NegativeBinomial",
            "args": {
                "alpha": "HalfCauchy"
            },
            "parent": "mu"
        },
        "link": "log",
    },
    "poisson": {
        "likelihood": {
            "name": "Poisson",
            "args": {},
            "parent": "mu"
        },
        "link": "log"
    },
    "t": {
        "likelihood": {
            "name": "StudentT",
            "args": {
                "lam": "HalfCauchy"
            },
            "parent": "mu"
        },
        "link": "identity",
    },
    "wald": {
        "likelihood": {
            "name": "Wald",
            "args": {
                "lam": "HalfCauchy"
            },
            "parent": "mu"
        },
        "link": "inverse_squared",
    },
}
# fmt: on


def generate_prior(name, **kwargs):
    prior = Prior(name, **SETTINGS_DISTRIBUTIONS[name])
    if kwargs:
        prior.update(**{k: generate_prior(v) for k, v in kwargs.items()})
    return prior


def generate_likelihood(name, args, parent):
    priors = {k: generate_prior(v) for k, v in args.items()}
    return Likelihood(name, parent, **priors)


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
    return generate_family(name, **SETTINGS_FAMILIES[name])
