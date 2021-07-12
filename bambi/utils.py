from .priors import Family

FAMILY_LINKS = {
    "gaussian": ["identity", "log", "inverse"],
    "gamma": ["identity", "log", "inverse"],
    "bernoulli": ["identity", "logit", "probit", "cloglog"],
    "wald": ["inverse", "inverse_squared", "identity", "log"],
    "negativebinomial": ["identity", "log", "cloglog"],
    "poisson": ["identity", "log"],
}

FAMILY_PARAMS = {
    "gaussian": ("sigma",),
    "negativebinomial": ("alpha",),
    "gamma": ("alpha",),
    "t": ("lam", "nu"),
}


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def extract_family_prior(family, priors):
    """Extract priors for a given family

    If a key in the priors dictionary matches the name of a nuisance parameter of the response
    distribution for the given family, this function extracts and returns the prior for that
    nuisance parameter. The result of this function can be safely used to update the ``Prior`` of
    the response term.

    Parameters
    ----------
    family: str or ``Family``
        The name of a built-in family or a ``Family`` instance.
    priors: dict
        A dictionary where keys represent parameter/term names and values represent
        prior distributions.
    """
    if isinstance(family, str) and family in FAMILY_PARAMS:
        names = FAMILY_PARAMS[family]
        priors = {name: priors.pop(name) for name in names if priors.get(name) is not None}
        if priors:
            return priors
    elif isinstance(family, Family):
        # Only work if there are nuisance parameters in the family, and if any of these nuisance
        # parameters is present in 'priors' dictionary.
        nuisance_params = list(family.likelihood.priors)
        if set(nuisance_params).intersection(set(priors)):
            return {k: priors.pop(k) for k in nuisance_params if k in priors}
    return None


def link_match_family(link, family_name):
    """Checks whether the a link can be used in a given family.

    When this function is used with built-in family names, it tests whether the link name can be
    used with the given built-in family. If the family name is not known, we return True because
    the user is working with a custom ``Family`` object.
    Which links can work with which families are taken from statsmodels.
    """
    if family_name in FAMILY_LINKS:
        return link in FAMILY_LINKS[family_name]

    # Custom family, we don't know what link functions can be used
    return True
