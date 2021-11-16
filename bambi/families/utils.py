from .family import Family

# Names of parameters that can receive a prior distribution for the built-in families
FAMILY_PARAMS = {
    "beta": ("kappa",),
    "gamma": ("alpha",),
    "gaussian": ("sigma",),
    "negativebinomial": ("alpha",),
    "t": ("sigma", "nu"),
    "wald": ("lam",),
}


def _extract_family_prior(family, priors):
    """Extract priors for a given family

    If a key in the priors dictionary matches the name of a nuisance parameter of the response
    distribution for the given family, this function extracts and returns the prior for that
    nuisance parameter. The result of this function can be safely used to update the ``Prior`` of
    the response term.

    Parameters
    ----------
    family: str or ``bambi.families.Family``
        The family for which we want to extract priors.
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
        # Only work if there are auxiliary parameters in the family, and if any of these are
        # present in 'priors' dictionary.
        nuisance_params = list(family.likelihood.priors)
        if set(nuisance_params).intersection(set(priors)):
            return {k: priors.pop(k) for k in nuisance_params if k in priors}
    return None
