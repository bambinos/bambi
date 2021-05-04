from .priors import Family


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def extract_family_prior(family, priors):
    # Given a family and a dictionary of priors, return the prior that could be
    # passed to update the prior of a parameter in that family.

    # Only built-in gaussian and negativebinomial have nuisance parameters
    # i.e, parameters not related to the link-transformed predicted outcome
    if isinstance(family, str):
        if family == "gaussian" and "sigma" in priors:
            return {"sigma": priors["sigma"]}
        elif family == "negativebinomial" and "alpha" in priors:
            return {"alpha": priors["alpha"]}
        elif family == "gamma" and "alpha" in priors:
            return {"alpha": priors["alpha"]}
    elif isinstance(family, Family):
        # Only work if there are nuisance parameters in the family, and if any of these nuisance
        # parameters is present in 'priors' dictionary.
        nuisance_params = [k for k in family.prior.args if k not in ["observed", family.parent]]
        if set(nuisance_params).intersection(set(priors)):
            return {k: priors[k] for k in nuisance_params if k in priors}
    return None


def link_match_family(link, family_name):  # pylint: disable= too-many-return-statements
    """Checks whether the a link can be used in a given family.

    When this function is used with built-in family names, it tests whether the link name can be
    used with the given built-in family. If the family name is not known, we return True because
    the user is working with a custom ``Family`` object.
    Which links can work with which families are taken from statsmodels.
    """
    if family_name == "gaussian":
        return link in ["identity", "log", "inverse"]
    elif family_name == "gamma":
        return link in ["identity", "log", "inverse"]
    elif family_name == "bernoulli":
        return link in ["identity", "logit", "probit", "cloglog"]
    elif family_name == "wald":
        return link in ["inverse", "inverse_squared", "identity", "log"]
    elif family_name == "negativebinomial":
        return link in ["identity", "log", "cloglog"]
    elif family_name == "poisson":
        return link in ["identity", "log"]
    else:  # Custom family, we don't know what link functions can be used
        return True
