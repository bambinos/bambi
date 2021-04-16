def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def extract_family_prior(family, priors):
    """
    Given a family and a dictionary of priors, return the prior that could be
    passed to update the prior of a parameter in that family.
    """

    # Only gaussian and negativebinomial have nuisance parameters
    # i.e, parameters not related to the link-transformed predicted outcome
    if family == "gaussian" and "sigma" in priors:
        return {"sigma": priors["sigma"]}
    elif family == "negativebinomial" and "mu" in priors:
        return {"mu": priors["mu"]}
    else:
        return None
