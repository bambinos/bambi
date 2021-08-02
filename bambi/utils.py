FAMILY_LINKS = {
    "bernoulli": ["identity", "logit", "probit", "cloglog"],
    "beta": ["identity", "logit", "probit", "cloglog"],
    "binomial": ["identity", "logit", "probit", "cloglog"],
    "gamma": ["identity", "log", "inverse"],
    "gaussian": ["identity", "log", "inverse"],
    "negativebinomial": ["identity", "log", "cloglog"],
    "poisson": ["identity", "log"],
    "t": ["identity", "log", "inverse"],
    "wald": ["inverse", "inverse_squared", "identity", "log"],
}


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def spacify(string):
    """Add 2 spaces to the beginning of each line in a multi-line string."""
    return "  " + "  ".join(string.splitlines(True))


def multilinify(sequence, sep=","):
    """Make a multi-line string out of a sequence of strings."""
    sep += "\n"
    return "\n" + sep.join(sequence)


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
