from statsmodels.genmod import families as sm_families

from bambi.families.link import Link

STATSMODELS_FAMILIES = {
    "bernoulli": sm_families.Binomial,
    "gamma": sm_families.Gamma,
    "gaussian": sm_families.Gaussian,
    "wald": sm_families.InverseGaussian,
    "negativebinomial": sm_families.NegativeBinomial,
    "poisson": sm_families.Poisson,
}

STATSMODELS_LINKS = {
    "identity": sm_families.links.identity(),
    "logit": sm_families.links.logit(),
    "probit": sm_families.links.probit(),
    "cloglog": sm_families.links.cloglog(),
    "inverse": sm_families.links.inverse_power(),
    "inverse_squared": sm_families.links.inverse_squared(),
    "log": sm_families.links.log(),
}


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        The name of the family. It can be any string.
    likelihood: Likelihood
        A ``bambi.families.Likelihood`` instace specifying the model likelihood function.
    link : str or Link
        The name of the link function or a ``bambi.families.Link`` instance. The link function
        transforms the linear model prediction to the mean parameter of the likelihood funtion.

    Examples
    --------

    >>> import bambi as bmb

    Replicate the Gaussian built-in family.

    >>> sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    >>> likelihood = bmb.Likelihood("Gaussian", parent="mu", sigma=sigma_prior)
    >>> family = bmb.Family("my_gaussian", likelihood, "identity")
    >>> # Then you can do
    >>> # bmb.Model("y ~ x", data, family=family)

    Replicate the Bernoulli built-in family.

    >>> likelihood = bmb.Likelihood("Bernoulli", parent="p")
    >>> family = bmb.Family("bernoulli2", likelihood, "logit")
    """

    def __init__(self, name, likelihood, link):
        self.smlink = None
        self.link = None
        self.name = name
        self.likelihood = likelihood
        self.smfamily = STATSMODELS_FAMILIES.get(name, None)
        self._set_link(link)

    def _set_link(self, link):
        """Set new link function.

        If ``link`` is of type ``str``, this method attempts to create a ``bambi.families.Link``
        from the name passed. If it is a recognized name, a builtin ``bambi.families.Link`` will be
        used. Otherwise, ``bambi.families.Link`` instantiation will raise an error.

        Parameters
        ----------
        link: str or bambi.families.Link
            If a string, it must the name of a link function recognized by Bambi.

        Returns
        -------
        None
        """
        if isinstance(link, str):
            self.link = Link(link)
            self.smlink = STATSMODELS_LINKS.get(link, None)
        elif isinstance(link, Link):
            self.link = link
        else:
            raise ValueError("'link' must be a string or a Link instance.")

    def __str__(self):
        msg_list = [f"Response distribution: {self.likelihood.name}", f"Link: {self.link.name}"]
        if self.likelihood.priors:
            priors_msg = "\n  ".join([f"{k} ~ {v}" for k, v in self.likelihood.priors.items()])
            msg_list += [f"Priors:\n  {priors_msg}"]
        msg = "\n".join(msg_list)
        return msg

    def __repr__(self):
        return self.__str__()


# Names of parameters that can receive a prior distribution for the built-in families
FAMILY_PARAMS = {
    "beta": ("kappa",),
    "gamma": ("alpha",),
    "gaussian": ("sigma",),
    "negativebinomial": ("alpha",),
    "t": ("lam", "nu"),
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
