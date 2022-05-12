from bambi.families.link import Link


class Family:
    """A specification of model family.

    Parameters
    ----------
    name: str
        The name of the family. It can be any string.
    likelihood: Likelihood
        A ``bambi.families.Likelihood`` instance specifying the model likelihood function.
    link: str or Link
        The name of the link function or a ``bambi.families.Link`` instance. The link function
        transforms the linear model prediction to the mean parameter of the likelihood function.

    Examples
    --------

    >>> import bambi as bmb

    Replicate the Gaussian built-in family.

    >>> sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    >>> likelihood = bmb.Likelihood("Gaussian", parent="mu", sigma=sigma_prior)
    >>> family = bmb.Family("gaussian", likelihood, "identity")
    >>> # Then you can do
    >>> # bmb.Model("y ~ x", data, family=family)

    Replicate the Bernoulli built-in family.

    >>> likelihood = bmb.Likelihood("Bernoulli", parent="p")
    >>> family = bmb.Family("bernoulli", likelihood, "logit")
    """

    SUPPORTED_LINKS = [
        "cloglog",
        "identity",
        "inverse_squared",
        "inverse",
        "log",
        "logit",
        "probit",
        "softmax",
        "tan_2",
    ]

    def __init__(self, name, likelihood, link):
        self.name = name
        self.likelihood = likelihood
        self.link = link
        self.aliases = {}

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, x):
        if isinstance(x, str):
            self.check_string_link(x)
            self._link = Link(x)
        elif isinstance(x, Link):
            self._link = x
        else:
            raise ValueError(".link must be set to a string or a Link instance.")

    def check_string_link(self, link):
        if not link in self.SUPPORTED_LINKS:
            raise ValueError(f"Link '{link}' cannot be used with family '{self.name}'")

    def set_alias(self, name, alias):
        """Set alias for an auxiliary variable of the family

        Parameters
        ----------
        name: str
            The name of the variable
        alias: str
            The new name for the variable
        """
        self.aliases.update({name: alias})

    def __str__(self):
        msg_list = [f"Response distribution: {self.likelihood.name}", f"Link: {self.link.name}"]
        if self.likelihood.priors:
            priors_msg = "\n  ".join([f"{k} ~ {v}" for k, v in self.likelihood.priors.items()])
            msg_list += [f"Priors:\n  {priors_msg}"]
        msg = "\n".join(msg_list)
        return msg

    def __repr__(self):
        return self.__str__()
