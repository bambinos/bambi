from bambi.families.link import Link

from typing import Dict, Union


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        The name of the family. It can be any string.
    likelihood : Likelihood
        A ``bambi.families.Likelihood`` instance specifying the model likelihood function.
    link : Dict[str, Union[str, Link]
        The link function that's used for every parameter in the likelihood function. 
        Keys are the names of the parameters and values are the link functions. 
        These can be a ``str`` with a name or a ``bambi.families.Link`` instance. 
        The link function transforms the linear predictors. 

    Examples
    --------
    FIXME
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

    def __init__(self, name, likelihood, link: Dict[str, Union[str, Link]]) :
        self.name = name
        self.likelihood = likelihood
        self.link = link
        self.aliases = {}

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        assert isinstance(value, dict), "Link functions must be specified with a 'dict'"
        links = {}
        for param_name, param_value in value.items():
            if isinstance(param_value, str):
                param_link = self.check_string_link(param_value)
            elif isinstance(value, Link):
                param_link = param_value
            else:
                raise ValueError(".link must be set to a string or a Link instance.")
            links[param_name] = param_link
        self._link = links

    def check_string_link(self, name):
        if not name in self.SUPPORTED_LINKS:
            raise ValueError(f"Link '{name}' cannot be used with family '{self.name}'")
        return Link(name)

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
        msg_list = [
            f"Family: {self.name}",
            f"Likelihood: {self.likelihood}", 
            f"Link: {self.link}"
        ]
        return "\n".join(msg_list)

    def __repr__(self):
        return self.__str__()
