from typing import Dict, Union

from bambi.families.link import Link
from bambi.utils import get_auxiliary_parameters


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        The name of the family. It can be any string.
    likelihood : Likelihood
        A ``bambi.families.Likelihood`` instance specifying the model likelihood function.
    link : Union[str, Dict[str, Union[str, Link]]]
        The link function that's used for every parameter in the likelihood function.
        Keys are the names of the parameters and values are the link functions.
        These can be a ``str`` with a name or a ``bambi.families.Link`` instance.
        The link function transforms the linear predictors.

    Examples
    --------
    >>> import bambi as bmb

    Replicate the Gaussian built-in family.

    >>> sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    >>> likelihood = bmb.Likelihood("Gaussian", params=["mu", "sigma"], parent="mu")
    >>> family = bmb.Family("gaussian", likelihood, "identity")
    >>> bmb.Model("y ~ x", data, family=family, priors={"sigma": sigma_prior})

    Replicate the Bernoulli built-in family.

    >>> likelihood = bmb.Likelihood("Bernoulli", parent="p")
    >>> family = bmb.Family("bernoulli", likelihood, "logit")
    >>> bmb.Model("y ~ x", data, family=family)
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

    def __init__(self, name, likelihood, link: Union[str, Dict[str, Union[str, Link]]]):
        self.name = name
        self.likelihood = likelihood
        self.link = link
        self.default_priors = {}

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        # The name of the link function. It's applied to the parent parameter of the likelihood
        if isinstance(value, (str, Link)):
            value = {self.likelihood.parent: value}
        links = {}
        for name, link in value.items():
            if isinstance(link, str):
                link = self.check_string_link(link, name)
            elif isinstance(link, Link):
                pass
            else:
                raise ValueError("'.link' must be set to a string or a Link instance.")
            links[name] = link
        self._link = links

    def check_string_link(self, link_name, param_name):
        # When you instantiate Family directly
        if isinstance(self.SUPPORTED_LINKS, list):
            supported_links = self.SUPPORTED_LINKS
        else:
            supported_links = self.SUPPORTED_LINKS[param_name]

        if not link_name in supported_links:
            raise ValueError(
                f"Link '{link_name}' cannot be used for '{param_name}' with family "
                f"'{self.name}'"
            )
        return Link(link_name)

    def set_default_priors(self, priors):
        """Set default priors for non-parent parameters

        Parameters
        ----------
        priors : dict
            The keys are the names of non-parent parameters and the values are their default priors.
        """
        auxiliary_parameters = get_auxiliary_parameters(self)
        priors = {k: v for k, v in priors.items() if k in auxiliary_parameters}
        self.default_priors.update(priors)

    def __str__(self):
        msg_list = [f"Family: {self.name}", f"Likelihood: {self.likelihood}", f"Link: {self.link}"]
        return "\n".join(msg_list)

    def __repr__(self):
        return self.__str__()
