from bambi.families.link import LINKS, Link
from bambi.families.types import ParamSpec, ResponseType


class Family:
    """A specification of model family

    Parameters
    ----------
    name : str
        The name of the family. It can be any string.
    likelihood : Likelihood
        A `bambi.families.Likelihood` instance specifying the model likelihood function.
    link : str or dict of str to (str or Link)
        The link function that's used for every parameter in the likelihood function.
        Keys are the names of the parameters and values are the link functions.
        These can be a `str` with a name or a `bambi.families.Link` instance.
        The link function transforms the linear predictors.

    Examples
    --------

    ```python
    import bambi as bmb
    ```

    Replicate the Gaussian built-in family.

    ```python
    sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    likelihood = bmb.Likelihood("Gaussian", params=["mu", "sigma"], parent="mu")
    family = bmb.Family("gaussian", likelihood, "identity")
    bmb.Model("y ~ x", data, family=family, priors={"sigma": sigma_prior})
    ```

    Replicate the Bernoulli built-in family.

    ```python
    likelihood = bmb.Likelihood("Bernoulli", parent="p")
    family = bmb.Family("bernoulli", likelihood, "logit")
    bmb.Model("y ~ x", data, family=family)
    ```
    """

    DATA_TYPE = ResponseType.NUMERIC
    RESPONSE_NDIM = 0
    PARAMETERS: dict[str, ParamSpec] | None = None

    def __init__(self, name, likelihood, link: str | dict[str, str | Link]):
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
            elif not isinstance(link, Link):
                raise ValueError("'.link' must be set to a string or a Link instance.")
            links[name] = link
        self._link = links

    @property
    def auxiliary_parameters(self):
        """Get names of auxiliary parameters

        Obtains the difference between all the parameters and the parent parameter of a family.
        These parameters are known as auxiliary or nuisance parameters.

        Returns
        -------
        set
            Names of auxiliary parameters in the family
        """
        return set(self.likelihood.params) - {self.likelihood.parent}

    def check_string_link(self, link_name, param_name):
        supported_links = self.get_param_spec(param_name).links

        if link_name not in supported_links:
            raise ValueError(
                f"Link '{link_name}' cannot be used for '{param_name}' with family "
                f"'{self.name}'"
            )
        return Link(link_name)

    def get_param_spec(self, param_name):
        """Return metadata for a likelihood parameter.

        Declared metadata is returned unchanged. When no metadata is declared, parameters are
        scalar, have no coefficient dimension, and accept all registered string links.
        """
        parameters = self.PARAMETERS
        if parameters is None:
            return ParamSpec(links=list(LINKS))

        param_spec = parameters.get(param_name)
        if param_spec is None:
            raise KeyError(param_name)
        return param_spec

    def set_default_priors(self, priors):
        """Set default priors for non-parent parameters

        Parameters
        ----------
        priors : dict
            The keys are the names of non-parent parameters and the values are their default priors.
        """
        priors = {k: v for k, v in priors.items() if k in self.auxiliary_parameters}
        self.default_priors.update(priors)

    def __str__(self):
        msg_list = [f"Family: {self.name}", f"Likelihood: {self.likelihood}", f"Link: {self.link}"]
        return "\n".join(msg_list)

    def __repr__(self):
        return self.__str__()
