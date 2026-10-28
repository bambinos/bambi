import numpy as np
import xarray as xr

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

    CIRCULAR = False

    def conditional_moments(self, model, posterior, data=None):
        r"""Get the conditional mean and variance of the response

        Computes the mean and the variance of the response conditional on the parameters,
        :math:`E(Y \mid \theta)` and :math:`\mathrm{Var}(Y \mid \theta)`, for every posterior
        draw and observation. Both are on the scale of the response (i.e. after applying the
        inverse link functions). This is used, for example, to compute the Bayesian :math:`R^2`.

        Parameters
        ----------
        model : bambi.Model
            The model
        posterior : xr.Dataset
            The xarray dataset that contains the draws for all the parameters in the posterior.
            It must contain the parameters of the response distribution, as added by
            `model.predict(idata, kind="response_params")`.
        data : pd.DataFrame or None
            Only used to get the number of trials in binomial-type models. If omitted, the
            original dataset is used. It must match the observations in `posterior`.

        Returns
        -------
        tuple of xr.DataArray
            The conditional mean and the conditional variance of the response.

        Raises
        ------
        NotImplementedError
            If the conditional moments are not implemented or not defined for the family.
        """
        params = {}
        for param in self.likelihood.params:
            parameter = model.parameters[param]
            var_name = parameter.alias if parameter.alias else param
            if var_name in posterior:
                params[param] = posterior[var_name]
            elif isinstance(getattr(parameter, "prior", None), (int, float)):
                params[param] = xr.DataArray(float(parameter.prior))
            else:
                raise ValueError(
                    f"The parameter '{var_name}' was not found in the posterior. "
                    "Use `model.predict(idata, kind='response_params')` first."
                )

        if model.response_term.is_binomial:
            if data is None:
                trials = model.response_term.data[:, 1]
            else:
                trials = model.response_term.eval_new_data(data)[:, 1]
            params["n"] = xr.DataArray(np.asarray(trials, dtype=float), dims="__obs__")

        return self.moments(params)

    def moments(self, params):  # pylint: disable=unused-argument
        """Compute the conditional mean and variance of the response

        Families that support the Bayesian :math:`R^2` override this method.

        Parameters
        ----------
        params : dict of str to xr.DataArray
            Maps the names of the parameters of the likelihood to their values on the response
            scale (i.e. after applying the inverse link functions). Binomial-type families also
            get the number of trials as `"n"`.

        Returns
        -------
        tuple of xr.DataArray
            The conditional mean and the conditional variance of the response. Families with a
            circular response (`CIRCULAR = True`) return the circular variance instead.

        Raises
        ------
        NotImplementedError
            When the moments are not defined or not implemented for the family.
        """
        if self.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
            reason = "has a categorical response, so the conditional variance is not defined"
        elif self.RESPONSE_NDIM > 0:
            reason = "has a multivariate response, so a scalar conditional variance is not defined"
        else:
            reason = "does not implement the conditional mean and variance of the response"
        raise NotImplementedError(f"The '{self.name}' family {reason}.")

    def __str__(self):
        msg_list = [f"Family: {self.name}", f"Likelihood: {self.likelihood}", f"Link: {self.link}"]
        return "\n".join(msg_list)

    def __repr__(self):
        return self.__str__()
