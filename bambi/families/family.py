from typing import Dict, Union

import numpy as np
import pymc as pm
import xarray as xr

from bambi.families.link import Link
from bambi.utils import get_aliased_name, response_evaluate_new_data


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        The name of the family. It can be any string.
    likelihood : Likelihood
        A `bambi.families.Likelihood` instance specifying the model likelihood function.
    link : Union[str, Dict[str, Union[str, Link]]]
        The link function that's used for every parameter in the likelihood function.
        Keys are the names of the parameters and values are the link functions.
        These can be a `str` with a name or a `bambi.families.Link` instance.
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
        priors = {k: v for k, v in priors.items() if k in self.auxiliary_parameters}
        self.default_priors.update(priors)

    def posterior_predictive(self, model, posterior, **kwargs):
        """Get draws from the posterior predictive distribution

        This function works for almost all the families. It grabs the draws for the parameters
        needed in the response distribution, and then gets samples from the posterior predictive
        distribution using `pm.draw()`. It won't work when the response distribution requires
        parameters that are not available in `posterior`.

        Parameters
        ----------
        model : bambi.Model
            The model
        posterior : xr.Dataset
            The xarray dataset that contains the draws for all the parameters in the posterior.
            It must contain the parameters that are needed in the distribution of the response, or
            the parameters that allow to derive them.
        kwargs :
            Parameters that are used to get draws but do not appear in the posterior object or
            other configuration parameters.
            For instance, the 'n' in binomial models and multinomial models.

        Returns
        -------
        xr.DataArray
            A data array with the draws from the posterior predictive distribution
        """
        response_dist = get_response_dist(model.family)
        response_term = model.response_component.term
        kwargs, coords = self._make_dist_kwargs_and_coords(model, posterior, **kwargs)

        # Handle constrained responses
        if response_term.is_constrained:
            # Bounds are scalars, we can safely pick them from the first row
            lower, upper = response_term.data[0, 1:]
            lower = lower if lower != -np.inf else None
            upper = upper if upper != np.inf else None
            output_array = pm.draw(
                pm.Truncated.dist(response_dist.dist(**kwargs), lower=lower, upper=upper)
            )
        else:
            output_array = pm.draw(response_dist.dist(**kwargs))

        return xr.DataArray(output_array, coords=coords)

    def log_likelihood(self, model, posterior, data, **kwargs):
        """Evaluate the model log-likelihood

        This method uses `pm.logp()`.

        Parameters
        ----------
        model : bambi.Model
            The model
        posterior : xr.Dataset
            The xarray dataset that contains the draws for all the parameters in the posterior.
            It must contain the parameters that are needed in the distribution of the response, or
            the parameters that allow to derive them.
        kwargs :
            Parameters that are used to get draws but do not appear in the posterior object or
            other configuration parameters.
            For instance, the 'n' in binomial models and multinomial models.

        Returns
        -------
        xr.DataArray
            A data array with the value of the log-likelihood for each chain, draw, and value
            of the response variable.
        """
        # Child classes pass "y_values" through the "y" kwarg
        y_values = kwargs.pop("y", None)

        # Get the values of the outcome variable
        if y_values is None:  # when it's not handled by the specific family
            if data is None:
                y_values = np.squeeze(model.response_component.term.data)
            else:
                y_values = response_evaluate_new_data(model, data)

        response_dist = get_response_dist(model.family)
        response_term = model.response_component.term
        kwargs, coords = self._make_dist_kwargs_and_coords(model, posterior, **kwargs)

        # If it's multivariate, it's going to have a fourth coord, but we actually don't need it
        # We just need "chain", "draw", "__obs__"
        coords = dict(list(coords.items())[:3])

        # Handle constrained responses
        if response_term.is_constrained:
            # Bounds are scalars, we can safely pick them from the first row
            lower, upper = response_term.data[0, 1:]
            lower = lower if lower != -np.inf else None
            upper = upper if upper != np.inf else None
            output_array = pm.logp(
                pm.Truncated.dist(response_dist.dist(**kwargs), lower=lower, upper=upper), y_values
            ).eval()
        else:
            output_array = pm.logp(response_dist.dist(**kwargs), y_values).eval()

        return xr.DataArray(output_array, coords=coords)

    def _make_dist_kwargs_and_coords(self, model, posterior, **kwargs):
        """Get kwargs and coordinates

        This utility method generates two things:

        * A dictionary that maps the names of the likelihood parameters to draws from the
        posterior distribtuion.
        * An `xr.Coordinates` object with the coordinates required for the response. For example:
        `(chain, draw, __obs__)` or `(chain, draw, __obs__, y_dim)`.

        It was created to abstract repetitive logic used in both `.posterior_predictive()` and
        `log_likelihood()`.
        """
        # Remove the 'data' kwarg
        kwargs.pop("data", None)

        # Get list of variables to ignore when reshaping
        dont_reshape = kwargs.pop("dont_reshape", [])

        # Collect coordinates from all the likelihood parameters
        params_coords = xr.Coordinates()

        for param in self.likelihood.params:
            # In the posterior xr.Dataset we need to consider aliases, but we can't use aliases
            # when passing kwargs to the PyMC distribution.
            component = model.components[param]
            var_name = component.alias if component.alias else param

            # Get posterior draws or a constant array if it was set to a constant in the prior
            if var_name in posterior:
                kwargs[param] = posterior[var_name].to_numpy()
                params_coords = params_coords.merge(posterior[var_name].coords)
            elif hasattr(component, "prior") and isinstance(component.prior, (int, float)):
                kwargs[param] = np.asarray(component.prior)
            else:
                raise ValueError(
                    "Non-parent parameter not found in posterior."
                    "This error shouldn't have happened!"
                )

        # Determine the array with largest number of dimensions
        ndims_max = max(x.ndim for x in kwargs.values())

        # Append a dimension when needed. Required to make `pm.logp()` and `pm.draw()` work.
        # However, some distributions like Multinomial, require some parameters to be of a smaller
        # dimension than others (n.ndim <= p.ndim - 1) so we don't reshape those.
        for key, values in kwargs.items():
            if key in dont_reshape:
                continue
            kwargs[key] = expand_array(values, ndims_max)

        # Sometimes the model uses a parametrization but we evaluate logp using another one
        if hasattr(model.family, "transform_kwargs"):
            kwargs = model.family.transform_kwargs(kwargs)

        # Get the actual coords as 'params_coords' is an object of type Dataset
        params_coords = params_coords.coords

        coord_names = ["chain", "draw", "__obs__"]
        is_multivariate = hasattr(model.family, "KIND") and model.family.KIND == "Multivariate"

        response_aliased_name = get_aliased_name(model.response_component.term)
        if is_multivariate:
            coord_names.append(response_aliased_name + "_dim")
        elif hasattr(model.family, "create_extra_pps_coord"):
            new_coords = model.family.create_extra_pps_coord()
            coord_names.append(response_aliased_name + "_dim")
            params_coords[response_aliased_name + "_dim"] = new_coords

        coords = {}
        for coord_name in coord_names:
            coords[coord_name] = params_coords[coord_name]

        return kwargs, coords

    def __str__(self):
        msg_list = [f"Family: {self.name}", f"Likelihood: {self.likelihood}", f"Link: {self.link}"]
        return "\n".join(msg_list)

    def __repr__(self):
        return self.__str__()


def get_response_dist(family):
    """Get the PyMC distribution for the response

    Parameters
    ----------
    family : bambi.Family
        The family for which the response distribution is wanted

    Returns
    -------
    pm.Distribution
        The response distribution
    """
    mapping = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical}

    if family.likelihood.dist:
        dist = family.likelihood.dist
    elif family.likelihood.name in mapping:
        dist = mapping[family.likelihood.name]
    else:
        dist = getattr(pm, family.likelihood.name)
    return dist


def expand_array(x, ndim):
    """Add dimensions to an array to match the number of desired dimensions

    If x.ndim < ndim, it adds ndim - x.ndim dimensions after the last axis. If not, it is left
    untouched.

    For example, if we have a normal regression model with n = 1000, chains = 2, and draws = 500
    the shape of the draws of mu will be (2, 500, 1000) but the shape of the draws of sigma will be
    (2, 500). This function makes sure the shape of the draws of sigma is (2, 500, 1) which is
    comaptible with (2, 500, 1000).

    Parameters
    ----------
    x : np.ndarray
        The array
    ndim : int
        The number of desired dimensions

    Returns
    -------
    np.ndarray
        The array with the expanded dimensions
    """
    if x.ndim == ndim:
        return x
    dims_to_expand = tuple(range(ndim - 1, x.ndim - 1, -1))
    return np.expand_dims(x, dims_to_expand)
