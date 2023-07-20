from typing import Dict, Union

import numpy as np
import pymc as pm
import xarray as xr

from bambi.families.link import Link
from bambi.utils import get_auxiliary_parameters, get_aliased_name


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
        params = model.family.likelihood.params
        response_aliased_name = get_aliased_name(model.response_component.response_term)

        kwargs.pop("data", None)  # Remove the 'data' kwarg
        dont_reshape = kwargs.pop("dont_reshape", [])
        output_dataset_list = []

        # In the posterior xr.Dataset we need to consider aliases,
        # but we don't use aliases when passing kwargs to the PyMC distribution.
        for param in params:
            # Extract posterior draws for the parent parameter
            if param == model.family.likelihood.parent:
                component = model.components[model.response_name]
                var_name = response_aliased_name + "_mean"
                kwargs[param] = posterior[var_name].to_numpy()
                output_dataset_list.append(posterior[var_name])
            else:
                # Extract posterior draws for non-parent parameters
                component = model.components[param]
                if component.alias:
                    var_name = component.alias
                else:
                    var_name = f"{response_aliased_name}_{param}"

                if var_name in posterior:
                    kwargs[param] = posterior[var_name].to_numpy()
                    output_dataset_list.append(posterior[var_name])
                elif hasattr(component, "prior") and isinstance(component.prior, (int, float)):
                    kwargs[param] = np.asarray(component.prior)
                else:
                    raise ValueError(
                        "Non-parent parameter not found in posterior."
                        "This error shouldn't have happened!"
                    )

        # Determine the array with largest number of dimensions
        ndims_max = max(x.ndim for x in kwargs.values())

        # Append a dimension when needed. Required to make `pm.draw()` work.
        # However, some distributions like Multinomial, require some parameters to be of a smaller
        # dimension than others (n.ndim <= p.ndim - 1) so we don't reshape those.
        for key, values in kwargs.items():
            if key in dont_reshape:
                continue
            kwargs[key] = expand_array(values, ndims_max)

        if hasattr(model.family, "transform_kwargs"):
            kwargs = model.family.transform_kwargs(kwargs)

        output_array = pm.draw(response_dist.dist(**kwargs))
        output_coords_all = xr.merge(output_dataset_list).coords

        coord_names = ["chain", "draw", response_aliased_name + "_obs"]
        is_multivariate = hasattr(model.family, "KIND") and model.family.KIND == "Multivariate"

        if is_multivariate:
            coord_names.append(response_aliased_name + "_dim")
        elif hasattr(model.family, "create_extra_pps_coord"):
            new_coords = model.family.create_extra_pps_coord()
            coord_names.append(response_aliased_name + "_dim")
            output_coords_all[response_aliased_name + "_dim"] = new_coords

        output_coords = {}
        for coord_name in coord_names:
            output_coords[coord_name] = output_coords_all[coord_name]
        return xr.DataArray(output_array, coords=output_coords)

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
