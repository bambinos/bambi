import numpy as np
import pymc as pm
import xarray as xr

from bambi.utils import get_aliased_name


def get_response_dist(family):
    """Get the PyMC distribution for the response
    
    Parameters
    ----------
    family : bambi.Family
        The family for which the response distribution is wanted

    Returns
    -------
    graphviz.Digraph
        The graph
    """
    if family.likelihood.dist:
        dist = family.likelihood.dist
    else:
        dist = getattr(pm, family.likelihood.name)
    return dist


def expand_array(x, ndim):
    if x.ndim == ndim:
        return x
    dims_to_expand = tuple(range(ndim - 1, x.ndim - 1, -1))
    return np.expand_dims(x, dims_to_expand)


def get_posterior_predictive_draws(model, posterior):
    response_dist = get_response_dist(model.family)
    params = model.family.likelihood.params
    response_aliased_name = get_aliased_name(model.response_component.response_term)

    kwargs = {}
    output_dataset_list = []

    # In the posterior xr.Dataset we need to consider aliases.
    # But we don't use aliases when passing kwargs to the PyMC distribution
    for param in params:
        # Extract posterior draws for the parent parameter
        if param == model.family.likelihood.parent:
            component = model.components[model.response_name]
            var_name = f"{response_aliased_name}_mean"
            kwargs[param] = posterior[var_name].to_numpy()
            output_dataset_list.append(posterior[var_name])
        else:
            # Extract posterior draws for non-parent parameters
            component = model.components[param]
            component_aliased_name = component.alias if component.alias else param
            var_name = f"{response_aliased_name}_{component_aliased_name}"
            if var_name in posterior:
                kwargs[param] = posterior[var_name].to_numpy()
                output_dataset_list.append(posterior[var_name])
            elif hasattr(component, "prior") and isinstance(component.prior, (int, float)):
                kwargs[param] = np.asarray(component.prior)

    # Determine the array with largest number of dimensions
    ndims_max = max(x.ndim for x in kwargs.values())

    # Append a dimension when needed. Required to make `pm.draw()` work.
    for key, values in kwargs.items():
        kwargs[key] = expand_array(values, ndims_max)

    # NOTE: Wouldn't it be better to always use parametrizations compatible with PyMC?
    # The current approach allows more flexibility, but it's more painful.
    if hasattr(model.family, "transform_backend_kwargs"):
        kwargs = model.family.transform_backend_kwargs(kwargs)

    output_array = pm.draw(response_dist.dist(**kwargs))
    output_coords = xr.merge(output_dataset_list).coords

    # Sometimes `output_array` has less dimensions than coords `output_coords`
    # An example is the categorical family.
    # This seems to work, but we should be open to better alternatives in the future
    # NOTE: Some dimension information about the response distribution could be taken from
    # response_dist.dist().ndim
    output_coords_filtered = {}
    for i, (name, values) in enumerate(output_coords.items()):
        output_coords_filtered[name] = values
        if i == output_array.ndim - 1:
            break
    return xr.DataArray(output_array, coords=output_coords_filtered)
