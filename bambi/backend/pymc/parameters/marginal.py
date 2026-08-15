import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.coords import coords_for_cutpoints
from bambi.backend.pymc.utils import get_distribution_from_prior

TRANSFORMS = {"ordered": pm.distributions.transforms.ordered}


def build_marginal_parameter(parameter, family, model: pm.Model):
    if isinstance(parameter.prior, (int, float)):
        return pm.Deterministic(
            parameter.label, pt.as_tensor_variable(parameter.prior), model=model
        )

    dims = tuple()
    param_spec = family.get_param_spec(parameter.name)
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims = tuple(model.__bambi_attrs__["response_coords"])
        elif param_spec.coefs_dim == "response_reduced":
            dims = tuple(model.__bambi_attrs__["response_coords_reduced"])
        elif param_spec.coefs_dim == "response_cutpoints":
            response_levels = list(model.__bambi_attrs__["response_coords"].values())[0]
            cutpoint_coords = coords_for_cutpoints(parameter.label, response_levels)
            model.add_coords(cutpoint_coords)
            dims = tuple(cutpoint_coords)

    dist = get_distribution_from_prior(parameter.prior)

    kwargs = {}
    for key, value in parameter.prior.args.items():
        if key == "transform" and isinstance(value, str):
            kwargs[key] = TRANSFORMS[value]
        else:
            kwargs[key] = value

    with model:
        rv = dist(parameter.label, **kwargs, dims=dims)
    return rv
