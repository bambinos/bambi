import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.backend.pymc.types import Coords
from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.families.types import ParamSpec


def flatten_data(data: pt.Variable, coords: Coords) -> pt.Variable:
    if not coords:
        return data
    # The linear predictor is computed with dot(data, params),
    # so named term dimensions are flattened back into design-matrix columns.
    return data.reshape((data.shape[0], -1))


def flatten_param(param: pt.Variable, term_coords: Coords, response_coords: Coords) -> pt.Variable:
    if not term_coords:
        return param

    # Match the flattened term data: dimensions collapse into one coefficient axis,
    # while response dimensions stay separate.
    response_shape = tuple(len(coord) for coord in response_coords.values())
    if response_shape:
        return param.reshape((-1, *response_shape))
    return param.reshape((-1,))


def shape_prior_arg(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if value.shape == shape:
        return value

    if value.size == np.prod(shape):
        # Flat prior arguments are interpreted in the same coordinate order as the parameter.
        return value.reshape(shape)

    if value.shape == shape[: value.ndim]:
        # Term-shaped prior arguments broadcast across response dimensions.
        value = value.reshape((*value.shape, *(1 for _ in shape[value.ndim :])))

    return np.broadcast_to(value, shape)


def build_common_term(
    term_info, param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, pt.Variable]:
    term = term_info.term
    param_name = term.label
    coords = term_info.coords
    data_name = predictor_data_name(term.label, term_info.data_dims, model)

    # Register coords
    if data_name not in model or param_name not in model:
        model.add_coords(coords)

    # Register data
    if data_name not in model:
        data = shape_common_data(term.data, coords)
        pm.Data(data_name, data, dims=term_info.data_dims, model=model)

    # Register parameter
    response_coords = {}
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            response_coords = model.__bambi_attrs__["response_coords"]
        elif param_spec.coefs_dim == "response_reduced":
            response_coords = model.__bambi_attrs__["response_coords_reduced"]

    param_coords = coords | response_coords
    param_dims = tuple(param_coords)
    param_shape = tuple(len(coord) for coord in param_coords.values())

    # Makes sure arguments are of the shape implied by dims and their coords
    kwargs = {name: shape_prior_arg(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        dist(param_name, **kwargs, dims=param_dims)

    data = flatten_data(model[data_name], coords)
    param = flatten_param(model[param_name], coords, response_coords)
    return data, param
