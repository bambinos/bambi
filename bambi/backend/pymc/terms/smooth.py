import pymc as pm
import pytensor.tensor as pt

from numpy.typing import ArrayLike

from bambi.priors.prior import Prior
from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.terms.info import SmoothTermInfo
from bambi.backend.pymc.types import Coords, Dims, Shape
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.families.types import ParamSpec
from bambi.terms import SmoothTerm


def flatten_data(data: pt.Variable, coords: Coords) -> pt.Variable:
    if not coords:
        return data
    # The linear predictor is computed with dot(data, params),
    # so named term dimensions are flattened back into design-matrix columns.
    return data.reshape((data.shape[0], -1))


def build_smooth_coefficients(term: SmoothTerm, dims: Dims, model: pm.Model) -> pt.Variable:
    base_shape = () if term.by_levels is None else (len(term.by_levels),)
    curvature_shape = (*base_shape, term.basis_dimension - term.null_space_dimension)

    gaussian_case = all(isinstance(p, Prior) and p.name == "Normal" for p in term.prior.values())

    if gaussian_case:
        # This is the default and most common case, it makes sense to have a specific builder.
        return _build_gaussian_coefficients(term, dims, model, base_shape, curvature_shape)
    return _build_component_coefficients(term, dims, model, base_shape, curvature_shape)


def _build_gaussian_coefficients(
    term: SmoothTerm,
    dims: Dims,
    model: pm.Model,
    base_shape: Shape,
    curvature_shape: Shape,
) -> pt.Variable:
    """Combine constant, linear and curvature priors into one Normal distribution."""
    name = term.label
    mu_blocks, sigma_blocks = [], []

    broadcast_shape = base_shape or (1,)
    hyperprior_shape = () if term.shared else base_shape

    for block_name, block_value in term.prior.items():
        block_mu = block_value.args["mu"]
        block_sigma = block_value.args["sigma"]
        if block_name == "curvature":
            # The term validates that mu is fixed, only sigma can be random.
            block_mu = _broadcast_curvature(block_mu, curvature_shape)
            if isinstance(block_sigma, Prior):
                block_sigma = _build_hyperprior(
                    block_sigma, f"{name}_sigma", model, hyperprior_shape
                )
            block_sigma = _broadcast_curvature(block_sigma, curvature_shape)
        else:
            # Both mu and sigma are constants, as validated by SmoothTerm.
            block_mu = pt.broadcast_to(block_mu, broadcast_shape).reshape((*base_shape, 1))
            block_sigma = pt.broadcast_to(block_sigma, broadcast_shape).reshape((*base_shape, 1))

        mu_blocks.append(block_mu)
        sigma_blocks.append(block_sigma)

    mu = pt.concatenate(mu_blocks, axis=-1)
    sigma = pt.concatenate(sigma_blocks, axis=-1)

    with model:
        rv = pm.Normal(name, mu=mu, sigma=sigma, dims=dims)
    return rv


def _build_component_coefficients(
    term: SmoothTerm,
    dims: Dims,
    model: pm.Model,
    base_shape: Shape,
    curvature_shape: Shape,
) -> pt.Variable:
    """Build each component separately and concatenate their coefficients."""
    name = term.label
    blocks = []
    hyperprior_shape = () if term.shared else base_shape

    for block, component in term.prior.items():
        label = f"{name}_{block}"
        shape = curvature_shape if block == "curvature" else base_shape
        if isinstance(component, Prior):
            kwargs = {}
            for key, parameter in component.args.items():
                if block == "curvature":
                    if isinstance(parameter, Prior):
                        parameter = _build_hyperprior(
                            parameter, f"{label}_{key}", model, hyperprior_shape
                        )
                    parameter = _broadcast_curvature(parameter, curvature_shape)
                else:
                    # Constant and linear components cannot have hyperpriors.
                    parameter = pt.broadcast_to(parameter, base_shape or (1,)).reshape(base_shape)
                kwargs[key] = parameter

            distribution = get_distribution_from_prior(component)
            with model:
                value = distribution(label, **kwargs, shape=shape)
        else:
            if block == "curvature":
                value = _broadcast_curvature(component, curvature_shape)
            else:
                value = pt.broadcast_to(component, base_shape or (1,)).reshape(base_shape)

        if block != "curvature":
            value = value.reshape((*base_shape, 1))
        blocks.append(value)

    with model:
        rv = pm.Deterministic(name, pt.concatenate(blocks, axis=-1), dims=dims)
    return rv


def _broadcast_curvature(value: ArrayLike | pt.Variable, shape: Shape) -> pt.Variable:
    value = pt.as_tensor_variable(value)
    if len(shape) == 2 and value.ndim == 1:
        # Align axes before broadcasting to the target shape.
        value = value[:, None]
    return pt.broadcast_to(value, shape)


def _build_hyperprior(prior: Prior, label: str, model: pm.Model, shape: Shape) -> pt.Variable:
    kwargs = {}
    for key, parameter in prior.args.items():
        if isinstance(parameter, Prior):
            parameter = _build_hyperprior(parameter, f"{label}_{key}", model, shape)
        else:
            # Accept length-one arrays for scalar hyperpriors too.
            parameter = pt.broadcast_to(parameter, shape or (1,)).reshape(shape)
        kwargs[key] = parameter

    distribution = get_distribution_from_prior(prior)
    with model:
        rv = distribution(label, **kwargs, shape=shape)
    return rv


def build_smooth_term(
    term_info: SmoothTermInfo, param_spec: ParamSpec, model: pm.Model
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

    if response_coords:
        raise NotImplementedError("Smooths require scalar response parameters.")

    param_coords = coords | response_coords
    param_dims = tuple(param_coords)
    param = build_smooth_coefficients(term, param_dims, model)

    return flatten_data(model[data_name], coords), param.flatten()
