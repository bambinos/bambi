import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.terms.common import shape_prior_arg
from bambi.backend.pymc.terms.info import GroupSpecificTermInfo
from bambi.backend.pymc.types import Dims
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.priors.prior import Prior
from bambi.families.types import ParamSpec


def build_group_specific_term_dot(
    term_info: GroupSpecificTermInfo, param_spec: ParamSpec, model: pm.Model
) -> pt.Variable:
    """Build a group-specific coefficient block for a shared sparse design matrix."""
    term = term_info.term
    param_name = term.label

    coords = term_info.factor_coords | term_info.expression_coords
    dims_expr = tuple(term_info.expression_coords)
    dims_factor = tuple(term_info.factor_coords)

    # Register coords
    if param_name not in model:
        model.add_coords(coords)

    # Register parameter
    dims_output = tuple()
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims_output = tuple(model.__bambi_attrs__["response_coordsd"])
        elif param_spec.coefs_dim == "response_reduced":
            dims_output = tuple(model.__bambi_attrs__["response_coords_reduced"])

    param_rv = build_distribution(
        prior=term.prior,
        label=param_name,
        dims_expr=dims_expr,
        dims_factor=dims_factor,
        dims_output=dims_output,
        noncentered=term.noncentered,
        hyperprior_aliases=term.hyperprior_alias,
        model=model,
    )

    # If response is multivariate: (q, K)
    # If response is univariate:   (q, )
    if dims_output:
        param_rv = param_rv.reshape((-1, param_rv.shape[-1]))
    else:
        param_rv = param_rv.flatten()

    return param_rv


def build_group_specific_term_idx(
    term_info: GroupSpecificTermInfo, param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, pt.Variable]:
    term = term_info.term
    is_intercept = term.is_intercept
    data_idx_name = f"{term.factor_name}__idx"
    param_name = term.label

    coords = term_info.factor_coords | term_info.expression_coords
    dims_expr = tuple(term_info.expression_coords)
    dims_factor = tuple(term_info.factor_coords)

    # Register coords
    if param_name not in model:
        model.add_coords(coords)

    # Register data: predictor
    predictor_data = None
    if not is_intercept:
        predictor_dims = ("__obs__",) + dims_expr
        data_value_name = predictor_data_name(term.expr_name, predictor_dims, model)
        if data_value_name in model:
            predictor_data = model[data_value_name]
        else:
            predictor = shape_common_data(term.predictor, term_info.expression_coords)
            predictor_data = pm.Data(data_value_name, predictor, dims=predictor_dims, model=model)

    # Register data: group index
    if data_idx_name in model:
        group_idx_data = model[data_idx_name]
    else:
        group_idx_data = pm.Data(data_idx_name, term.group_index, dims=("__obs__",), model=model)

    # Register parameter
    dims_output = tuple()
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims_output = tuple(model.__bambi_attrs__["response_coordsd"])
        elif param_spec.coefs_dim == "response_reduced":
            dims_output = tuple(model.__bambi_attrs__["response_coords_reduced"])

    param_rv = build_distribution(
        prior=term.prior,
        label=param_name,
        dims_factor=dims_factor,
        dims_expr=dims_expr,
        dims_output=dims_output,
        noncentered=term.noncentered,
        hyperprior_aliases=term.hyperprior_alias,
        model=model,
    )

    if len(dims_factor) > 1:
        tail_shape = tuple(param_rv.shape[i] for i in range(len(dims_factor), param_rv.ndim))
        param_rv = param_rv.reshape((-1, *tail_shape))

    selected_param = param_rv[group_idx_data]

    if is_intercept:
        return selected_param, selected_param

    if dims_output and predictor_data is not None:
        # (n, )    -> (n, 1)
        # (n, q_j) -> (n, q_j, 1)
        predictor_data = predictor_data[..., np.newaxis]

    # (n, ) * (n, )             -> (n, )
    # (n, q_j) * (n, q_j)       -> (n, q_j)
    # (n, K) * (n, 1)           -> (n, K)
    # (n, q_j, K) * (n, q_j, 1) -> (n, q_j, K)
    contribution = selected_param * predictor_data
    if dims_expr:
        axes = tuple(range(1, len(dims_expr) + 1))
        contribution = contribution.sum(axis=axes)

    return selected_param, contribution


def build_distribution(
    prior: Prior,
    label: str,
    dims_factor: Dims,
    dims_expr: Dims,
    dims_output: Dims,
    noncentered: bool,
    hyperprior_aliases: dict[str, str] | None,
    model: pm.Model,
) -> pt.Variable:
    kwargs = {}
    hyperprior_aliases = hyperprior_aliases or {}
    # From slowest to fastest changing
    dims = dims_factor + dims_expr + dims_output
    shape = tuple(len(model.coords[dim]) for dim in dims)

    for name, value in prior.args.items():
        if isinstance(value, Prior):
            hyperparam_label = f"{label}_{hyperprior_aliases.get(name, name)}"
            kwargs[name] = build_distribution(
                prior=value,
                label=hyperparam_label,
                dims_factor=tuple(),
                dims_expr=dims_expr,
                dims_output=dims_output,
                noncentered=noncentered,
                hyperprior_aliases=None,
                model=model,
            )
        else:
            kwargs[name] = shape_prior_arg(value, shape)

    effective_noncentered = prior.noncentered if prior.noncentered is not None else noncentered

    if effective_noncentered and any(isinstance(v, pt.TensorVariable) for v in kwargs.values()):
        # non-centered is only relevant when distribution arguments are random variables.
        if prior.name == "Normal" and isinstance(kwargs.get("sigma", None), pt.TensorVariable):
            sigma = kwargs["sigma"]
            with model:
                offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims)
                model.__bambi_attrs__["offset_names"].add(label + "_offset")
                rv = pm.Deterministic(label, offset * sigma, dims=dims)
            return rv

        raise NotImplementedError(
            "The non-centered parametrization is only supported for Normal priors, "
            f"got {prior.name}."
        )

    dist = get_distribution_from_prior(prior)

    with model:
        rv = dist(label, **kwargs, dims=dims)

    return rv
