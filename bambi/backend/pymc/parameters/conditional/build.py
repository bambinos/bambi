import numpy as np
import pymc as pm
import pytensor
import pytensor.sparse as ps
import pytensor.tensor as pt
import scipy.sparse as sp

from bambi.backend.pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_group_specific_term_idx,
    build_hsgp_term,
    build_intercept_term,
)
from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.terms.info import CommonTermInfo, GroupSpecificTermInfo
from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.config import config as bmb_config
from bambi.families import Family
from bambi.families.types import ParamSpec
from bambi.terms import CommonTerm

from .state import (
    ConditionalParameterInfo,
    DenseGroupSpecificParameterGraph,
    DenseGroupSpecificTermGraph,
    GroupSpecificGraphState,
    SparseGroupSpecificParameterGraph,
    make_sparse_matrix_data,
)


def build_conditional_parameter(
    parameter_info: ConditionalParameterInfo,
    family: Family,
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    parameter = parameter_info.parameter
    param_spec = family.get_param_spec(parameter.name)
    link = family.link[parameter.name]
    inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
    transform_predictor = transforms_registry.get_predictor_transform(family, parameter.name)
    transform_parameters = {}
    if transform_predictor:
        transform_parameters = {
            name: model[name] for name in family.likelihood.params if name != parameter.name
        }
    return build_additive_parameter(
        parameter_info,
        param_spec,
        inverse_link,
        transform_predictor,
        transform_parameters,
        group_specific_state,
        model,
    )


def build_nonlinear_predictor(
    parameter_info: ConditionalParameterInfo,
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    param_spec = ParamSpec(links=["identity"])
    return build_additive_parameter(
        parameter_info,
        param_spec,
        INVERSE_LINKS["identity"],
        None,
        {},
        group_specific_state,
        model,
    )


def build_additive_parameter(
    parameter_info: ConditionalParameterInfo,
    param_spec: ParamSpec,
    inverse_link,
    transform_predictor,
    transform_parameters,
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    parameter = parameter_info.parameter
    value = 0
    center_predictors = parameter.intercept_term and parameter.center_predictors

    if parameter_info.common_terms or parameter.intercept_term:
        value += _build_common_and_intercept(
            common_terms=parameter_info.common_terms,
            intercept_term=parameter.intercept_term,
            center=center_predictors,
            param_spec=param_spec,
            model=model,
        )

    if parameter_info.group_specific_terms:
        group_specific_contribution = _build_group_specific(
            parameter_info=parameter_info,
            param_spec=param_spec,
            group_specific_state=group_specific_state,
            model=model,
        )
        value += group_specific_contribution

    value += _build_offset_terms(parameter_info.offset_terms, model)

    for term_info in parameter_info.hsgp_terms:
        value += build_hsgp_term(term_info, param_spec, model)

    if transform_predictor:
        value = transform_predictor(value, transform_parameters, inverse_link)
    else:
        value = inverse_link(value)

    coords = model.__bambi_attrs__["response_coords_data"]
    if param_spec.ndim > 0:
        coords = coords | model.__bambi_attrs__["response_coords"]

    dims = tuple(coords)
    only_intercept = (
        parameter.intercept_term
        and not parameter.common_terms
        and not parameter.group_specific_terms
        and not parameter.offset_terms
        and not parameter.hsgp_terms
    )
    value = pt.as_tensor_variable(value)
    if value.ndim < len(dims) or only_intercept:
        value = pt.broadcast_to(value, tuple(model.dim_lengths[dim] for dim in dims))
    return pm.Deterministic(parameter.label, value, dims=dims, model=model)


_ENSURE_NDIM_MAPPING = {
    0: pt.atleast_1d,
    1: pt.atleast_2d,
}


def _ensure_2d(x: pt.Variable) -> pt.Variable:
    # Concatenation requires data arrays to be all 2d
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _build_offset_terms(offset_terms: tuple[CommonTermInfo, ...], model: pm.Model) -> pt.Variable:
    contribution = 0
    for term_info in offset_terms:
        term = term_info.term
        data_name = predictor_data_name(term.label, term_info.data_dims, model)

        if data_name not in model:
            model.add_coords(term_info.coords)
            data = shape_common_data(term.data, term_info.coords)
            pm.Data(data_name, data, dims=term_info.data_dims, model=model)

        contribution += model[data_name]
    return contribution


def _build_common_and_intercept(
    common_terms: tuple[CommonTermInfo, ...],
    intercept_term: CommonTerm | None,
    center: bool,
    param_spec: ParamSpec,
    model: pm.Model,
) -> pt.Variable:
    # Build common terms, then build intercept
    ndim = 0 if param_spec.coefs_dim is None else 1
    ensure_ndim = _ENSURE_NDIM_MAPPING[ndim]
    data_mean = None
    params = None
    intercept_contribution = 0
    common_contribution = 0

    if common_terms:
        data_list = []
        param_list = []

        for term_info in common_terms:
            data, param = build_common_term(term_info, param_spec, model)
            data_list.append(_ensure_2d(data))
            param_list.append(ensure_ndim(param))

        params = pt.concatenate(param_list, axis=0)  # (p, ) or (p, K)
        data = pt.concatenate(data_list, axis=1)  # (n, p)

        if center:
            # .eval() is required to not recompute the mean in out-of-sample predictions.
            data_mean = data.mean(0).eval()
            data = data - data_mean

        # (n, ) or (n, K)
        common_contribution = pt.dot(data, params)

    if intercept_term:
        intercept_contribution = ensure_ndim(
            build_intercept_term(intercept_term, data_mean, params, param_spec, model)
        )

    return intercept_contribution + common_contribution


def _build_group_specific(
    parameter_info: ConditionalParameterInfo,
    param_spec: ParamSpec,
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    terms = parameter_info.group_specific_terms
    contribution_dims = ("__obs__",)

    if param_spec.coefs_dim == "response":
        contribution_dims += tuple(model.__bambi_attrs__["response_coords"])
    elif param_spec.coefs_dim == "response_reduced":
        contribution_dims += tuple(model.__bambi_attrs__["response_coords_reduced"])

    if bmb_config["SPARSE_DOT"]:
        return _build_sparse_group_specific(
            parameter_info, param_spec, contribution_dims, group_specific_state, model
        )

    return _build_dense_group_specific(
        parameter_info, terms, param_spec, contribution_dims, group_specific_state, model
    )


def _build_sparse_group_specific(
    parameter_info: ConditionalParameterInfo,
    param_spec: ParamSpec,
    contribution_dims: tuple[str, ...],
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    contribution, term_labels, matrix = _build_sparse_group_specific_dot(
        parameter_info, param_spec, model
    )
    group_specific_state.parameters[parameter_info.label] = SparseGroupSpecificParameterGraph(
        contribution=contribution,
        contribution_dims=contribution_dims,
        term_labels=term_labels,
        matrix=matrix,
    )
    return contribution


def _build_dense_group_specific(
    parameter_info: ConditionalParameterInfo,
    terms: tuple[GroupSpecificTermInfo, ...],
    param_spec: ParamSpec,
    contribution_dims: tuple[str, ...],
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pt.Variable:
    contribution, term_graphs = _build_dense_group_specific_idx(terms, param_spec, model)
    group_specific_state.parameters[parameter_info.label] = DenseGroupSpecificParameterGraph(
        contribution=contribution,
        contribution_dims=contribution_dims,
        terms=term_graphs,
    )
    return contribution


def _build_sparse_group_specific_dot(
    parameter_info: ConditionalParameterInfo, param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, tuple[str, ...], pt.Variable]:
    terms = parameter_info.group_specific_terms
    data = sp.hstack([term_info.term.data for term_info in terms], format="csr")
    sparse_data = make_sparse_matrix_data(parameter_info.label)
    data_buffer = pm.Data(
        sparse_data.data_name,
        data.data.astype(pytensor.config.floatX),
        dims=sparse_data.entry_dim,
        model=model,
    )
    indices_buffer = pm.Data(
        sparse_data.indices_name, data.indices, dims=sparse_data.entry_dim, model=model
    )
    indptr_buffer = pm.Data(
        sparse_data.indptr_name, data.indptr, dims=sparse_data.indptr_dim, model=model
    )
    ncols_buffer = pm.Data(sparse_data.ncols_name, np.asarray(data.shape[1]), model=model)
    matrix = ps.CSR(
        data_buffer,
        indices_buffer,
        indptr_buffer,
        pt.stack([model.dim_lengths["__obs__"], ncols_buffer]),
    )

    param_blocks = []
    term_labels = []
    for term_info in terms:
        param = build_group_specific_term_dot(term_info, param_spec, model)
        param_blocks.append(param)
        term_labels.append(term_info.term.label)

    # Coefficients array: shape (q, ) or (q, K)
    coefs = pt.concatenate(param_blocks, axis=0)

    is_univariate = coefs.ndim == 1
    if is_univariate:
        # PyTensor expects 2D
        coefs = coefs[:, np.newaxis]

    # (n, ) or (n, K)
    dot_output = ps.structured_dot(matrix, coefs)
    if is_univariate:
        return dot_output.squeeze(), tuple(term_labels), matrix

    return dot_output, tuple(term_labels), matrix


def _build_dense_group_specific_idx(
    terms: tuple[GroupSpecificTermInfo, ...], param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, dict[str, DenseGroupSpecificTermGraph]]:
    contribution = 0
    term_graphs = {}
    for term_info in terms:
        lookup, term_contribution = build_group_specific_term_idx(term_info, param_spec, model)
        term_graphs[term_info.term.label] = DenseGroupSpecificTermGraph(lookup=lookup)
        contribution += term_contribution
    return contribution, term_graphs
