import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.sparse as ps
import pytensor.tensor as pt
import scipy.sparse as sp
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.model.transform.basic import prune_vars_detached_from_observed
from pymc.pytensorf import replace_vars_in_graphs, toposort_replace

from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.terms.common import shape_prior_arg
from bambi.backend.pymc.terms.info import GroupSpecificTermInfo
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.priors.prior import Prior

from .state import (
    ConditionalParameterInfo,
    DenseGroupSpecificFactorPlan,
    DenseGroupSpecificParameterGraph,
    GroupSpecificGraphState,
    SparseGroupSpecificFactorPlan,
    SparseGroupSpecificParameterGraph,
    make_sparse_matrix_data,
)


def remove_group_specific_contributions(
    group_specific_state: GroupSpecificGraphState, model: pm.Model
) -> pm.Model:
    """Discard group-specific branches from a model clone and prune the detached variables."""
    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    replacements = []

    for parameter_graph in group_specific_state.parameters.values():
        contribution = memo[parameter_graph.contribution]
        contribution_dims = parameter_graph.contribution_dims
        shape = tuple(memo[model.dim_lengths[dim]] for dim in contribution_dims)
        replacements.append((contribution, pt.zeros(shape, dtype=contribution.dtype)))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return prune_vars_detached_from_observed(model)


def add_new_dense_group_specific_contributions(
    plans: list[DenseGroupSpecificFactorPlan],
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pm.Model:
    """Replace dense group-specific lookups for out-of-sample groups."""
    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    replacements = []

    for plan in plans:
        group_idx = model[f"{plan.factor_name}__idx"]
        effective_idx = group_idx
        # `groups_index` uses -1 for missing levels
        unknown_mask = plan.groups_index == -1

        if unknown_mask.any():
            # Share donors across terms to preserve their posterior association.
            p = np.full(plan.groups_n, 1 / plan.groups_n)
            donor_idx = pm.Categorical.dist(p=p, shape=plan.groups_index.shape[0])
            effective_idx = pt.where(unknown_mask, donor_idx, group_idx)

        for term_info in plan.terms:
            term = term_info.term
            coefficients = model[term.label]
            if plan.factor_ndim > 1:
                # `effective_idx` is flat, including for missing-level donors.
                tail_shape = tuple(
                    coefficients.shape[i] for i in range(plan.factor_ndim, coefficients.ndim)
                )
                coefficients = coefficients.reshape((-1, *tail_shape))

            if plan.groups_new:
                new_coefficients = _create_new_group_coefficients(
                    term_info, len(plan.groups_new), plan.factor_ndim, model
                )
                coefficients = pt.concatenate([coefficients, new_coefficients], axis=0)

            replacement = replace_vars_in_graphs([coefficients[effective_idx]], memo)[0]
            parameter_graph = group_specific_state.parameters[plan.parameter_label]

            if not isinstance(parameter_graph, DenseGroupSpecificParameterGraph):
                raise TypeError("Expected a dense group-specific graph.")

            lookup = memo[parameter_graph.terms[term.label].lookup]
            replacements.append((lookup, replacement))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    return model_from_fgraph(fgraph, mutate_fgraph=True)


def add_new_sparse_group_specific_contributions(
    plans: list[SparseGroupSpecificFactorPlan],
    group_specific_state: GroupSpecificGraphState,
    model: pm.Model,
) -> pm.Model:
    """Extend sparse coefficient blocks and add donor contributions for missing groups."""
    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    replacements = []
    plans_by_parameter: dict[str, list[SparseGroupSpecificFactorPlan]] = {}

    for plan in plans:
        plans_by_parameter.setdefault(plan.parameter_label, []).append(plan)

    for parameter_label, parameter_plans in plans_by_parameter.items():
        parameter_graph = group_specific_state.parameters[parameter_label]
        if not isinstance(parameter_graph, SparseGroupSpecificParameterGraph):
            raise TypeError("Expected a sparse group-specific graph.")
        is_univariate = len(parameter_graph.contribution_dims) == 1

        plans_by_factor = {plan.factor_name: plan for plan in parameter_plans}
        coefficient_blocks: list[pt.Variable] = []
        missing_contributions: list[pt.Variable] = []

        for factor_plan in parameter_plans:
            unknown_mask = factor_plan.groups_index == -1
            if not unknown_mask.any():
                continue
            p = np.full(factor_plan.groups_n, 1 / factor_plan.groups_n)
            donor_idx = pm.Categorical.dist(p=p, shape=factor_plan.groups_index.shape[0])

            for term_info in factor_plan.terms:
                missing_contributions.append(
                    _build_sparse_missing_group_contribution(
                        term_info, factor_plan, donor_idx, unknown_mask, memo, model
                    )
                )

        for term_label in parameter_graph.term_labels:
            term_info = next(
                term_info
                for plan in parameter_plans
                for term_info in plan.terms
                if term_info.term.label == term_label
            )
            plan = plans_by_factor[term_info.term.factor_name]
            coefficient_block = _get_sparse_term_coefficients(model[term_label], is_univariate)
            coefficient_block = replace_vars_in_graphs([coefficient_block], memo)[0]

            if plan.groups_new:
                new_coefficients = _create_new_group_coefficients(
                    term_info, len(plan.groups_new), plan.factor_ndim, model
                )
                new_coefficients = _flatten_sparse_coefficient_block(
                    new_coefficients, coefficient_block.ndim
                )
                coefficient_block = pt.concatenate([coefficient_block, new_coefficients], axis=0)

            coefficient_blocks.append(coefficient_block)

        coefficients = pt.concatenate(coefficient_blocks, axis=0)
        if is_univariate:
            coefficients = coefficients[:, np.newaxis]

        sparse_matrix = replace_vars_in_graphs([parameter_graph.matrix], memo)[0]
        contribution = ps.structured_dot(sparse_matrix, coefficients)
        if is_univariate:
            contribution = contribution.squeeze()

        if missing_contributions:
            contribution += sum(missing_contributions)

        old_contribution = replace_vars_in_graphs([parameter_graph.contribution], memo)[0]
        replacements.append((old_contribution, contribution))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    return model_from_fgraph(fgraph, mutate_fgraph=True)


def build_new_dense_conditional_parameter_data(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame, model: pm.Model
) -> tuple[dict[str, np.ndarray], list[DenseGroupSpecificFactorPlan]]:
    """Build dense new-observation data and a plan per grouping factor."""
    data_dict = _build_new_non_group_specific_data(parameter_info, data, model)
    factor_plans = _build_new_dense_group_specific_data(parameter_info, data)
    data_dict.update(_build_new_dense_group_specific_predictors(factor_plans, data, model))
    return data_dict, factor_plans


def build_new_sparse_conditional_parameter_data(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame, model: pm.Model
) -> tuple[dict[str, np.ndarray], list[SparseGroupSpecificFactorPlan], dict[str, range]]:
    """Build sparse new-observation data and a plan per grouping factor."""
    data_dict = _build_new_non_group_specific_data(parameter_info, data, model)
    factor_plans = _build_new_sparse_group_specific_plans(parameter_info, data)
    sparse_data, sparse_coords = _build_new_sparse_group_specific_data(parameter_info, factor_plans)
    data_dict.update(sparse_data)
    return data_dict, factor_plans, sparse_coords


def _build_new_non_group_specific_data(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame, model: pm.Model
) -> dict[str, np.ndarray]:
    data_dict: dict[str, np.ndarray] = {}

    for term_info in parameter_info.common_terms:
        term = term_info.term
        term_data_name = predictor_data_name(term.label, term_info.data_dims, model)
        term_data_dims = model.named_vars_to_dims[term_data_name][1:]  # drop __obs__
        data_dict[term_data_name] = shape_common_data(
            data=term.term.eval_new_data(data),
            coords={dim: model.coords[dim] for dim in term_data_dims},
        )

    for term_info in parameter_info.hsgp_terms:
        term = term_info.term
        term_data = term.term.eval_new_data(data)
        if term.by_levels is not None:
            data_dict[f"{term.label}_by_idx"] = term_data[:, -1].astype(int)
            term_data = term_data[:, :-1]
        data_dict[f"{term.label}_data"] = term_data

    return data_dict


def _build_new_dense_group_specific_data(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame
) -> list[DenseGroupSpecificFactorPlan]:
    factor_plans = []

    for factor_info in parameter_info.group_specific_factors:
        representative = factor_info.terms[0].term
        group_index, new_groups = representative.term.eval_new_data_group_index(data)
        factor_plans.append(
            DenseGroupSpecificFactorPlan(
                parameter_label=parameter_info.label,
                factor_name=factor_info.factor_name,
                factor_ndim=factor_info.factor_ndim,
                terms=factor_info.terms,
                groups_index=group_index,
                groups_new=new_groups,
                groups_n=factor_info.groups_n,
            )
        )

    return factor_plans


def _build_new_dense_group_specific_predictors(
    factor_plans: list[DenseGroupSpecificFactorPlan], data: pd.DataFrame, model: pm.Model
) -> dict[str, np.ndarray]:
    data_dict: dict[str, np.ndarray] = {}

    for factor_plan in factor_plans:
        data_dict[f"{factor_plan.factor_name}__idx"] = factor_plan.groups_index
        for term_info in factor_plan.terms:
            term = term_info.term
            if term.is_intercept:
                continue
            term_value_name = predictor_data_name(
                term.expr_name, ("__obs__", *term_info.expression_coords), model
            )
            term_value_dims = model.named_vars_to_dims[term_value_name][1:]  # drop __obs__
            data_dict[term_value_name] = shape_common_data(
                data=term.expr.eval_new_data(data),
                coords={dim: model.coords[dim] for dim in term_value_dims},
            )

    return data_dict


def _build_new_sparse_group_specific_plans(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame
) -> list[SparseGroupSpecificFactorPlan]:
    factor_plans = []

    for factor_info in parameter_info.group_specific_factors:
        representative = factor_info.terms[0].term
        group_index, new_groups = representative.term.eval_new_data_group_index(data)
        predictors = {
            term_info.term.label: _new_group_specific_predictor(term_info, data)
            for term_info in factor_info.terms
        }
        factor_plans.append(
            SparseGroupSpecificFactorPlan(
                parameter_label=parameter_info.label,
                factor_name=factor_info.factor_name,
                factor_ndim=factor_info.factor_ndim,
                terms=factor_info.terms,
                groups_index=group_index,
                groups_new=new_groups,
                groups_n=factor_info.groups_n,
                predictors=predictors,
            )
        )

    return factor_plans


def _build_new_sparse_group_specific_data(
    parameter_info: ConditionalParameterInfo,
    factor_plans: list[SparseGroupSpecificFactorPlan],
) -> tuple[dict[str, np.ndarray], dict[str, range]]:
    """Build one CSR design matrix from all group-specific terms of a parameter."""
    plans_by_factor = {plan.factor_name: plan for plan in factor_plans}
    blocks: list[sp.csr_matrix] = []

    for term_info in parameter_info.group_specific_terms:
        term = term_info.term
        plan = plans_by_factor[term.factor_name]
        predictor = plan.predictors[term.label]
        blocks.append(_build_sparse_group_specific_block(predictor, plan))

    matrix = sp.hstack(blocks, format="csr")
    sparse_matrix_data = make_sparse_matrix_data(parameter_info.label)
    data_dict = {
        sparse_matrix_data.data_name: matrix.data.astype(pytensor.config.floatX),
        sparse_matrix_data.indices_name: matrix.indices,
        sparse_matrix_data.indptr_name: matrix.indptr,
        sparse_matrix_data.ncols_name: np.asarray(matrix.shape[1]),
    }
    coords = {
        sparse_matrix_data.entry_dim: range(matrix.nnz),
        sparse_matrix_data.indptr_dim: range(matrix.indptr.size),
    }
    return data_dict, coords


def _build_sparse_group_specific_block(
    predictor: np.ndarray, plan: SparseGroupSpecificFactorPlan
) -> sp.csr_matrix:
    """Encode one term using factor-major, expression-minor CSR columns."""
    n_observations = predictor.shape[0]
    predictor = predictor.reshape(n_observations, -1)
    expression_size = predictor.shape[1]
    groups_index = plan.groups_index
    valid = groups_index >= 0
    rows = np.repeat(np.nonzero(valid)[0], expression_size)
    columns = (
        groups_index[valid, np.newaxis] * expression_size + np.arange(expression_size)
    ).ravel()
    values = predictor[valid].ravel()
    n_groups = plan.groups_n + len(plan.groups_new)
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(n_observations, n_groups * expression_size),
    )


def _new_group_specific_predictor(
    term_info: GroupSpecificTermInfo, data: pd.DataFrame
) -> np.ndarray:
    term = term_info.term
    if term.is_intercept:
        return np.ones(len(data))

    return np.asarray(term.expr.eval_new_data(data))


def _build_sparse_missing_group_contribution(
    term_info: GroupSpecificTermInfo,
    plan: SparseGroupSpecificFactorPlan,
    donor_idx: pt.Variable,
    unknown_mask: np.ndarray,
    memo: dict[pt.Variable, pt.Variable],
    model: pm.Model,
) -> pt.Variable:
    """Evaluate rows with missing factor values from a sampled fitted-group donor."""
    term = term_info.term
    coefficients = model[term.label]
    if plan.factor_ndim > 1:
        tail_shape = tuple(
            coefficients.shape[i] for i in range(plan.factor_ndim, coefficients.ndim)
        )
        coefficients = coefficients.reshape((-1, *tail_shape))

    selected = coefficients[donor_idx]
    if term.is_intercept:
        predictor = pt.ones((unknown_mask.size,))
    else:
        predictor = pt.as_tensor_variable(plan.predictors[term.label])

    if selected.ndim > predictor.ndim:
        predictor = predictor[..., np.newaxis]

    contribution = selected * predictor
    expression_ndim = len(term_info.expression_coords)
    if expression_ndim:
        contribution = contribution.sum(axis=tuple(range(1, expression_ndim + 1)))

    mask = pt.as_tensor_variable(unknown_mask)
    if contribution.ndim > 1:
        mask = mask[..., np.newaxis]
    contribution = pt.where(mask, contribution, 0)
    return replace_vars_in_graphs([contribution], memo)[0]


def _flatten_sparse_coefficient_block(coefficients: pt.Variable, ndim: int) -> pt.Variable:
    if ndim == 1:
        return coefficients.flatten()
    return coefficients.reshape((-1, coefficients.shape[-1]))


def _get_sparse_term_coefficients(coefficients: pt.Variable, is_univariate: bool) -> pt.Variable:
    if is_univariate:
        return coefficients.flatten()
    return coefficients.reshape((-1, coefficients.shape[-1]))


def _create_new_group_coefficients(
    term_info: GroupSpecificTermInfo,
    n_new_groups: int,
    factor_ndim: int,
    model: pm.Model,
) -> pt.Variable:
    """Create unregistered population draws for newly identified group levels."""
    term = term_info.term
    term_dims = model.named_vars_to_dims[term.label]
    # Keep expression and response axes after replacing factor axes with new groups.
    tail_dims = term_dims[factor_ndim:]
    tail_shape = tuple(len(model.coords[dim]) for dim in tail_dims)
    kwargs = {}

    for name, value in term.prior.args.items():
        if isinstance(value, Prior):
            # Reuse hyperprior RVs from the fitted model.
            hyperprior_name = term.hyperprior_alias.get(name, name)
            kwargs[name] = model[f"{term.label}_{hyperprior_name}"]
        else:
            # Match fixed prior arguments to the retained axes.
            kwargs[name] = shape_prior_arg(value, tail_shape)

    # Create unregistered draws, one per new group.
    distribution = get_distribution_from_prior(term.prior)
    return distribution.dist(**kwargs, size=(n_new_groups, *tail_shape))
