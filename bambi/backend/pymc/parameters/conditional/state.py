from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from bambi.backend.pymc.coords import coords_from_common, coords_from_hsgp
from bambi.backend.pymc.terms.info import (
    CommonTermInfo,
    GroupSpecificFactorInfo,
    GroupSpecificTermInfo,
    HSGPTermInfo,
)
from bambi.backend.pymc.types import Dims
from bambi.parameters import ConditionalParameter


@dataclass(frozen=True)
class ConditionalParameterInfo:
    parameter: ConditionalParameter
    common_terms: tuple[CommonTermInfo, ...]
    offset_terms: tuple[CommonTermInfo, ...]
    hsgp_terms: tuple[HSGPTermInfo, ...]
    group_specific_factors: tuple[GroupSpecificFactorInfo, ...]

    @property
    def label(self) -> str:
        return self.parameter.label

    @property
    def group_specific_terms(self) -> tuple[GroupSpecificTermInfo, ...]:
        return tuple(term for factor in self.group_specific_factors for term in factor.terms)


@dataclass(frozen=True)
class DenseGroupSpecificFactorPlan:
    parameter_label: str
    factor_name: str
    factor_ndim: int
    terms: tuple[GroupSpecificTermInfo, ...]
    groups_index: np.ndarray
    groups_new: tuple[object, ...]
    groups_n: int


@dataclass(frozen=True)
class SparseGroupSpecificFactorPlan(DenseGroupSpecificFactorPlan):
    predictors: dict[str, np.ndarray]


@dataclass(frozen=True)
class SparseMatrixData:
    """Names and dimensions for the dense buffers of a CSR design matrix."""

    data_name: str
    indices_name: str
    indptr_name: str
    ncols_name: str
    entry_dim: str
    indptr_dim: str


def make_sparse_matrix_data(parameter_label: str) -> SparseMatrixData:
    """Create names unique to one conditional parameter's sparse design matrix."""
    prefix = f"{parameter_label}__group_specific"
    return SparseMatrixData(
        data_name=f"{prefix}_data",
        indices_name=f"{prefix}_indices",
        indptr_name=f"{prefix}_indptr",
        ncols_name=f"{prefix}_ncols",
        entry_dim=f"{prefix}_entry",
        indptr_dim=f"{prefix}_indptr_dim",
    )


@dataclass
class DenseGroupSpecificTermGraph:
    lookup: pt.Variable

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "DenseGroupSpecificTermGraph":
        return DenseGroupSpecificTermGraph(lookup=memo[self.lookup])


@dataclass
class DenseGroupSpecificParameterGraph:
    contribution: pt.Variable
    contribution_dims: Dims
    terms: dict[str, DenseGroupSpecificTermGraph]

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "DenseGroupSpecificParameterGraph":
        return DenseGroupSpecificParameterGraph(
            contribution=memo[self.contribution],
            contribution_dims=self.contribution_dims,
            terms={label: term.clone(memo) for label, term in self.terms.items()},
        )


@dataclass
class SparseGroupSpecificParameterGraph:
    contribution: pt.Variable
    contribution_dims: Dims
    term_labels: tuple[str, ...]
    matrix: pt.Variable

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "SparseGroupSpecificParameterGraph":
        return SparseGroupSpecificParameterGraph(
            contribution=memo[self.contribution],
            contribution_dims=self.contribution_dims,
            term_labels=self.term_labels,
            matrix=memo[self.matrix],
        )


GroupSpecificParameterGraph = DenseGroupSpecificParameterGraph | SparseGroupSpecificParameterGraph


@dataclass
class GroupSpecificGraphState:
    parameters: dict[str, GroupSpecificParameterGraph] = field(default_factory=dict)

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "GroupSpecificGraphState":
        return GroupSpecificGraphState(
            parameters={
                label: parameter.clone(memo) for label, parameter in self.parameters.items()
            }
        )


def make_conditional_parameter_info(parameter: ConditionalParameter) -> ConditionalParameterInfo:
    common_terms = tuple(
        CommonTermInfo(term=term, coords=coords_from_common(term))
        for term in parameter.common_terms.values()
    )
    offset_terms = tuple(
        CommonTermInfo(term=term, coords=coords_from_common(term))
        for term in parameter.offset_terms.values()
    )
    hsgp_terms = tuple(
        HSGPTermInfo(term=term, coords=coords_from_hsgp(term))
        for term in parameter.hsgp_terms.values()
    )

    terms_by_factor = {}
    for term in parameter.group_specific_terms.values():
        terms_by_factor.setdefault(term.factor, []).append(term)

    group_specific_factors = tuple(
        GroupSpecificFactorInfo(group_specific_terms=tuple(terms))
        for terms in terms_by_factor.values()
    )
    return ConditionalParameterInfo(
        parameter=parameter,
        common_terms=common_terms,
        offset_terms=offset_terms,
        hsgp_terms=hsgp_terms,
        group_specific_factors=group_specific_factors,
    )
