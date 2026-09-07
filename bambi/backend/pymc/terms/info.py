"""Static information used to build backend term graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bambi.backend.pymc.coords import coords_from_group_specific
from bambi.backend.pymc.types import Coords, Dims

if TYPE_CHECKING:
    from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm


@dataclass(frozen=True)
class CommonTermInfo:
    term: CommonTerm
    coords: Coords

    @property
    def data_dims(self) -> Dims:
        return ("__obs__", *self.coords)


@dataclass(frozen=True)
class GroupSpecificTermInfo:
    term: GroupSpecificTerm
    expression_coords: Coords
    factor_coords: Coords


@dataclass(frozen=True)
class HSGPTermInfo:
    term: HSGPTerm
    coords: Coords


@dataclass(frozen=True)
class GroupSpecificFactorInfo:
    """Static information for all group-specific terms sharing one factor."""

    group_specific_terms: tuple[GroupSpecificTerm, ...]
    factor_name: str = field(init=False)
    factor_ndim: int = field(init=False)
    terms: tuple[GroupSpecificTermInfo, ...] = field(init=False)
    groups_n: int = field(init=False)

    def __post_init__(self) -> None:
        representative = self.group_specific_terms[0]
        terms = []
        for term in self.group_specific_terms:
            expression_coords, factor_coords = coords_from_group_specific(term)
            terms.append(
                GroupSpecificTermInfo(
                    term=term,
                    expression_coords=expression_coords,
                    factor_coords=factor_coords,
                )
            )
        term_infos = tuple(terms)
        object.__setattr__(self, "factor_name", representative.factor_name)
        object.__setattr__(self, "factor_ndim", len(term_infos[0].factor_coords))
        object.__setattr__(self, "terms", term_infos)
        object.__setattr__(self, "groups_n", len(representative.groups))
