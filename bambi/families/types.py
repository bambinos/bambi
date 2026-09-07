from enum import StrEnum
from typing import NamedTuple


class ResponseType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"


class DimType(StrEnum):
    RESPONSE = "response"
    RESPONSE_REDUCED = "response_reduced"
    RESPONSE_CUTPOINTS = "response_cutpoints"


class ParamSpec(NamedTuple):
    links: list[str]
    ndim: int = 0
    coefs_dim: DimType | None = None
