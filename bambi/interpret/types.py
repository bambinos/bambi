from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike
from pandas import Series

# A Variable is a wrapper around pd.Series
Variable = Series

# A user may provide one of the following types to "condition on"
ConditionalParam = None | str | list[str] | dict[str, np.ndarray | list | int | float]

# A user may provide one of the following contrast types
ConstrastParam = str | dict[str, np.ndarray | list | int | float]

# Values provided by a user can be one of the following
Values = list[int | float] | ArrayLike | Series


@dataclass(frozen=True)
class Contrast:
    variable: Variable


@dataclass(frozen=True)
class Conditional:
    """Conditional type represents a sequence of variables (and their corresponding values)
    to condition on."""

    variables: tuple[Variable, ...]
