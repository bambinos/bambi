from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike
from pandas import Series

# Type alias for a variable, which is a wrapper around pd.Series
Variable = Series
"""Type alias for a variable represented as a pandas Series."""

# Type alias for conditional parameters that users may provide
ConditionalParam = None | str | list[str] | dict[str, np.ndarray | list | int | float]
"""Type alias for conditional parameter specifications.

Can be:
- None: No conditioning
- str: Single variable name
- list[str]: Multiple variable names
- dict: Mapping of variable names to their values
"""

# Type alias for contrast parameters that users may provide
ContrastParam = str | dict[str, np.ndarray | list | int | float]
"""Type alias for contrast parameter specifications.

Can be:
- str: Variable name
- dict: Mapping of variable name to contrast values
"""

# Type alias for values provided by users
Values = list[int | float] | ArrayLike | Series
"""Type alias for user-provided values.

Can be:
- list[int | float]: List of numeric values
- ArrayLike: NumPy array or array-like object
- Series: Pandas Series
"""


@dataclass(frozen=True)
class Contrast:
    """Contrast type represents a variable used for creating comparisons.

    Parameters
    ----------
    variable : Variable
        A pandas Series containing the values to create contrasts from.
    """

    variable: Variable


@dataclass(frozen=True)
class Conditional:
    """Conditional type represents a sequence of variables to condition on.

    Parameters
    ----------
    variables : tuple[Variable, ...]
        A tuple of pandas Series representing the variables (and their corresponding values)
        to condition predictions or comparisons on.
    """

    variables: tuple[Variable, ...]
