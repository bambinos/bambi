from dataclasses import dataclass

from numpy.typing import ArrayLike
from pandas import Series

# Type alias for a variable, which is a wrapper around pd.Series
Variable = Series
"""Type alias for a variable represented as a pandas Series."""

# Type alias for values provided by users
Values = list[int | float | str] | ArrayLike | Series
"""Type alias for user-provided values.

Can be:
- list[int | float | str]: List of integer, float, or string values
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
