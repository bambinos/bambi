import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas import Series
from pandas.api.types import is_numeric_dtype

# Type alias for user-provided values
Values = list[int | float | str] | ArrayLike | Series


def validate_category_values(
    values: Values, var_name: str, reference: Series
) -> Series:
    """Validates user-provided values against the original Pandas Categorical values used to
    fit a Bambi model.

    Parameters
    ----------
    values : Values
        User-provided values to validate. Can be a list, numpy array, or pandas Series.
    var_name : str
        Name of the variable being validated.
    reference : Series
        Reference pandas Series with categorical dtype from the original model data.

    Raises
    ------
    TypeError
        If reference series is not categorical or if values is not a list, array, or Series.
    ValueError
        If values is empty or contains invalid categories not present in reference.

    Returns
    -------
    Series
        A pandas Series with categorical dtype matching the reference categories and ordering.
    """
    if not isinstance(reference.dtype, pd.CategoricalDtype):
        raise TypeError(f"Reference series for '{var_name}' must be categorical.")

    # Extract categories
    match values:
        case list() as lst:
            if len(lst) == 0:
                raise ValueError(f"List values for '{var_name}' cannot be empty")
            vals = lst
        case np.ndarray() as arr:
            if arr.size == 0:
                raise ValueError(f"Array values for '{var_name}' cannot be empty")
            vals = arr.tolist()
        case Series() as series:
            if len(series) == 0:
                raise ValueError(f"Series values for '{var_name}' cannot be empty")
            vals = series.tolist()
        case _:
            raise TypeError(
                f"Categorical values for '{var_name}' must be one of: list, np.ndarray, or pd.Series. "
                f"Got: {type(values).__name__}"
            )

    # Validate categories
    valid_cats = set(reference.cat.categories)
    provided_cats = set(vals)

    if not provided_cats.issubset(valid_cats):
        invalid = provided_cats - valid_cats
        raise ValueError(
            f"Invalid categories for '{var_name}': {invalid}. "
            f"Valid categories: {list(valid_cats)}"
        )

    # Create categorical series with reference categories and ordering
    return pd.Series(
        pd.Categorical(
            vals,
            categories=reference.cat.categories,
            ordered=reference.cat.ordered,
        ),
        name=var_name,
    )


def validate_numeric_values(
    values: Values,
    var_name: str,
    target_dtype: np.dtype | None = None,
) -> Series:
    """Validates user-provided values against the original Pandas numerical values used to
    fit a Bambi model.

    Parameters
    ----------
    values : Values
        User-provided values to validate. Can be a list, numpy array, or pandas Series.
    var_name : str
        Name of the variable being validated.
    target_dtype : np.dtype or None, optional
        Target dtype to convert the values to. If None, no conversion is performed.

    Raises
    ------
    TypeError
        If values is not a list, array, or Series, or if conversion to target_dtype fails.
    ValueError
        If values is empty.

    Returns
    -------
    Series
        A pandas Series with validated numeric values, optionally converted to target_dtype.
    """

    def convert_to_dtype(series: Series) -> Series:
        if target_dtype is not None and series.dtype != target_dtype:
            try:
                return series.astype(target_dtype)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Cannot convert values for '{var_name}' to target dtype {target_dtype}: {e}"
                )
        return series

    match values:
        case list() as lst:
            if len(lst) == 0:
                raise ValueError(f"List values for '{var_name}' cannot be empty")
            if not all(isinstance(x, (int, float)) for x in lst):
                raise TypeError(
                    f"List values for '{var_name}' must contain only int or float, "
                    f"got types: {set(type(x).__name__ for x in lst)}"
                )
            series = pd.Series(lst, name=var_name)
            return convert_to_dtype(series)

        case np.ndarray() as arr:
            if arr.size == 0:
                raise ValueError(f"Array values for '{var_name}' cannot be empty")
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"Array values for '{var_name}' must be numeric, got dtype: {arr.dtype}"
                )
            series = pd.Series(arr, name=var_name)
            return convert_to_dtype(series)

        case Series() as series:
            if len(series) == 0:
                raise ValueError(f"Series values for '{var_name}' cannot be empty")
            if not is_numeric_dtype(series.dtype):
                raise TypeError(
                    f"Series values for '{var_name}' must be numeric, got dtype: {series.dtype}"
                )
            result = series.copy().rename(var_name)
            return convert_to_dtype(result)

        case _:
            raise TypeError(
                f"Values for '{var_name}' must be one of: list[int|float], np.ndarray, or pd.Series. "
                f"Got: {type(values).__name__}"
            )
