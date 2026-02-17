# pylint: disable=unused-argument
from typing import Callable

from xarray.core.dataarray import DataArray

# A comparison function performs an operation (op) between a reference and
# a contrast DataArray and returns a result DataArray
ComparisonFunc = Callable[[DataArray, DataArray], DataArray]


def diff(reference: DataArray, contrast: DataArray) -> DataArray:
    """Compute difference between contrast and reference.

    Parameters
    ----------
    reference : DataArray
        The reference/baseline values.
    contrast : DataArray
        The contrast/comparison values.

    Returns
    -------
    DataArray
        The difference: contrast - reference
    """
    return contrast - reference


def ratio(reference: DataArray, contrast: DataArray) -> DataArray:
    """Compute ratio of contrast to reference.

    Parameters
    ----------
    reference : DataArray
        The reference/baseline values.
    contrast : DataArray
        The contrast/comparison values.

    Returns
    -------
    DataArray
        The ratio: contrast / reference
    """
    return contrast / reference


def lift(reference: DataArray, contrast: DataArray) -> DataArray:
    """Compute lift (relative difference) between contrast and reference.

    Parameters
    ----------
    reference : DataArray
        The reference/baseline values.
    contrast : DataArray
        The contrast/comparison values.

    Returns
    -------
    DataArray
        The lift: (contrast - reference) / reference
    """
    return (contrast - reference) / reference


COMPARISON_TYPES: dict[str, ComparisonFunc] = {
    "diff": diff,
    "ratio": ratio,
    "lift": lift,
}


def get_comparison_func(comparison: str | ComparisonFunc) -> ComparisonFunc:
    """Match a comparison specification to a callable function.

    Parameters
    ----------
    comparison : str or ComparisonFunc
        Either a string name from the registry ("diff", "ratio", "lift")
        or a custom callable with signature (reference, contrast) -> result.

    Returns
    -------
    ComparisonFunc
        The resolved comparison function.

    Raises
    ------
    ValueError
        If comparison is a string but not found in the registry.
    TypeError
        If comparison is neither a string nor a callable.
    """
    match comparison:
        case str():
            if comparison not in COMPARISON_TYPES:
                available = ", ".join(f"'{k}'" for k in COMPARISON_TYPES)
                raise ValueError(
                    f"Unknown comparison '{comparison}'. Available options: {available}"
                )
            return COMPARISON_TYPES[comparison]
        case _ if callable(comparison):
            return comparison
        case _:
            raise TypeError(
                f"'comparison' must be a callable or string, got {type(comparison).__name__}."
            )


# A slope function scales the raw derivative (dydx) given the evaluation point x
# and the response y, and returns a scaled DataArray
SlopeFunc = Callable[[DataArray, DataArray, DataArray], DataArray]


def dydx(derivative: DataArray, x: DataArray, y: DataArray) -> DataArray:
    """Unit change in x associated with a unit change in y.

    Parameters
    ----------
    derivative : DataArray
        The raw derivative dy/dx.
    x : DataArray
        The evaluation point values.
    y : DataArray
        The response values at x.

    Returns
    -------
    DataArray
        The unscaled derivative (identity).
    """
    return derivative


def eyex(derivative: DataArray, x: DataArray, y: DataArray) -> DataArray:
    """Percent change in x associated with a percent change in y.

    Parameters
    ----------
    derivative : DataArray
        The raw derivative dy/dx.
    x : DataArray
        The evaluation point values.
    y : DataArray
        The response values at x.

    Returns
    -------
    DataArray
        The elasticity: (dy/dx) * (x / y)
    """
    return derivative * (x / y)


def eydx(derivative: DataArray, x: DataArray, y: DataArray) -> DataArray:
    """Unit change in x associated with a percent change in y.

    Parameters
    ----------
    derivative : DataArray
        The raw derivative dy/dx.
    x : DataArray
        The evaluation point values.
    y : DataArray
        The response values at x.

    Returns
    -------
    DataArray
        The semi-elasticity: (dy/dx) / y
    """
    return derivative / y


def dyex(derivative: DataArray, x: DataArray, y: DataArray) -> DataArray:
    """Percent change in x associated with a unit change in y.

    Parameters
    ----------
    derivative : DataArray
        The raw derivative dy/dx.
    x : DataArray
        The evaluation point values.
    y : DataArray
        The response values at x.

    Returns
    -------
    DataArray
        The scaled derivative: (dy/dx) * x
    """
    return derivative * x


SLOPE_TYPES: dict[str, SlopeFunc] = {
    "dydx": dydx,
    "eyex": eyex,
    "eydx": eydx,
    "dyex": dyex,
}


def get_slope_func(slope: str | SlopeFunc) -> SlopeFunc:
    """Match a slope specification to a callable function.

    Parameters
    ----------
    slope : str or SlopeFunc
        Either a string name from the registry ("dydx", "eyex", "eydx", "dyex")
        or a custom callable with signature (derivative, x, y) -> result.

    Returns
    -------
    SlopeFunc
        The resolved slope function.

    Raises
    ------
    ValueError
        If slope is a string but not found in the registry.
    TypeError
        If slope is neither a string nor a callable.
    """
    match slope:
        case str():
            if slope not in SLOPE_TYPES:
                available = ", ".join(f"'{k}'" for k in SLOPE_TYPES)
                raise ValueError(f"Unknown slope '{slope}'. Available options: {available}")
            return SLOPE_TYPES[slope]
        case _ if callable(slope):
            return slope
        case _:
            raise TypeError(f"'slope' must be a callable or string, got {type(slope).__name__}.")
