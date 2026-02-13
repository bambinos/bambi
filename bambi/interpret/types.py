from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

from .validate import Values, validate_category_values, validate_numeric_values

# Strategy type: given a Series, produce default values as a Series
DefaultStrategy = Callable[[Series], Series]


def _contrast_defaults(series: Series) -> Series:
    """Generate default contrast values based on dtype."""
    match series.dtype:
        case pd.CategoricalDtype():
            return pd.Series(series, name=series.name)
        case dtype if is_float_dtype(dtype):
            mean = series.mean()
            return pd.Series([mean - 0.5, mean + 0.5], name=series.name).astype(dtype)
        case dtype if is_integer_dtype(dtype):
            mode = series.mode().iloc[0]
            return pd.Series([mode - 1, mode + 1], name=series.name).astype(dtype)
        case _:
            raise TypeError(f"Unsupported dtype for contrast: {series.dtype}")


def _conditional_defaults(series: Series) -> Series:
    """Generate default conditional values based on dtype."""
    match series.dtype:
        case pd.CategoricalDtype():
            return pd.Series(series.cat.categories, name=series.name).astype(
                series.dtype
            )
        case dtype if is_float_dtype(dtype):
            xs = np.linspace(series.min(), series.max(), num=50)
            return pd.Series(xs, name=series.name).astype(dtype)
        case dtype if is_integer_dtype(dtype):
            return pd.Series(series.unique(), name=series.name).astype(dtype)
        case _:
            raise TypeError(f"Unsupported dtype for conditional: {series.dtype}")


def _default_defaults(series: Series) -> Series:
    """Generate default values (mode for categorical/integer, mean for float)."""
    match series.dtype:
        case pd.CategoricalDtype():
            return pd.Series(series.mode().iloc[0], name=series.name)
        case dtype if is_float_dtype(dtype):
            return pd.Series(series.mean(), name=series.name)
        case dtype if is_integer_dtype(dtype):
            return pd.Series(series.mode().iloc[0], name=series.name)
        case _:
            raise TypeError(f"Unsupported dtype for default: {series.dtype}")


def _resolve_values(
    name: str,
    data: DataFrame,
    values: Values | None,
    defaults: DefaultStrategy,
) -> Series:
    """Resolve a variable's values from user input or dtype-based defaults.

    Parameters
    ----------
    name : str
        Name of the variable in the DataFrame.
    data : DataFrame
        DataFrame containing the variable.
    values : Values or None
        User-provided values. If None, defaults are generated via the strategy.
    defaults : DefaultStrategy
        Strategy function to generate default values when `values` is None.

    Returns
    -------
    Series
        A pandas Series with the resolved values.

    Raises
    ------
    KeyError
        If `name` is not found in the DataFrame.
    TypeError
        If the dtype is unsupported or values have wrong type.
    """
    if name not in data.columns:
        raise KeyError(
            f"'{name}' not found in DataFrame. Available: {list(data.columns)}"
        )

    series = data[name]

    match (series.dtype, values):
        # User-provided categorical values
        case (pd.CategoricalDtype(), vals) if vals is not None:
            return validate_category_values(vals, name, reference=series)
        # User-provided numeric values
        case (_, vals) if is_numeric_dtype(series.dtype) and vals is not None:
            return validate_numeric_values(vals, name, target_dtype=series.dtype)
        # No values provided — delegate to context-specific strategy
        case (_, None):
            return defaults(series)
        case _:
            raise TypeError(f"Unsupported dtype: {series.dtype}")


@dataclass(frozen=True)
class ContrastVariable:
    """A single variable with values to compare across.

    Parameters
    ----------
    variable : Series
        A pandas Series containing the contrast values (at least 2).
    """

    variable: Series

    @staticmethod
    def from_param(
        data: DataFrame,
        contrast: str | dict[str, Values],
    ) -> ContrastVariable:
        """Create a ContrastVariable from user input.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the variable data.
        contrast : str or dict[str, Values]
            Either a variable name (uses dtype defaults) or a single-entry
            dict mapping variable name to values.

        Returns
        -------
        ContrastVariable

        Raises
        ------
        ValueError
            If dict has != 1 key or the resolved values have fewer than 2 entries.
        TypeError
            If contrast type is unsupported.
        """
        match contrast:
            case str():
                series = _resolve_values(contrast, data, None, _contrast_defaults)
            case dict() if len(contrast) == 1:
                name, values = next(iter(contrast.items()))
                series = _resolve_values(name, data, values, _contrast_defaults)
            case dict():
                raise ValueError(
                    f"Contrast dict must have exactly one key-value pair, got {len(contrast)}"
                )
            case _:
                raise TypeError(f"Unsupported contrast type: {type(contrast)}")

        if len(series) < 2:
            raise ValueError(
                f"Contrast '{series.name}' must contain at least 2 values, got {len(series)}"
            )

        return ContrastVariable(variable=series)


@dataclass(frozen=True)
class ConditionalVariables:
    """A collection of variables to condition on.

    Parameters
    ----------
    variables : tuple[Series, ...]
        A tuple of pandas Series representing the conditional variables.
    """

    variables: tuple[Series, ...]

    @staticmethod
    def from_param(
        data: DataFrame,
        conditional: str | list[str] | dict[str, Values] | None,
    ) -> ConditionalVariables:
        """Create ConditionalVariables from user input.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the variable data.
        conditional : str, list[str], dict[str, Values], or None
            Variable specification: a single name, list of names, dict mapping
            names to values, or None for empty conditioning.

        Returns
        -------
        ConditionalVariables

        Raises
        ------
        KeyError
            If a variable name is not found in the DataFrame.
        TypeError
            If conditional type is unsupported.
        """
        match conditional:
            case str():
                return ConditionalVariables(
                    (_resolve_values(conditional, data, None, _conditional_defaults),)
                )
            case list():
                return ConditionalVariables(
                    tuple(
                        _resolve_values(name, data, None, _conditional_defaults)
                        for name in conditional
                    )
                )
            case dict():
                return ConditionalVariables(
                    tuple(
                        _resolve_values(name, data, values, _conditional_defaults)
                        for name, values in conditional.items()
                    )
                )
            case None:
                return ConditionalVariables(())
            case _:
                raise TypeError(f"Unsupported conditional type: {type(conditional)}")

    @property
    def names(self) -> set[str]:
        """Return the set of variable names."""
        return {v.name for v in self.variables}


@dataclass(frozen=True)
class DefaultVariables:
    """Default values for model covariates not explicitly provided.

    Parameters
    ----------
    variables : tuple[Series, ...]
        A tuple of pandas Series with default values (mode or mean).
    """

    variables: tuple[Series, ...]

    @staticmethod
    def from_model(
        data: DataFrame,
        model_covariates: list[str],
        provided_names: set[str],
    ) -> DefaultVariables:
        """Create DefaultVariables for covariates not already provided.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the variable data.
        model_covariates : list[str]
            All covariate names from the model.
        provided_names : set[str]
            Names already provided as contrast or conditional variables.

        Returns
        -------
        DefaultVariables
        """
        default_names = set(model_covariates) - provided_names
        return DefaultVariables(
            tuple(
                _resolve_values(name, data, None, _default_defaults)
                for name in default_names
            )
        )
