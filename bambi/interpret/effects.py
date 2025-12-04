from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import arviz as az
import numpy as np
import pandas as pd
from arviz import InferenceData
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from utils import get_model_covariates
from xarray import DataArray

from bambi import Model
from bambi.interpret.utils import (
    aggregate,
    create_plot_config,
    get_response_and_target,
    identity,
)

from .helpers import compare, create_inference_data
from .plots import PlotConfig, plot
from .types import (
    Conditional,
    ConditionalParam,
    ConstrastParam,
    Contrast,
    Values,
    Variable,
)


def validate_values(
    values: Values, var_name: str, target_dtype: np.dtype | None = None
) -> Series:
    """Validate input values and convert to pandas Series with consistent dtype."""
    match values:
        case list() as lst:
            # Validate all elements are int or float
            if not all(isinstance(x, (int, float)) for x in lst):
                raise TypeError(
                    f"List values for '{var_name}' must contain only int or float, "
                    f"got types: {set(type(x).__name__ for x in lst)}"
                )
            if len(lst) == 0:
                raise ValueError(f"List values for '{var_name}' cannot be empty")
            # Convert list to Series with target dtype
            series = pd.Series(lst, name=var_name)
            if target_dtype is not None:
                try:
                    series = series.astype(target_dtype)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Cannot convert list values for '{var_name}' to target dtype {target_dtype}: {e}"
                    )
            return series

        case np.ndarray() as arr:
            # Validate array is numeric
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"Array values for '{var_name}' must be numeric, got dtype: {arr.dtype}"
                )
            if arr.size == 0:
                raise ValueError(f"Array values for '{var_name}' cannot be empty")
            # Convert array to Series with target dtype
            series = pd.Series(arr, name=var_name)
            if target_dtype is not None:
                try:
                    series = series.astype(target_dtype)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Cannot convert array values for '{var_name}' to target dtype {target_dtype}: {e}"
                    )
            return series

        case Series() as series:
            # Validate series is numeric
            if not is_numeric_dtype(series.dtype):
                raise TypeError(
                    f"Series values for '{var_name}' must be numeric, got dtype: {series.dtype}"
                )
            if len(series) == 0:
                raise ValueError(f"Series values for '{var_name}' cannot be empty")
            # Return copy with proper name and target dtype
            result = series.copy().rename(var_name)
            if target_dtype is not None and result.dtype != target_dtype:
                try:
                    result = result.astype(target_dtype)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Cannot convert Series values for '{var_name}' to target dtype {target_dtype}: {e}"
                    )
            return result

        case None:
            raise TypeError(
                f"Values for '{var_name}' cannot be None. "
                "This indicates a programming error in parse_conditional."
            )

        case _:
            raise TypeError(
                f"Values for '{var_name}' must be one of: list[int|float], np.ndarray, or pd.Series. "
                f"Got: {type(values).__name__}"
            )


def get_defaults(var: str, data: DataFrame):
    """Computes a default value for the given `var` in the `data`."""
    series = data[var]

    match series.dtype:
        case pd.CategoricalDtype():
            # Takes the first mode if there are multiple modes
            return pd.Series(series.mode().iloc[0], name=var)  # .astype(series.dtype)
        case dtype if is_float_dtype(dtype):
            return pd.Series(series.mean(), name=var)  # .astype(series.dtype)
        case dtype if is_integer_dtype(dtype):
            # Takes the first mode if there are multiple modes
            return pd.Series(series.mode().iloc[0], name=var)
        case _:
            raise TypeError(f"Unsupported data type: {series.dtype}")


def parse_constrast(constrast: ConstrastParam, data: DataFrame):
    def create_variable(name: str, values: Values | None = None) -> Variable:
        if name not in data.columns:
            raise KeyError(
                f"Variable '{name}' not found in DataFrame. Available: {list(data.columns)}"
            )

        series = data[name]

        # TODO: Validation for if user passes a list or array and it only contains one element.

        match series.dtype:
            case pd.CategoricalDtype():
                if values is not None:
                    if isinstance(values, (list, np.ndarray)):
                        valid_cats = set(series.cat.categories)
                        provided_cats = (
                            set(values)
                            if isinstance(values, list)
                            else set(values.tolist())
                        )
                        if not provided_cats.issubset(valid_cats):
                            invalid = provided_cats - valid_cats
                            raise ValueError(
                                f"Invalid category for '{name}: {invalid}' "
                                f"Valid categories: {list(valid_cats)}"
                            )

                        return pd.Series(
                            pd.Categorical(
                                values,
                                categories=series.cat.categories,
                                ordered=series.cat.ordered,
                            ),
                            name=name,
                        )
                    else:
                        raise ValueError(
                            f"Categorical variable '{name}' values must be a list or array of categories."
                        )
                else:
                    return pd.Series(series, name=name)
            case dtype if is_float_dtype(dtype):
                if values is not None:
                    # Validate and return the values as a Pandas Series
                    return validate_values(values, name, target_dtype=series.dtype)
                else:
                    mean = series.mean()
                    return pd.Series([mean - 0.5, mean + 0.5], name=name).astype(dtype)

            case dtype if is_integer_dtype(dtype):
                if values is not None:
                    # Validate and return the values as a Pandas Series
                    return validate_values(values, name, target_dtype=series.dtype)
                else:
                    # If more than one mode, use the first one
                    mode = series.mode().iloc[0]
                    return pd.Series([mode - 1, mode + 1], name=name).astype(dtype)

    match constrast:
        case str():
            variable = create_variable(constrast)
        case dict():
            # For dict, assume single key-value pair for contrast
            if len(constrast) != 1:
                raise ValueError(
                    f"Contrast dict must have exactly one key-value pair, got {len(constrast)}"
                )
            name, values = next(iter(constrast.items()))
            variable = create_variable(name, values)
        case _:
            raise TypeError(f"Unsupported contrast type: {type(constrast)}")

    return variable


def parse_conditional(
    conditional: ConditionalParam,
    data: DataFrame,
):
    def create_variable(name: str, values: Values | None = None) -> Variable:
        if name not in data.columns:
            raise KeyError(
                f"Variable '{name}' not found in DataFrame. Available: {list(data.columns)}"
            )

        series = data[name]

        match series.dtype:
            case pd.CategoricalDtype():
                if values is not None:
                    if isinstance(values, (list, np.ndarray)):
                        valid_cats = set(series.cat.categories)
                        provided_cats = (
                            set(values)
                            if isinstance(values, list)
                            else set(values.tolist())
                        )
                        if not provided_cats.issubset(valid_cats):
                            invalid = provided_cats - valid_cats
                            raise ValueError(
                                f"Invalid category for '{name}: {invalid}' "
                                f"Valid categories: {list(valid_cats)}"
                            )

                        return pd.Series(
                            pd.Categorical(
                                values,
                                categories=series.cat.categories,
                                ordered=series.cat.ordered,
                            ),
                            name=name,
                        )
                    else:
                        raise ValueError(
                            f"Categorical variable '{name}' values must be a list or array of categories."
                        )
                else:
                    return pd.Series(series.cat.categories, name=name).astype(
                        series.dtype
                    )
            case dtype if is_float_dtype(dtype):
                if values is not None:
                    # Validate and return the values as a pd.Series
                    return validate_values(values, name, target_dtype=series.dtype)
                else:
                    xs = np.linspace(series.min(), series.max(), num=50)
                    return pd.Series(xs, name=name).astype(dtype)
            case dtype if is_integer_dtype(dtype):
                if values is not None:
                    # Validate and return the values as a pd.Series
                    return validate_values(values, name, target_dtype=series.dtype)
                else:
                    return pd.Series(series.unique(), name=name).astype(dtype)
                    # TODO: Casting back to Int will causes a bug
                    # return pd.Series(
                    #     series.quantile(q=0.5), name=name
                    # ).astype(dtype)

    match conditional:
        case str():
            variables = (create_variable(conditional),)
        case list():
            variables = tuple(create_variable(name) for name in conditional)
        case dict():
            variables = tuple(
                create_variable(name, values) for name, values in conditional.items()
            )
        case _:
            raise TypeError(f"Unsupported conditional type: {type(conditional)}")

    return variables


def plot_predictions(
    model: Model,
    idata: InferenceData,
    conditional: ConditionalParam,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
):
    """
    Parameters
    ----------
    subplot_kwargs :
        Overrides default plotting sequence.
    """
    # Cannot plot more than three-dimensions
    provided_vars = parse_conditional(conditional, model.data)
    if len(provided_vars) > 3 and average_by is None:
        raise ValueError(
            f"Cannot plot more than 3 conditional variables. Received: {len(provided_vars)}. "
            f"Consider removing a variable(s) or passing a value(s) to `average_by`."
        )

    provided_var_names = [var.name for var in provided_vars]

    plot_config = create_plot_config(provided_var_names, subplot_kwargs)

    out = predictions(
        model=model,
        idata=idata,
        conditional=conditional,
        average_by=average_by,
        target=target,
        pps=pps,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
        sample_new_groups=sample_new_groups,
    )

    plot(out, plot_config)


def predictions(
    model: Model,
    idata: InferenceData,
    conditional: ConditionalParam,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> DataFrame:
    """Compute Conditional Adjusted Predictions."""
    if not 0 < prob < 1:
        raise ValueError(
            f"'prob' must be greater than 0 and smaller than 1. It is {prob}."
        )

    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    provided_vars = parse_conditional(conditional, model.data)

    # Get all terms (covariates) defined in the model formula
    model_var_names = get_model_covariates(model).tolist()
    # Get the names of the variables passed by user
    provided_var_names = [var.name for var in provided_vars]
    # Determine variables that are defaults
    default_var_names = tuple(set(model_var_names) - set(provided_var_names))
    default_vars = tuple(get_defaults(var, model.data) for var in default_var_names)
    # Combine provided and default vars into a Conditional type
    all_vars = provided_vars + default_vars
    conditional = Conditional(variables=all_vars)

    # Create data grid (cross-product of conditional variables data)
    vals = [var.array for var in conditional.variables]
    names = [var.name for var in conditional.variables]
    # Cross-product to build data grid.
    product = pd.MultiIndex.from_product(
        vals, names=names
    )  # Naturally preserves dtypes
    preds_data = product.to_frame(index=False)

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "sample_new_groups": sample_new_groups,
        "inplace": False,
    }
    idata = model.predict(**pred_kwargs, **({"kind": "response"} if pps else {}))
    group = "posterior_predictive" if pps else "posterior"
    var = response_name if pps else target
    y_hat = idata[group][var]

    stats_data = get_summary_stats(y_hat, prob, response_transform)
    summary_df = aggregate(data=preds_data.join(stats_data, on=None), by=average_by)

    return summary_df, preds_data, idata


def comparisons(
    model: Model,
    idata: InferenceData,
    contrast: ConstrastParam,
    conditional: ConditionalParam,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    comparison: Callable | str = "diff",
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
):
    """Compute conditional adjusted comparisons."""
    if not 0 < prob < 1:
        raise ValueError(
            f"'prob' must be greater than 0 and smaller than 1. It is {prob}."
        )

    match comparison:
        case "diff":
            comparison_fn = lambda a, b: b - a
        case "ratio":
            comparison_fn = lambda a, b: b / a
        case "lift":
            comparison_fn = lambda a, b: (b - a) / a
        case _ if callable(comparison):
            comparison_fn = comparison
        case _:
            raise TypeError(
                f"'comparison' must be a callable or string, got {type(comparison).__name__}."
            )

    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    # Parse provided constrast and conditional variables
    provided_contrast_vars = parse_constrast(contrast, model.data)
    provided_conditional_vars = parse_conditional(conditional, model.data)
    provided_contrast_var_names = [provided_contrast_vars.name]
    provided_conditional_var_names = [var.name for var in provided_conditional_vars]

    # Get all terms (covariates) defined in the model formula
    model_var_names = get_model_covariates(model).tolist()
    # Determine variables that are defaults
    default_var_names = tuple(
        set(model_var_names)
        - set(provided_contrast_var_names)
        - set(provided_conditional_var_names)
    )
    default_vars = tuple(get_defaults(var, model.data) for var in default_var_names)
    # Combine provided and default vars into a Conditional type
    conditional_vars = provided_conditional_vars + default_vars
    conditional: Conditional = Conditional(variables=conditional_vars)
    contrast: Contrast = Contrast(variable=provided_contrast_vars)

    # Create data grid (cross-product of contrast and conditional variables data)
    vals = [contrast.variable.array] + [var.array for var in conditional.variables]
    names = [contrast.variable.name] + [var.name for var in conditional.variables]

    # Cross-product to build data grid.
    product = pd.MultiIndex.from_product(
        vals, names=names
    )  # Naturally preserves dtypes
    preds_data = product.to_frame(index=False)

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "sample_new_groups": sample_new_groups,
        "inplace": False,
    }
    preds_idata = model.predict(**pred_kwargs, **({"kind": "response"} if pps else {}))
    group = "posterior_predictive" if pps else "posterior"
    var = response_name if pps else target

    compare_idata = create_inference_data(preds_idata, preds_data)
    compared_draws = compare(
        compare_idata,
        contrast,
        var,
        group,
        comparison_fn=comparison_fn,
    )

    # Compute mean and uncertainty over (chain, draw)
    summary_draws = {
        k: get_summary_stats(v, prob, response_transform)
        for k, v in compared_draws.items()
    }
    # Comparison column name corresponds to the contrast values being compared (e.g., 1_vs_4)
    comparison_df = pd.concat(summary_draws, names=["comparison", "index"]).reset_index(
        level=0
    )
    # Use index of both dataframes to join on.
    # This is useful when there are multiple contrast variables and or values
    summary_df = (
        preds_data[[var.name for var in conditional.variables]]
        .drop_duplicates()  # Okay because the join on index below will duplicate if necessary
        .reset_index(drop=True)
        .join(comparison_df, on=None)
    )

    summary_df = aggregate(data=summary_df, by=average_by)

    return summary_df


def plot_comparisons(
    model: Model,
    idata: InferenceData,
    contrast: ConstrastParam,
    conditional: ConditionalParam,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    comparison: Callable | str = "diff",
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
):
    """
    Parameters
    ----------
    subplot_kwargs :
        Overrides default plotting sequence.
    """
    # Cannot plot more than three-dimensions
    provided_vars = parse_conditional(conditional, model.data)
    if len(provided_vars) > 3 and average_by is None:
        raise ValueError(
            f"Cannot plot more than 3 conditional variables. Received: {len(provided_vars)}. "
            f"Consider removing a variable(s) or passing a value(s) to `average_by`."
        )

    provided_var_names = [var.name for var in provided_vars]

    plot_config = create_plot_config(provided_var_names, subplot_kwargs)

    out = comparisons(
        model=model,
        idata=idata,
        contrast=contrast,
        conditional=conditional,
        average_by=average_by,
        target=target,
        pps=pps,
        comparison=comparison,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
        sample_new_groups=sample_new_groups,
    )

    plot(out, plot_config)


def get_summary_stats(x: DataArray, prob: float, transforms) -> DataFrame:
    """Computes summary statistics (mean and uncertainty interval) of an array.

    Parameters
    ----------

    Returns
    -------
    """
    x = transforms(x)
    mean = x.mean(dim=("chain", "draw")).to_series().rename("estimate").to_frame()

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound
    bounds = (
        x.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))
        .to_series()
        .unstack(level="quantile")
        .rename(
            columns={
                lower_bound: f"lower_{lower_bound}%",
                upper_bound: f"upper_{upper_bound}%",
            }
        )
    )

    stats = mean.join(bounds).reset_index().drop("__obs__", axis=1)

    return stats
