from typing import Any, Callable, Mapping, Optional

import arviz as az
import numpy as np
import pandas as pd
from arviz import InferenceData
from pandas import DataFrame, Series
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from xarray import DataArray

from bambi import Model
from bambi.interpret.utils import (
    aggregate,
    create_plot_config,
    get_model_covariates,
    get_response_and_target,
    identity,
)

from .helpers import compare, create_inference_data
from .plots import plot
from .types import (
    Conditional,
    ConditionalParam,
    Contrast,
    ContrastParam,
    Values,
    Variable,
)


def validate_values(
    values: Values, var_name: str, target_dtype: np.dtype | None = None
) -> Series:
    """Validate and convert user provided values to a Pandas Series.

    Parameters
    ----------
    values : Values
        User-provided values as a list, numpy array, or pandas Series.
    var_name : str
        Name of the variable for error messages and Series naming.
    target_dtype : np.dtype or None
        Target data type to convert values to. If None, uses the natural dtype.

    Returns
    -------
    Series
        A pandas Series containing the validated and converted values.

    Raises
    ------
    TypeError
        If values are not a list, ndarray, or Series, or if they cannot be converted
        to the target dtype.
    ValueError
        If the values container is empty.
    """
    match values:
        # Values can be a list
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
        # Values can be an ndarray
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
        # Values can be a Pandas Series
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
        case _:
            raise TypeError(
                f"Values for '{var_name}' must be one of: list[int|float], np.ndarray, or pd.Series. "
                f"Got: {type(values).__name__}"
            )


def get_defaults(var: str, data: DataFrame) -> Series:
    """Compute a default value for the given variable in the data.

    Parameters
    ----------
    var : str
        Name of the variable to compute defaults for.
    data : DataFrame
        DataFrame containing the variable.

    Returns
    -------
    Series
        A pandas Series containing the default value(s) for the variable.
        For categorical and integer types, returns the mode.
        For float types, returns the mean.

    Raises
    ------
    TypeError
        If the variable has an unsupported data type.
    """
    series = data[var]

    match series.dtype:
        case pd.CategoricalDtype():
            # Takes the first mode if there are multiple modes
            return pd.Series(series.mode().iloc[0], name=var)
        case dtype if is_float_dtype(dtype):
            return pd.Series(series.mean(), name=var)
        case dtype if is_integer_dtype(dtype):
            # Takes the first mode if there are multiple modes
            return pd.Series(series.mode().iloc[0], name=var)
        case _:
            raise TypeError(f"Unsupported data type: {series.dtype}")


def get_contrast(contrast: ContrastParam, data: DataFrame) -> Contrast:
    """Parse contrast parameter and create a Variable for comparisons.

    Parameters
    ----------
    contrast : ContrastParam
        The contrast specification, either as a string (variable name) or
        a dictionary mapping variable name to values.
    data : DataFrame
        DataFrame containing the variable data.

    Returns
    -------
    Variable
        A pandas Series representing the contrast variable with appropriate values.
        For categorical variables, returns all categories or user-specified categories.
        For float variables, returns mean ± 0.5 or user-specified values.
        For integer variables, returns mode ± 1 or user-specified values.

    Raises
    ------
    KeyError
        If the variable name is not found in the DataFrame.
    ValueError
        If provided categorical values are not valid categories, or if the
        contrast dict does not have exactly one key-value pair.
    TypeError
        If the contrast type is not supported.
    """

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

    match contrast:
        case str():
            variable = Contrast(variable=create_variable(contrast))
        case dict():
            # For dict, assume single key-value pair for contrast
            if len(contrast) != 1:
                raise ValueError(
                    f"Contrast dict must have exactly one key-value pair, got {len(contrast)}"
                )
            name, values = next(iter(contrast.items()))
            variable = Contrast(variable=create_variable(name, values))
        case _:
            raise TypeError(f"Unsupported contrast type: {type(contrast)}")

    return variable


def get_conditional(
    conditional: ConditionalParam,
    data: DataFrame,
) -> Conditional:
    """Parse conditional parameter and create Variables for conditioning.

    Parameters
    ----------
    conditional : ConditionalParam
        The conditional specification, either as a string (single variable name),
        a list of strings (multiple variable names), or a dictionary mapping
        variable names to values.
    data : DataFrame
        DataFrame containing the variable data.

    Returns
    -------
    tuple[Variable, ...]
        A tuple of pandas Series representing the conditional variables with appropriate values.
        For categorical variables, returns all categories or user-specified categories.
        For float variables, returns 50 equally-spaced points from min to max or user-specified values.
        For integer variables, returns all unique values or user-specified values.

    Raises
    ------
    KeyError
        If a variable name is not found in the DataFrame.
    ValueError
        If provided categorical values are not valid categories.
    TypeError
        If the conditional type is not supported.
    """

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
            variables = Conditional((create_variable(conditional),))
        case list():
            variables = Conditional(
                tuple(create_variable(name) for name in conditional)
            )
        case dict():
            variables = Conditional(
                tuple(
                    create_variable(name, values)
                    for name, values in conditional.items()
                )
            )
        case None:
            variables = Conditional(tuple())
        case _:
            raise TypeError(f"Unsupported conditional type: {type(conditional)}")

    return variables


def create_grid(variables: tuple[Variable, ...]) -> DataFrame:
    """Create cross-product grid from variables.

    Parameters
    ----------
    variables : tuple[Variable, ...]
        Tuple of pandas Series representing variables.

    Returns
    -------
    DataFrame
        A DataFrame containing the Cartesian product of all variable values.
    """
    vals = [var.array for var in variables]
    names = [var.name for var in variables]
    product = pd.MultiIndex.from_product(vals, names=names)

    return product.to_frame(index=False)


def get_summary_stats(x: DataArray, prob: float, transforms: Callable) -> DataFrame:
    """Compute summary statistics (mean and uncertainty interval) of an array.

    Parameters
    ----------
    x : DataArray
        The xarray DataArray containing posterior samples.
    prob : float
        Probability for the credible interval (between 0 and 1).
    transforms : Callable
        Function to transform the data before computing statistics.

    Returns
    -------
    DataFrame
        A DataFrame containing the estimate (mean) and lower/upper bounds of the credible interval.
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


def predictions(
    model: Model,
    idata: InferenceData,
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> DataFrame:
    """Compute conditional adjusted predictions.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    conditional : ConditionalParam
        Variables to condition on for predictions.
    average_by : str or list or bool or None
        Variables to average predictions over.
    target : str
        The target parameter to predict. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to predictions.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.

    Returns
    -------
    DataFrame
        A DataFrame containing the conditional adjusted predictions with summary statistics.

    Raises
    ------
    ValueError
        If prob is not between 0 and 1.
    """
    if not 0 < prob < 1:
        raise ValueError(
            f"'prob' must be greater than 0 and smaller than 1. It is {prob}."
        )

    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    conditional = get_conditional(conditional, model.data)

    # Parse defaults
    # Get all terms (covariates) defined in the model formula
    model_var_names = get_model_covariates(model).tolist()

    # Determine variables that are defaults
    default_var_names = tuple(
        set(model_var_names) - set(var.name for var in conditional.variables)
    )
    default_vars = tuple(get_defaults(var, model.data) for var in default_var_names)
    all_vars = conditional.variables + default_vars

    if not conditional.variables:
        # Unit-level data
        preds_data = model.data.copy()
    else:
        # Data grid (cross-product)
        conditional = Conditional(variables=all_vars)
        preds_data = create_grid(conditional.variables)

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

    return summary_df


def plot_predictions(
    model: Model,
    idata: InferenceData,
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
) -> None:
    """Plot conditional adjusted predictions.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    conditional : ConditionalParam
        Variables to condition on for predictions.
    average_by : str or list or bool or None
        Variables to average predictions over.
    target : str
        The target parameter to predict. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to predictions.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
    subplot_kwargs : Mapping[str, str] or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    None
        Displays the plot.

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    # Cannot plot more than three dimensions
    _conditional = get_conditional(conditional, model.data)
    if len(_conditional.variables) > 3 and average_by is None:
        raise ValueError(
            f"Cannot plot more than 3 conditional variables. Received: {len(_conditional.variables)}. "
            f"Consider removing a variable(s) or passing a value(s) to `average_by`."
        )

    provided_var_names = [var.name for var in _conditional.variables]
    plot_config = create_plot_config(provided_var_names, subplot_kwargs)

    summary_df = predictions(
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

    plot(summary_df, plot_config)


def comparisons(
    model: Model,
    idata: InferenceData,
    contrast: str | dict[str, np.ndarray | list | int | float],
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    comparison: Callable | str = "diff",
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> DataFrame:
    """Compute conditional adjusted comparisons.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    contrast : ContrastParam
        Variable(s) to create contrasts for.
    conditional : ConditionalParam
        Variables to condition on for comparisons.
    average_by : str or list or bool or None
        Variables to average comparisons over.
    target : str
        The target parameter to compare. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    comparison : Callable or str
        Comparison function or string ("diff", "ratio", "lift"). Default is "diff".
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to comparisons.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.

    Returns
    -------
    DataFrame
        A DataFrame containing the conditional adjusted comparisons with summary statistics.

    Raises
    ------
    ValueError
        If prob is not between 0 and 1.
    TypeError
        If comparison is not a callable or valid string.
    """
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

    contrast = get_contrast(contrast, model.data)
    conditional = get_conditional(conditional, model.data)

    # Parse defaults
    # Get all terms (covariates) defined in the model formula
    model_var_names = get_model_covariates(model).tolist()

    # Determine variables that are defaults
    default_var_names = tuple(
        set(model_var_names)
        - set(var.name for var in conditional.variables)
        - set([contrast.variable.name])
    )
    default_vars = tuple(get_defaults(var, model.data) for var in default_var_names)
    conditional_vars = conditional.variables + default_vars

    if not conditional.variables:
        # Unit-level data
        preds_data = model.data.copy()
    else:
        # Data grid (cross-product)
        conditional = Conditional(variables=conditional_vars)
        preds_data = create_grid((contrast.variable, *conditional.variables))

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
    contrast: str | dict[str, np.ndarray | list | int | float],
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
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
) -> None:
    """Plot conditional adjusted comparisons.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    contrast : contrastParam
        Variable(s) to create contrasts for.
    conditional : ConditionalParam
        Variables to condition on for comparisons.
    average_by : str or list or bool or None
        Variables to average comparisons over.
    target : str
        The target parameter to compare. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    comparison : Callable or str
        Comparison function or string ("diff", "ratio", "lift"). Default is "diff".
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to comparisons.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
    subplot_kwargs : Mapping[str, str] or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    None
        Displays the plot.

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    # Cannot plot more than three dimensions
    _conditional = get_conditional(conditional, model.data)
    if len(_conditional.variables) > 3 and average_by is None:
        raise ValueError(
            f"Cannot plot more than 3 conditional variables. Received: {len(_conditional.variables)}. "
            f"Consider removing a variable(s) or passing a value(s) to `average_by`."
        )

    provided_var_names = [var.name for var in _conditional.variables]
    plot_config = create_plot_config(provided_var_names, subplot_kwargs)

    summary_df = comparisons(
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

    plot(summary_df, plot_config)
