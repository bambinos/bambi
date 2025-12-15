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
from seaborn.objects import Plot
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
    Contrast,
    Values,
    Variable,
)


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
        """Convert series to target dtype if specified.

        Parameters
        ----------
        series : Series
            The pandas Series to convert.

        Returns
        -------
        Series
            The series converted to target_dtype if specified, otherwise unchanged.

        Raises
        ------
        TypeError
            If conversion to target_dtype fails.
        """
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


def get_contrast(
    data: DataFrame, contrast: str | dict[str, np.ndarray | list | int | float]
) -> Contrast:
    """Parse contrast parameter and create a 'Contrast' for comparisons.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the variable data.
    contrast : str | dict[str, np.ndarray | list | int | float]
        The contrast specification, either as a string (variable name) or
        a dictionary mapping variable name to values.

    Returns
    -------
    Contrast
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
        """Create a Variable for contrast analysis.

        Parameters
        ----------
        name : str
            Name of the variable to create.
        values : Values or None, optional
            User-provided values for the variable. If None, default values are generated
            based on the variable's dtype.

        Returns
        -------
        Variable
            A pandas Series representing the contrast variable with appropriate values.

        Raises
        ------
        KeyError
            If the variable name is not found in the DataFrame.
        ValueError
            If the contrast must contain more than one value but doesn't.
        TypeError
            If the variable dtype is not supported for contrasts.
        """
        if name not in data.columns:
            raise KeyError(
                f"Contrast '{name}' not found in DataFrame. Available: {list(data.columns)}"
            )

        if len(values) < 2:
            raise ValueError(
                f"Contrast '{name}' must contain more than one value. Received: {len(values)}"
            )

        series = data[name]
        dtype = series.dtype

        match (dtype, values):
            # User-provided categorical values
            case (pd.CategoricalDtype(), vals) if vals is not None:
                return validate_category_values(vals, name, reference=series)
            # Default categorical values
            case (pd.CategoricalDtype(), None):
                return pd.Series(series, name=name)

            # User-provided float values
            case (dtype, vals) if is_float_dtype(dtype) and vals is not None:
                return validate_numeric_values(vals, name, target_dtype=series.dtype)
            # Default float values
            case (dtype, None) if is_float_dtype(dtype):
                eps = 0.5
                mean = series.mean()
                return pd.Series([mean - eps, mean + eps], name=name).astype(dtype)

            # User-provided integer values
            case (dtype, vals) if is_integer_dtype(dtype) and vals is not None:
                return validate_numeric_values(vals, name, target_dtype=series.dtype)
            # Default integer values
            case (dtype, None) if is_integer_dtype(dtype):
                eps = 1
                mode = series.mode().iloc[0]  # Use first mode
                return pd.Series([mode - eps, mode + eps], name=name).astype(dtype)

            case _:
                raise TypeError(f"Unsupported dtype for contrast: {dtype}")

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
    data: DataFrame,
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
) -> Conditional:
    """Parse conditional parameter and create Variables for conditioning.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the variable data.
    conditional : Optional[str | list[str] | dict[str, np.ndarray | list | int | float]]
        The conditional specification, either as a string (single variable name),
        a list of strings (multiple variable names), or a dictionary mapping
        variable names to values.

    Returns
    -------
    Conditional
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
        """Create a Variable for conditional analysis.

        Parameters
        ----------
        name : str
            Name of the variable to create.
        values : Values or None, optional
            User-provided values for the variable. If None, default values are generated
            based on the variable's dtype (all categories for categorical, 50 points for float,
            unique values for integer).

        Returns
        -------
        Variable
            A pandas Series representing the conditional variable with appropriate values.

        Raises
        ------
        KeyError
            If the variable name is not found in the DataFrame.
        TypeError
            If the variable dtype is not supported for conditionals.
        """
        if name not in data.columns:
            raise KeyError(
                f"Variable '{name}' not found in DataFrame. Available: {list(data.columns)}"
            )

        series = data[name]
        dtype = series.dtype

        match (dtype, values):
            # User-provided categorical values
            case (pd.CategoricalDtype(), vals) if vals is not None:
                return validate_category_values(vals, name, reference=series)
            # Default categorical values
            case (pd.CategoricalDtype(), None):
                return pd.Series(series.cat.categories, name=name).astype(series.dtype)

            # User-provided float values
            case (dtype, vals) if is_float_dtype(dtype) and vals is not None:
                return validate_numeric_values(vals, name, target_dtype=series.dtype)
            # Default float values
            case (dtype, None) if is_float_dtype(dtype):
                xs = np.linspace(series.min(), series.max(), num=50)
                return pd.Series(xs, name=name).astype(dtype)

            # User-provided integer values
            case (dtype, vals) if is_integer_dtype(dtype) and vals is not None:
                return validate_numeric_values(vals, name, target_dtype=series.dtype)
            # Default integer values
            case (dtype, None) if is_integer_dtype(dtype):
                return pd.Series(series.unique(), name=name).astype(dtype)

            case _:
                raise TypeError(f"Unsupported dtype for contrast: {dtype}")

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

    Takes multiple variables (pandas Series) and creates a DataFrame containing all
    possible combinations of their values using Cartesian product.

    Parameters
    ----------
    variables : tuple[Variable, ...]
        Tuple of pandas Series representing variables. Each Series should have a name
        that will be used as a column name in the resulting DataFrame.

    Returns
    -------
    DataFrame
        A DataFrame containing the Cartesian product of all variable values.
        Each row represents one unique combination of values across all variables.
        Column names correspond to the names of the input Series.
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
        The xarray DataArray containing posterior samples with 'chain' and 'draw' dimensions.
    prob : float
        Probability for the credible interval (between 0 and 1). For example, 0.95
        corresponds to a 95% credible interval.
    transforms : Callable
        Function to transform the data before computing statistics. Common transforms
        include identity, log, exp, etc.

    Returns
    -------
    DataFrame
        A DataFrame containing summary statistics with columns:
        - 'estimate': posterior mean
        - 'lower_X%': lower bound of credible interval
        - 'upper_Y%': upper bound of credible interval
        The '__obs__' index column is dropped from the final result.
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

    conditional = get_conditional(model.data, conditional)

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
    theme: dict[str, Any] = {},
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
) -> Plot:
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
    theme : dict or None
        A dictionary of 'matplotlib rc' parameters.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
    subplot_kwargs : Mapping[str, str] or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    Plot
        Displays and returns a Seaborn objects 'Plot'

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    # Cannot plot more than three dimensions
    _conditional = get_conditional(model.data, conditional)
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

    p = plot(summary_df, plot_config, theme)

    return p


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

    contrast = get_contrast(model.data, contrast)
    conditional = get_conditional(model.data, conditional)

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
        .drop_duplicates()  # This is fine because the join on index below will duplicate if necessary
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
    theme: dict[str, Any] = {},
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
) -> Plot:
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
    Plot
        Displays and returns a Seaborn objects 'Plot'

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    # Cannot plot more than three dimensions
    _conditional = get_conditional(model.data, conditional)
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

    p = plot(summary_df, plot_config, theme)

    return p
