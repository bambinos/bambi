from functools import partial
from itertools import combinations
from typing import Any, Callable, Mapping, Optional

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from arviz import InferenceData
from pandas import DataFrame
from seaborn.objects import Plot
from xarray import DataArray

from bambi.interpret.ops import (
    ComparisonFunc,
    SlopeFunc,
    get_comparison_func,
    get_slope_func,
)
from bambi.interpret.plots import PlottingConfig, plot
from bambi.interpret.types import (
    ComparisonVariable,
    ConditionalVariables,
    DefaultVariables,
    Result,
    SlopeVariable,
)
from bambi.interpret.utils import (
    aggregate,
    create_inference_data,
    get_model_covariates,
    get_response_and_target,
    identity,
)
from bambi.models import Model


def _determine_plot_vars(
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]],
    average_by: str | list[str] | None,
    model_data: DataFrame,
) -> list[str]:
    """Determine which variables to plot based on conditional and average_by parameters.

    Parameters
    ----------
    conditional : ConditionalParam
        User-specified conditional variables.
    average_by : str, list, or None
        Variables to average over.
    model_data : DataFrame
        Model data used to parse conditional variables.

    Returns
    -------
    list[str]
        Variable names to use for plotting configuration.
    """
    cond = ConditionalVariables.from_param(model_data, conditional)
    provided_var_names = [var.name for var in cond.variables]

    match average_by:
        case None:
            return provided_var_names
        case "all":
            return []
        case str():
            return [average_by]
        case list():
            return list(average_by)


def _extract_dim_columns(summary_df: DataFrame, var_names: list[str]) -> list[str]:
    """Extract dimension columns from summary dataframe.

    These are additional columns (like class indices for Categorical models)
    that should be plotted but aren't part of the user-specified variables.

    Parameters
    ----------
    summary_df : DataFrame
        The summary dataframe from a Result object.
    var_names : list[str]
        Already-determined variable names from user input.

    Returns
    -------
    list[str]
        Dimension column names found in the summary.
    """
    # Exclude metadata and statistic columns
    metadata_cols = ["term", "estimate_type", "value"]
    stat_keywords = ["estimate", "lower", "upper"]

    dim_cols = [
        col
        for col in summary_df.columns
        if "dim" in col.lower()
        and col not in metadata_cols
        and col not in var_names
        and not any(keyword in col for keyword in stat_keywords)
    ]

    return dim_cols


def filter_draws(
    val: Any, idata: InferenceData, group: str, target: str, variable: pd.Series
) -> DataArray:
    """Filter draws from an InferenceData group based on variable values.

    Parameters
    ----------
    val : Any
        The value to filter by.
    idata : InferenceData
        The InferenceData object containing the draws.
    group : str
        The name of the group to filter from (e.g., 'posterior', 'posterior_predictive').
    target : str
        The target variable name within the group.
    variable : pd.Series
        The variable (pandas Series) to use for filtering.

    Returns
    -------
    DataArray
        An xarray DataArray containing the filtered draws.
    """
    coordinate_name = list(idata["data"].coords)[0]

    # Get indices where condition is true
    # np.logical_and.reduce is useful if there are multiple conditions (contrast values)
    idx = np.where(np.logical_and.reduce([idata["data"][variable.name] == val]))[0]
    draws = idata[group].isel({coordinate_name: idx})[target]

    # In the case of main and or parent parameters (e.g., distributional models)
    if coordinate_name in draws.coords:
        new_coords = np.arange(len(idx))
        draws = draws.assign_coords({coordinate_name: new_coords})

    return draws


def compare(
    idata: InferenceData,
    contrast: ComparisonVariable,
    target: str,
    group: str,
    comparison_fn: Callable,
) -> dict[str, DataArray]:
    """Compare samples in an InferenceData group given a `ComparisonVariable`.

    Parameters
    ----------
    idata : InferenceData
        The InferenceData object containing the samples to compare.
    contrast : ComparisonVariable
    The ComparisonVariable specifying the variable to create contrasts for.
    target : str
        The target variable name to compare within the group.
    group : str
        The name of the group to compare (e.g., 'posterior', 'posterior_predictive').
    comparison_fn : Callable
        The comparison function to apply to pairs of draws (e.g., difference, ratio).

    Returns
    -------
    dict[str, DataArray]
        A dictionary mapping comparison labels (e.g., "1_vs_2") to DataArrays
        containing the comparison results.
    """
    filter_fn = partial(
        filter_draws,
        idata=idata,
        group=group,
        target=target,
        variable=contrast.variable,
    )

    # Apply filter_draws over all contrast variable values
    filtered_draws = list(map(filter_fn, contrast.variable))
    # Generate unique pairs for each draw
    paired_draws = combinations(enumerate(filtered_draws), r=2)
    # Apply a comparison function to each pair
    res = {
        f"{contrast.variable[i]}_vs_{contrast.variable[j]}": comparison_fn(a, b)
        for (i, a), (j, b) in paired_draws
    }

    return res


def create_grid(variables: tuple[pd.Series, ...]) -> DataFrame:
    """Create a grid (cross-product) of data from `variables`.

    Takes multiple `variables` (Pandas Series) and creates a DataFrame containing all
    possible combinations of their values using Cartesian product.

    Parameters
    ----------
    variables : tuple[Series, ...]
        Tuple of pandas Series representing variables. Each Series should have a name
        that will be used as a column name in the resulting DataFrame.

    Returns
    -------
    DataFrame
        A DataFrame containing the Cartesian product of all variable values.
    """
    vals = [var.array for var in variables]
    names = [var.name for var in variables]
    product = pd.MultiIndex.from_product(vals, names=names)

    return product.to_frame(index=False)


def get_summary_stats(x: DataArray, prob: float, use_hdi: bool = True) -> DataFrame:
    """Compute summary statistics (mean and uncertainty interval) of an array.

    Parameters
    ----------
    x : DataArray
        The xarray DataArray containing posterior samples with 'chain' and 'draw' dimensions.
    prob : float
        Probability for the credible interval (between 0 and 1).

    Returns
    -------
    DataFrame
        A DataFrame containing summary statistics with columns:
        - 'estimate': posterior mean
        - 'lower_X%': lower bound of credible interval
        - 'upper_Y%': upper bound of credible interval
    """
    mean = x.mean(dim=("chain", "draw")).to_series().rename("estimate").to_frame()

    if use_hdi:
        hdi = az.hdi(x, hdi_prob=prob)
        # az.hdi returns a Dataset with MultiIndex (__obs__, hdi) and one column (var name)
        # We need to unstack 'hdi' to get columns, then flatten the resulting MultiIndex
        var_name = list(hdi.data_vars)[0]
        lower_bound = round((1 - prob) / 2, 4)
        upper_bound = 1 - lower_bound
        bounds = (
            hdi.to_dataframe()
            .unstack(level="hdi")[var_name]
            .rename(
                columns={
                    "lower": f"lower_{lower_bound * 100}%",
                    "higher": f"upper_{upper_bound * 100}%",
                }
            )
        )
    else:
        lower_bound = round((1 - prob) / 2, 4)
        upper_bound = 1 - lower_bound
        bounds = (
            x.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))
            .to_series()
            .unstack(level="quantile")
            .rename(
                columns={
                    lower_bound: f"lower_{lower_bound * 100}%",
                    upper_bound: f"upper_{upper_bound * 100}%",
                }
            )
        )

    stats = mean.join(bounds).reset_index().drop("__obs__", axis=1)

    return stats


def _build_predictions(
    model: Model,
    idata: InferenceData,
    focal_variable: pd.Series,
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]],
    target: str,
    pps: bool,
    transforms: dict | None,
    sample_new_groups: bool,
) -> tuple[InferenceData, DataFrame, list[str], str, str, Callable]:
    """Shared prediction pipeline for comparisons and slopes.

    Resolves variables, builds the data grid, runs model predictions,
    and creates inference data for downstream contrast/slope computation.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    focal_variable : Series
        The focal variable values (contrast values for comparisons,
        [x, x+eps] pairs for slopes).
    conditional : str, list, dict, or None
        Variables to condition on.
    target : str
        The target parameter to predict.
    pps : bool
        Whether to use posterior predictive samples.
    transforms : dict or None
        Dictionary of transformations.
    sample_new_groups : bool
        Whether to sample new group levels.

    Returns
    -------
    tuple
        (compare_idata, preds_data, context_columns, var, group, response_transform)
    """
    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    cond = ConditionalVariables.from_param(model.data, conditional)
    covariates = get_model_covariates(model).tolist()
    defaults = DefaultVariables.from_model(
        model.data, covariates, cond.names | {focal_variable.name}
    )

    # Unit level: copy observed data with focal variable substituted
    if not cond.variables:
        focal_name = focal_variable.name
        empirical_data = model.data[covariates].copy()
        preds_data = pd.concat(
            [empirical_data.assign(**{focal_name: val}) for val in focal_variable],
            ignore_index=True,
        )
        context_columns = [c for c in covariates if c != focal_name]
    # Grid level: Cartesian product
    else:
        all_vars = (focal_variable, *cond.variables, *defaults.variables)
        preds_data = create_grid(all_vars)
        context_columns = [var.name for var in (*cond.variables, *defaults.variables)]

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "sample_new_groups": sample_new_groups,
        "inplace": False,
    }
    preds_idata = model.predict(**pred_kwargs, **({} if not pps else {"kind": "response"}))
    group = "posterior_predictive" if pps else "posterior"
    var = response_name if pps or target is None else target

    compare_idata = create_inference_data(preds_idata, preds_data)

    return compare_idata, preds_data, context_columns, var, group, response_transform


def predictions(
    model: Model,
    idata: InferenceData,
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list[str] | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> Result:
    """Compute conditional adjusted predictions.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    conditional : ConditionalParam
        Variables to condition on for predictions.
    average_by : str, list or None
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
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    cond = ConditionalVariables.from_param(model.data, conditional)
    covariates = get_model_covariates(model).tolist()
    defaults = DefaultVariables.from_model(model.data, covariates, cond.names)

    # Unit level predictions
    if not cond.variables:
        preds_data = model.data[covariates].copy()
    # Data grid predictions
    else:
        all_vars = cond.variables + defaults.variables
        preds_data = create_grid(all_vars)

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "sample_new_groups": sample_new_groups,
        "inplace": False,
    }
    idata = model.predict(**pred_kwargs, **({} if not pps else {"kind": "response"}))
    group = "posterior_predictive" if pps else "posterior"
    var = response_name if pps or target is None else target
    y_hat = idata[group][var]

    stats_data = get_summary_stats(response_transform(y_hat), prob, use_hdi)
    summary_df = aggregate(data=preds_data.join(stats_data, on=None), by=average_by)

    return Result(summary=summary_df, draws=idata)


def plot_predictions(
    model: Model,
    idata: InferenceData,
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[dict[str, str]] = None,
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
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization. Use the 'theme' key
        to pass a dictionary of matplotlib rc parameters.
    subplot_kwargs : dict or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    Plot
        A Seaborn objects Plot. In Jupyter notebooks, the plot automatically displays.
        In scripts, call `.show()` to display. The returned Plot object can be
        customized before displaying using method chaining (e.g., `.label()`, `.theme()`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    var_names = _determine_plot_vars(conditional, average_by, model.data)

    result = predictions(
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

    # Add dimension columns for multi-output models (e.g., Categorical family)
    dim_cols = _extract_dim_columns(result.summary, var_names)
    all_var_names = var_names + dim_cols

    plot_config = PlottingConfig.from_params(all_var_names, subplot_kwargs, fig_kwargs)

    return plot(result.summary, plot_config)


def comparisons(
    model: Model,
    idata: InferenceData,
    contrast: str | dict[str, np.ndarray | list | int | float],
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list[str] | None = None,
    target: str = "mean",
    pps: bool = False,
    comparison: ComparisonFunc | str = "diff",
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> Result:
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
    average_by : str, list or None
        Variables to average comparisons over.
    target : str
        The target parameter to compare. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    comparison : ComparisonFunc or str
        Comparison function or string name. Built-in options: "diff" (difference),
        "ratio" (ratio), "lift" (relative difference). Default is "diff".
        Custom functions should accept (reference, contrast) DataArrays and return a DataArray.
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
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    comparison_fn = get_comparison_func(comparison)
    con = ComparisonVariable.from_param(model.data, contrast)

    compare_idata, preds_data, context_columns, var, group, response_transform = _build_predictions(
        model,
        idata,
        con.variable,
        conditional,
        target,
        pps,
        transforms,
        sample_new_groups,
    )

    compared_draws = compare(
        compare_idata,
        con,
        var,
        group,
        comparison_fn,
    )

    # Compute mean and uncertainty over (chain, draw)
    summary_draws = {
        k: get_summary_stats(response_transform(v), prob, use_hdi)
        for k, v in compared_draws.items()
    }
    # Comparison column name corresponds to the contrast values being compared (e.g., 1_vs_4)
    comparison_df = pd.concat(summary_draws, names=["comparison", "index"]).reset_index(level=0)

    summary_df = (
        preds_data.loc[preds_data[con.variable.name] == con.variable.iloc[0], context_columns]
        .reset_index(drop=True)
        .join(comparison_df, on=None)
    )

    summary_df = summary_df.rename(columns={"comparison": "value"})
    summary_df = aggregate(data=summary_df, by=average_by, preserve=["value"])

    # Add summary metadata
    estimate_type = comparison if isinstance(comparison, str) else comparison.__name__
    summary_df.insert(0, "term", con.variable.name)
    summary_df.insert(1, "estimate_type", estimate_type)

    return Result(summary=summary_df, draws=compare_idata)


def plot_comparisons(
    model: Model,
    idata: InferenceData,
    contrast: str | dict[str, np.ndarray | list | int | float],
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list | bool | None = None,
    target: str = "mean",
    pps: bool = False,
    comparison: ComparisonFunc | str = "diff",
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
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
    comparison : ComparisonFunc or str
        Comparison function or string name. Built-in options: "diff" (difference),
        "ratio" (ratio), "lift" (relative difference). Default is "diff".
        Custom functions should accept (reference, contrast) DataArrays and return a DataArray.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to comparisons.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization. Use the 'theme' key
        to pass a dictionary of matplotlib rc parameters.
    subplot_kwargs : Mapping[str, str] or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    Plot
        A Seaborn objects Plot. In Jupyter notebooks, the plot automatically displays.
        In scripts, call `.show()` to display. The returned Plot object can be
        customized before displaying using method chaining (e.g., `.label()`, `.theme()`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    var_names = _determine_plot_vars(conditional, average_by, model.data)

    result = comparisons(
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

    # Add dimension columns for multi-output models (e.g., Categorical family)
    dim_cols = _extract_dim_columns(result.summary, var_names)
    all_var_names = var_names + dim_cols

    plot_config = PlottingConfig.from_params(all_var_names, subplot_kwargs, fig_kwargs)

    return plot(result.summary, plot_config)


def slopes(
    model: Model,
    idata: InferenceData,
    wrt: str | dict[str, float | int],
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list[str] | None = None,
    eps: float = 1e-4,
    slope: str | SlopeFunc = "dydx",
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
) -> Result:
    """Compute conditional adjusted slopes.

    Slopes are computed using finite differences. The wrt variable is evaluated at
    [x, x + eps] and the slope is approximated as (f(x + eps) - f(x)) / eps.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    wrt : str or dict
        The predictor variable to compute the slope with respect to. Either a variable
        name (uses mean/mode as evaluation point) or a single-entry dict mapping
        variable name to a specific evaluation point.
    conditional : ConditionalParam
        Variables to condition on for slopes.
    average_by : str, list or None
        Variables to average slopes over.
    eps : float
        Perturbation size for finite differencing. Default is 1e-4.
    slope : str or SlopeFunc
        The type of slope to compute. Default is 'dydx'. Built-in options:
        'dydx' - unit change in wrt associated with a unit change in response.
        'eyex' - percent change in wrt associated with a percent change in response.
        'eydx' - unit change in wrt associated with a percent change in response.
        'dyex' - percent change in wrt associated with a unit change in response.
    target : str
        The target parameter to compute slopes for. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to predictions before differencing.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.

    Returns
    -------
    DataFrame
        A DataFrame containing the conditional adjusted slopes with summary statistics.

    Raises
    ------
    ValueError
        If prob is not between 0 and 1.
    TypeError
        If slope is not a callable or valid string.
    """
    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    slope_fn = get_slope_func(slope)
    wrt_var = SlopeVariable.from_param(model.data, wrt, eps)

    compare_idata, preds_data, context_columns, var, group, response_transform = _build_predictions(
        model,
        idata,
        wrt_var.variable,
        conditional,
        target,
        pps,
        transforms,
        sample_new_groups,
    )

    # Compute finite-differences
    x_val = wrt_var.variable.iloc[0]
    x_eps_val = wrt_var.variable.iloc[1]

    y_at_x = filter_draws(x_val, compare_idata, group, var, wrt_var.variable)
    y_at_x_eps = filter_draws(x_eps_val, compare_idata, group, var, wrt_var.variable)

    # Apply response transform before differencing
    y_at_x = response_transform(y_at_x)
    y_at_x_eps = response_transform(y_at_x_eps)

    dydx = (y_at_x_eps - y_at_x) / eps

    # Apply slope type scaling
    x_draws = xr.full_like(y_at_x, x_val)
    scaled_draws = slope_fn(dydx, x_draws, y_at_x)

    # Compute summary statistics
    stats = get_summary_stats(scaled_draws, prob, use_hdi)

    summary_df = (
        preds_data.loc[preds_data[wrt_var.variable.name] == x_val, context_columns]
        .reset_index(drop=True)
        .join(stats, on=None)
    )

    estimate_type = slope if isinstance(slope, str) else slope.__name__

    summary_df = aggregate(data=summary_df, by=average_by)

    # Add summary metadata
    summary_df.insert(0, "term", wrt_var.variable.name)
    summary_df.insert(1, "estimate_type", estimate_type)
    summary_df.insert(2, "value", wrt_var.variable.iloc[0])

    return Result(summary=summary_df, draws=compare_idata)


def plot_slopes(
    model: Model,
    idata: InferenceData,
    wrt: str | dict[str, float | int],
    conditional: Optional[str | list[str] | dict[str, np.ndarray | list | int | float]] = None,
    average_by: str | list | bool | None = None,
    eps: float = 1e-4,
    slope: str | SlopeFunc = "dydx",
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob: float = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool = False,
    fig_kwargs: Optional[dict[str, Any]] = None,
    subplot_kwargs: Optional[Mapping[str, str]] = None,
) -> Plot:
    """Plot conditional adjusted slopes.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : InferenceData
        InferenceData object containing the posterior samples.
    wrt : str or dict
        The predictor variable to compute the slope with respect to.
    conditional : ConditionalParam
        Variables to condition on for slopes.
    average_by : str or list or bool or None
        Variables to average slopes over.
    eps : float
        Perturbation size for finite differencing. Default is 1e-4.
    slope : str or SlopeFunc
        The type of slope to compute. Default is 'dydx'.
    target : str
        The target parameter to compute slopes for. Default is "mean".
    pps : bool
        Whether to use posterior predictive samples. Default is False.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float
        Probability for the credible interval. Default is from arviz rcParams.
    transforms : dict or None
        Dictionary of transformations to apply to predictions before differencing.
    sample_new_groups : bool
        Whether to sample new group levels. Default is False.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
    subplot_kwargs : Mapping[str, str] or None
        Overrides default plotting sequence (main, group, panel).

    Returns
    -------
    Plot
        A Seaborn objects Plot. In Jupyter notebooks, the plot automatically displays.
        In scripts, call `.show()` to display. The returned Plot object can be
        customized before displaying using method chaining (e.g., `.label()`, `.theme()`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    var_names = _determine_plot_vars(conditional, average_by, model.data)

    result = slopes(
        model=model,
        idata=idata,
        wrt=wrt,
        conditional=conditional,
        average_by=average_by,
        eps=eps,
        slope=slope,
        target=target,
        pps=pps,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
        sample_new_groups=sample_new_groups,
    )

    # Add dimension columns for multi-output models (e.g., Categorical family)
    dim_cols = _extract_dim_columns(result.summary, var_names)
    all_var_names = var_names + dim_cols

    plot_config = PlottingConfig.from_params(all_var_names, subplot_kwargs, fig_kwargs)

    return plot(result.summary, plot_config)
