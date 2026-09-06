import warnings
from functools import partial
from itertools import combinations
from typing import Any, Callable

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from pandas import DataFrame
from xarray import DataArray, DataTree

from bambi.interpret._typing import ConditionalValues
from bambi.interpret.ops import get_comparison_func, get_slope_func
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
    create_datatree,
    get_model_covariates,
    identity,
    resolve_target,
)
from bambi.interpret.validate import validate_prob
from bambi.models import Model
from bambi.utils import as_dataset


def _warn_deprecated_sample_new_groups(sample_new_groups: bool | None) -> None:
    if sample_new_groups is not None:
        warnings.warn(
            "'sample_new_groups' is deprecated and has no effect. Bambi now automatically "
            "determines how to predict for new groups based on the type of group.",
            FutureWarning,
            stacklevel=3,
        )


def _determine_plot_vars(
    conditional: str | list[str] | dict[str, ConditionalValues] | None,
    average_by: str | list[str] | None,
    model_data: DataFrame,
) -> list[str]:
    """Determine which variables to plot based on conditional and average_by parameters.

    Parameters
    ----------
    conditional : str, list[str], dict[str, ConditionalValues], or None
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
        if "_dim" in col.lower()
        and col not in metadata_cols
        and col not in var_names
        and not any(keyword in col for keyword in stat_keywords)
    ]

    return dim_cols


def filter_draws(
    val: Any, idata: DataTree, group: str, target: str, variable: pd.Series
) -> DataArray:
    """Filter draws from a DataTree group based on variable values.

    Parameters
    ----------
    val : Any
        The value to filter by.
    idata : DataTree
        The DataTree object containing the draws.
    group : str
        The name of the group to filter from (e.g., 'posterior', 'predictions').
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
    data_group = as_dataset(idata["data"])

    # Get indices where condition is true
    # np.logical_and.reduce is useful if there are multiple conditions (contrast values)
    idx = np.where(np.logical_and.reduce([data_group[variable.name] == val]))[0]
    draws = as_dataset(idata[group]).isel({coordinate_name: idx})[target]

    # In the case of main and or parent parameters (e.g., distributional models)
    if coordinate_name in draws.coords:
        new_coords = np.arange(len(idx))
        draws = draws.assign_coords({coordinate_name: new_coords})

    return draws


def compare(
    idata: DataTree,
    contrast: ComparisonVariable,
    target: str,
    group: str,
    comparison_fn: Callable,
) -> dict[str, DataArray]:
    """Compare samples in a DataTree group given a `ComparisonVariable`.

    Parameters
    ----------
    idata : DataTree
        The DataTree object containing the samples to compare.
    contrast : ComparisonVariable
        The ComparisonVariable specifying the variable to create contrasts for.
    target : str
        The target variable name to compare within the group.
    group : str
        The name of the group to compare (e.g., 'posterior', 'predictions').
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


def _compute_bounds(x: DataArray, prob: float, use_hdi: bool) -> DataFrame:
    """Compute lower/upper bounds for a single probability level.

    Parameters
    ----------
    x : DataArray
        The xarray DataArray containing posterior samples.
    prob : float
        Probability for the credible interval.
    use_hdi : bool
        Whether to use highest density interval.

    Returns
    -------
    DataFrame
        A DataFrame with lower and upper bound columns.
    """
    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    if use_hdi:
        hdi = az.hdi(x, prob=prob)
        bounds = (
            hdi.to_series()
            .unstack(level="ci_bound")
            .rename(
                columns={
                    "lower": f"lower_{lower_bound * 100}%",
                    "upper": f"upper_{upper_bound * 100}%",
                }
            )
        )
    else:
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

    return bounds


def get_summary_stats(
    x: DataArray, prob: float | list[float] | None, use_hdi: bool = True
) -> DataFrame:
    """Compute summary statistics (mean and uncertainty intervals) of an array.

    Parameters
    ----------
    x : DataArray
        The xarray DataArray containing posterior samples with 'chain' and 'draw' dimensions.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals (each between 0 and 1).
        When a list is provided, multiple pairs of lower/upper columns are returned,
        sorted by interval width (widest first). Pass None to return point estimates only.
    use_hdi : bool
        Whether to compute highest density or equal-tailed intervals. Default is True.

    Returns
    -------
    DataFrame
        A DataFrame containing summary statistics with columns:
        - 'estimate': posterior mean
        - 'lower_X%' / 'upper_Y%': bounds for each probability level
    """
    if prob is None:
        prob = []
    elif isinstance(prob, (int, float)):
        prob = [prob]

    # Sort descending so widest interval columns come first
    prob = sorted(prob, reverse=True)

    mean = x.mean(dim=("chain", "draw")).to_series().rename("estimate").to_frame()

    bounds_list = [_compute_bounds(x, p, use_hdi) for p in prob]
    if bounds_list:
        mean = mean.join(pd.concat(bounds_list, axis=1))

    stats = mean.reset_index().drop("__obs__", axis=1)

    return stats


def _join_prediction_data(preds_data: DataFrame, stats_data: DataFrame) -> DataFrame:
    """Attach output statistics to each row in a prediction grid.

    A multivariate response has one row of summary statistics per output level,
    while the prediction grid has one row per observation.
    Repeat each grid row for its output levels before joining by position.
    """
    n_levels, remainder = divmod(len(stats_data), len(preds_data))
    if remainder:
        raise ValueError(
            "The number of prediction statistics must be a multiple of the prediction grid size."
        )

    indexes = np.repeat(np.arange(len(preds_data)), n_levels)
    expanded_data = preds_data.iloc[indexes].reset_index(drop=True)
    return expanded_data.join(stats_data.reset_index(drop=True))


def _build_predictions(
    model: Model,
    idata: DataTree,
    focal_variable: pd.Series,
    conditional: str | list[str] | dict[str, ConditionalValues] | None,
    target: str,
    transforms: dict | None,
) -> tuple[DataTree, DataFrame, list[str], str, str, Callable]:
    """Shared prediction pipeline for comparisons and slopes.

    Resolves variables, builds the data grid, runs model predictions,
    and creates inference data for downstream contrast/slope computation.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    focal_variable : Series
        The focal variable values (contrast values for comparisons,
        [x, x+eps] pairs for slopes).
    conditional : str, list, dict, or None
        Variables to condition on.
    target : str
        Which quantity to extract. `"mean"` for the posterior of the parent
        parameter, the response variable name for posterior predictive samples, or a
        distributional component name.
    transforms : dict or None
        Dictionary of transformations.
    Returns
    -------
    tuple
        (compare_idata, preds_data, context_columns, var_name, group, response_transform)
    """
    transforms = transforms or {}

    target_info = resolve_target(model, target)
    response_transform = transforms.get(target_info.response_name, identity)

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

    if model.response_term.is_binomial:
        # If response is binomial, trials is not a literal, and it's not passed as conditional,
        # set trials to 1 for predictions.
        trials = model.response_term.components[0].call.args[1]
        trials_name = getattr(trials, "name", None)
        if trials_name is not None and trials_name not in preds_data:
            preds_data[trials_name] = 1

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "inplace": False,
    }
    preds_idata = model.predict(**pred_kwargs, kind=target_info.predict_kind)

    compare_idata = create_datatree(preds_idata, preds_data)

    return (
        compare_idata,
        preds_data,
        context_columns,
        target_info.var_name,
        target_info.group,
        response_transform,
    )


def predictions(
    model: Model,
    idata: DataTree,
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list[str] | None = None,
    target: str = "mean",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
) -> Result:
    """Compute conditional adjusted predictions.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for predictions.
    average_by : str, list or None
        Variables to average predictions over.
    target : str
        Which quantity to extract. `"mean"` (default) for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for posterior
        predictive samples. Pass a distributional component name (e.g. `"sigma"`) for the
        posterior of that component.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, multiple nested intervals are computed.
        Pass None to omit credible intervals.
    transforms : dict or None
        Dictionary of transformations to apply to predictions.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.

    Returns
    -------
    Result
        A named tuple with `.summary` (DataFrame of summary statistics) and
        `.draws` (DataTree of raw posterior draws).

    Raises
    ------
    ValueError
        If any prob value is not between 0 and 1.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

    prob = validate_prob(prob)

    transforms = transforms or {}

    target_info = resolve_target(model, target)
    response_transform = transforms.get(target_info.response_name, identity)

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

    if model.response_term.is_binomial:
        # If response is binomial, trials is not a literal, and it's not passed as conditional,
        # set trials to 1 for predictions.
        trials = model.response_term.components[0].call.args[1]
        trials_name = getattr(trials, "name", None)
        if trials_name is not None and trials_name not in preds_data:
            preds_data[trials_name] = 1

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "inplace": False,
    }
    idata = model.predict(**pred_kwargs, kind=target_info.predict_kind)
    y_hat = as_dataset(idata[target_info.group])[target_info.var_name]

    stats_data = get_summary_stats(response_transform(y_hat), prob, use_hdi)
    summary_df = aggregate(
        data=_join_prediction_data(preds_data, stats_data),
        by=average_by,
        preserve=_extract_dim_columns(stats_data, []),
    )

    return Result(summary=summary_df, draws=idata)


def plot_predictions(
    model: Model,
    idata: DataTree,
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list[str] | None = None,
    target: str = "mean",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
    fig_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, str] | None = None,
    on: Axes | Figure | SubFigure | None = None,
) -> Figure:
    """Plot conditional adjusted predictions.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for predictions.
    average_by : str, list[str], or None
        Variables to average predictions over.
    target : str
        Which quantity to extract. `"mean"` (default) for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for posterior
        predictive samples. Pass a distributional component name (e.g. `"sigma"`) for the
        posterior of that component.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, nested bands with decreasing opacity
        are drawn. Pass None to omit bands.
    transforms : dict or None
        Dictionary of transformations to apply to predictions.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
        Use the 'theme' key to pass a dictionary of matplotlib rc parameters.
    subplot_kwargs : dict or None
        Overrides default plotting sequence (main, group, panel).
    on : Axes, Figure, SubFigure, or None
        Matplotlib target on which to draw the plot. If None, a new figure is created.

    Returns
    -------
    Figure
        A Matplotlib Figure. In Jupyter notebooks, the figure automatically displays.
        In scripts, call `.show()` to display it. The returned Figure can be customized
        through its axes (e.g., `figure.axes[0].set_title(...)`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

    var_names = _determine_plot_vars(conditional, average_by, model.data)

    result = predictions(
        model=model,
        idata=idata,
        conditional=conditional,
        average_by=average_by,
        target=target,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
    )

    df_plot = result.summary
    dim_columns = _extract_dim_columns(df_plot, var_names)

    # Extract original variable names from dimension columns for plotting without `dim` suffix.
    for column in dim_columns:
        new_column = column.replace("_dim", "")
        var_names.append(new_column)
        df_plot[new_column] = df_plot[column]

    plot_config = PlottingConfig.from_params(var_names, subplot_kwargs, fig_kwargs)

    return plot(df_plot, plot_config, on=on)


def comparisons(
    model: Model,
    idata: DataTree,
    contrast: str | dict[str, ConditionalValues],
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list[str] | None = None,
    target: str = "mean",
    comparison: Callable[[DataArray, DataArray], DataArray] | str = "diff",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
) -> Result:
    """Compute conditional adjusted comparisons.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    contrast : str or dict[str, ConditionalValues]
        Variable(s) to create contrasts for.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for comparisons.
    average_by : str, list[str], or None
        Variables to average comparisons over.
    target : str
        The target parameter to compare. Default is "mean".
    comparison : Callable[[DataArray, DataArray], DataArray] or str
        Comparison function or string name. Built-in options: "diff" (difference),
        "ratio" (ratio), "lift" (relative difference). Default is "diff".
        Custom functions should accept (reference, contrast) DataArrays and return a DataArray.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, multiple nested intervals are computed.
        Pass None to omit credible intervals.
    transforms : dict or None
        Dictionary of transformations to apply to comparisons.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.

    Returns
    -------
    Result
        A named tuple with `.summary` (DataFrame of summary statistics) and
        `.draws` (DataTree of raw posterior draws).

    Raises
    ------
    ValueError
        If any prob value is not between 0 and 1.
    TypeError
        If comparison is not a callable or valid string.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

    prob = validate_prob(prob)

    comparison_fn = get_comparison_func(comparison)
    con = ComparisonVariable.from_param(model.data, contrast)

    compare_idata, preds_data, context_columns, var, group, response_transform = _build_predictions(
        model,
        idata,
        con.variable,
        conditional,
        target,
        transforms,
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

    context_rows = preds_data[con.variable.name] == con.variable.iloc[0]
    summary_df = _join_prediction_data(preds_data.loc[context_rows, context_columns], comparison_df)

    summary_df = summary_df.rename(columns={"comparison": "value"})
    summary_df = aggregate(
        data=summary_df,
        by=average_by,
        preserve=["value", *_extract_dim_columns(comparison_df, [])],
    )

    # Add summary metadata
    estimate_type = comparison if isinstance(comparison, str) else comparison.__name__
    summary_df.insert(0, "term", con.variable.name)
    summary_df.insert(1, "estimate_type", estimate_type)

    return Result(summary=summary_df, draws=compare_idata)


def plot_comparisons(
    model: Model,
    idata: DataTree,
    contrast: str | dict[str, ConditionalValues],
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list | None = None,
    target: str = "mean",
    comparison: Callable[[DataArray, DataArray], DataArray] | str = "diff",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
    fig_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, str] | None = None,
    on: Axes | Figure | SubFigure | None = None,
) -> Figure:
    """Plot conditional adjusted comparisons.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    contrast : str or dict[str, ConditionalValues]
        Variable(s) to create contrasts for.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for comparisons.
    average_by : str or list or None
        Variables to average comparisons over.
    target : str
        Which quantity to extract. `"mean"` (default) for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for posterior
        predictive samples. Pass a distributional component name (e.g. `"sigma"`) for the
        posterior of that component.
    comparison : Callable[[DataArray, DataArray], DataArray] or str
        Comparison function or string name. Built-in options: "diff" (difference),
        "ratio" (ratio), "lift" (relative difference). Default is "diff".
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, nested bands with decreasing opacity
        are drawn. Pass None to omit bands.
    transforms : dict or None
        Dictionary of transformations to apply to comparisons.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
        Use the 'theme' key to pass a dictionary of matplotlib rc parameters.
    subplot_kwargs : dict[str, str] or None
        Overrides default plotting sequence (main, group, panel).
    on : Axes, Figure, SubFigure, or None
        Matplotlib target on which to draw the plot. If None, a new figure is created.

    Returns
    -------
    Figure
        A Matplotlib Figure. In Jupyter notebooks, the figure automatically displays.
        In scripts, call `.show()` to display it. The returned Figure can be customized
        through its axes (e.g., `figure.axes[0].set_title(...)`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

    var_names = _determine_plot_vars(conditional, average_by, model.data)

    result = comparisons(
        model=model,
        idata=idata,
        contrast=contrast,
        conditional=conditional,
        average_by=average_by,
        target=target,
        comparison=comparison,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
    )

    df_plot = result.summary
    dim_columns = _extract_dim_columns(df_plot, var_names)

    # Extract original variable names from dimension columns for plotting without `dim` suffix.
    for column in dim_columns:
        new_column = column.replace("_dim", "")
        var_names.append(new_column)
        df_plot[new_column] = df_plot[column]

    plot_config = PlottingConfig.from_params(var_names, subplot_kwargs, fig_kwargs)

    return plot(df_plot, plot_config, on=on)


def slopes(
    model: Model,
    idata: DataTree,
    wrt: str | dict[str, float | int],
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list[str] | None = None,
    eps: float = 1e-4,
    slope: str | Callable[[DataArray, DataArray, DataArray], DataArray] = "dydx",
    target: str = "mean",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
) -> Result:
    """Compute conditional adjusted slopes.

    Slopes are computed using finite differences. The wrt variable is evaluated at
    [x, x + eps] and the slope is approximated as (f(x + eps) - f(x)) / eps.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    wrt : str or dict
        The predictor variable to compute the slope with respect to. Either a variable
        name (uses mean/mode as evaluation point) or a single-entry dict mapping
        variable name to a specific evaluation point.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for slopes.
    average_by : str, list or None
        Variables to average slopes over.
    eps : float
        Perturbation size for finite differencing. Default is 1e-4.
    slope : str or Callable[[DataArray, DataArray, DataArray], DataArray]
        Slope function or string name. Built-in options: "dydx" (unit/unit),
        "eyex" (percent/percent), "eydx" (percent/unit), "dyex" (unit/percent).
        Default is "dydx". Custom functions should accept (derivative, x, y) DataArrays
        and return a DataArray.
    target : str
        Which quantity to extract. `"mean"` (default) for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for posterior
        predictive samples. Pass a distributional component name (e.g. `"sigma"`) for the
        posterior of that component.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, multiple nested intervals are computed.
        Pass None to omit credible intervals.
    transforms : dict or None
        Dictionary of transformations to apply to predictions before differencing.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.

    Returns
    -------
    Result
        A named tuple with `.summary` (DataFrame of summary statistics) and
        `.draws` (DataTree of raw posterior draws).

    Raises
    ------
    ValueError
        If any prob value is not between 0 and 1.
    TypeError
        If slope is not a callable or valid string.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

    prob = validate_prob(prob)

    slope_fn = get_slope_func(slope)
    wrt_var = SlopeVariable.from_param(model.data, wrt, eps)

    compare_idata, preds_data, context_columns, var, group, response_transform = _build_predictions(
        model,
        idata,
        wrt_var.variable,
        conditional,
        target,
        transforms,
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

    estimate_type = slope if isinstance(slope, str) else slope.__name__

    context_rows = preds_data[wrt_var.variable.name] == x_val
    summary_df = aggregate(
        data=_join_prediction_data(preds_data.loc[context_rows, context_columns], stats),
        by=average_by,
        preserve=_extract_dim_columns(stats, []),
    )

    # Add summary metadata
    summary_df.insert(0, "term", wrt_var.variable.name)
    summary_df.insert(1, "estimate_type", estimate_type)
    summary_df.insert(2, "value", wrt_var.variable.iloc[0])

    return Result(summary=summary_df, draws=compare_idata)


def plot_slopes(
    model: Model,
    idata: DataTree,
    wrt: str | dict[str, float | int],
    conditional: str | list[str] | dict[str, ConditionalValues] | None = None,
    average_by: str | list[str] | None = None,
    eps: float = 1e-4,
    slope: str | Callable[[DataArray, DataArray, DataArray], DataArray] = "dydx",
    target: str = "mean",
    use_hdi: bool = True,
    prob: float | list[float] | None = az.rcParams["stats.ci_prob"],
    transforms: dict | None = None,
    sample_new_groups: bool | None = None,
    fig_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, str] | None = None,
    on: Axes | Figure | SubFigure | None = None,
) -> Figure:
    """Plot conditional adjusted slopes.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    idata : DataTree
        DataTree object containing the posterior samples.
    wrt : str or dict
        The predictor variable to compute the slope with respect to.
    conditional : str, list[str], dict[str, ConditionalValues], or None
        Variables to condition on for slopes.
    average_by : str or list[str] or None
        Variables to average slopes over.
    eps : float
        Perturbation size for finite differencing. Default is 1e-4.
    slope : Callable[[DataArray, DataArray, DataArray], DataArray] or str
        Slope function or string name. Built-in options: "dydx" (unit/unit),
        "eyex" (percent/percent), "eydx" (percent/unit), "dyex" (unit/percent).
        Default is "dydx".
    target : str
        Which quantity to extract. `"mean"` (default) for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for posterior
        predictive samples. Pass a distributional component name (e.g. `"sigma"`) for the
        posterior of that component.
    use_hdi : bool
        Whether to use highest density interval. Default is True.
    prob : float or list[float] or None
        Probability or list of probabilities for credible intervals. Default is from
        arviz rcParams. When a list is provided, nested bands with decreasing opacity
        are drawn. Pass None to omit bands.
    transforms : dict or None
        Dictionary of transformations to apply to predictions before differencing.
    sample_new_groups : bool or None
        Deprecated and has no effect. Bambi automatically determines how to predict for new
        groups based on the type of group. Default is None.
    fig_kwargs : dict or None
        Additional keyword arguments for figure customization.
        Use the 'theme' key to pass a dictionary of matplotlib rc parameters.
    subplot_kwargs : dict[str, str] or None
        Overrides default plotting sequence (main, group, panel).
    on : Axes, Figure, SubFigure, or None
        Matplotlib target on which to draw the plot. If None, a new figure is created.

    Returns
    -------
    Figure
        A Matplotlib Figure. In Jupyter notebooks, the figure automatically displays.
        In scripts, call `.show()` to display it. The returned Figure can be customized
        through its axes (e.g., `figure.axes[0].set_title(...)`).

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    _warn_deprecated_sample_new_groups(sample_new_groups)

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
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
    )

    df_plot = result.summary
    dim_columns = _extract_dim_columns(df_plot, var_names)

    # Extract original variable names from dimension columns for plotting without `dim` suffix.
    for column in dim_columns:
        new_column = column.replace("_dim", "")
        var_names.append(new_column)
        df_plot[new_column] = df_plot[column]

    plot_config = PlottingConfig.from_params(var_names, subplot_kwargs, fig_kwargs)

    return plot(df_plot, plot_config, on=on)
