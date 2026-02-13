from typing import Any, Callable, Mapping, Optional

import arviz as az
import numpy as np
import pandas as pd
from arviz import InferenceData
from pandas import DataFrame
from seaborn.objects import Plot
from xarray import DataArray

from bambi import Model
from bambi.interpret.utils import (
    aggregate,
    get_model_covariates,
    get_response_and_target,
    identity,
)

from .helpers import compare, create_inference_data
from .plots import PlottingConfig, plot
from .types import ConditionalVariables, ContrastVariable, DefaultVariables


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
        bounds = (
            hdi.to_dataframe()
            .unstack(level="hdi")[var_name]
            .rename(columns={"lower": f"lower_{prob}", "higher": f"upper_{prob}"})
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
    average_by: str | list[str] | None = None,
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
        raise ValueError(
            f"'prob' must be greater than 0 and smaller than 1. It is {prob}."
        )

    transforms = transforms or {}

    response_name, target = get_response_and_target(model, target)
    response_transform = transforms.get(response_name, identity)

    cond = ConditionalVariables.from_param(model.data, conditional)
    covariates = get_model_covariates(model).tolist()
    defaults = DefaultVariables.from_model(model.data, covariates, cond.names)

    if not cond.variables:
        preds_data = model.data[covariates].copy()
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
    var = response_name if pps else target
    y_hat = idata[group][var]

    stats_data = get_summary_stats(response_transform(y_hat), prob, use_hdi)
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
        Displays and returns a Seaborn objects 'Plot'.

    Raises
    ------
    ValueError
        If more than 3 conditional variables are provided without averaging.
    """
    cond = ConditionalVariables.from_param(model.data, conditional)
    provided_var_names = [var.name for var in cond.variables]
    plot_config = PlottingConfig.from_params(
        provided_var_names, subplot_kwargs, fig_kwargs
    )

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

    p = plot(summary_df, plot_config)

    return p


def comparisons(
    model: Model,
    idata: InferenceData,
    contrast: str | dict[str, np.ndarray | list | int | float],
    conditional: Optional[
        str | list[str] | dict[str, np.ndarray | list | int | float]
    ] = None,
    average_by: str | list[str] | None = None,
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
    average_by : str, list or None
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

    con = ContrastVariable.from_param(model.data, contrast)
    cond = ConditionalVariables.from_param(model.data, conditional)
    covariates = get_model_covariates(model).tolist()
    defaults = DefaultVariables.from_model(
        model.data, covariates, cond.names | {con.variable.name}
    )

    if not cond.variables:
        preds_data = model.data.copy()
    else:
        all_vars = (con.variable, *cond.variables, *defaults.variables)
        preds_data = create_grid(all_vars)

    pred_kwargs = {
        "idata": idata,
        "data": preds_data,
        "sample_new_groups": sample_new_groups,
        "inplace": False,
    }
    preds_idata = model.predict(
        **pred_kwargs, **({} if not pps else {"kind": "response"})
    )
    group = "posterior_predictive" if pps else "posterior"
    var = response_name if pps else target

    compare_idata = create_inference_data(preds_idata, preds_data)
    compared_draws = compare(
        compare_idata,
        con,
        var,
        group,
        comparison_fn=comparison_fn,
    )

    # Compute mean and uncertainty over (chain, draw)
    summary_draws = {
        k: get_summary_stats(response_transform(v), prob, use_hdi)
        for k, v in compared_draws.items()
    }
    # Comparison column name corresponds to the contrast values being compared (e.g., 1_vs_4)
    comparison_df = pd.concat(summary_draws, names=["comparison", "index"]).reset_index(
        level=0
    )
    # Use index of both dataframes to join on
    summary_df = (
        preds_data[[var.name for var in cond.variables]]
        .drop_duplicates()
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
        Additional keyword arguments for figure customization. Use the 'theme' key
        to pass a dictionary of matplotlib rc parameters.
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
    cond = ConditionalVariables.from_param(model.data, conditional)
    provided_var_names = [var.name for var in cond.variables]
    plot_config = PlottingConfig.from_params(
        provided_var_names, subplot_kwargs, fig_kwargs
    )

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

    p = plot(summary_df, plot_config)

    return p
