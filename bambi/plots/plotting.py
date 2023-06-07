# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from statistics import mode
from typing import Union, Callable, Tuple, Any

import arviz as az
import numpy as np
import pandas as pd

from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
from formulae.terms.call import Call
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

import bambi as bmb
from bambi.utils import listify, get_aliased_name
from bambi.plots.create_data import create_cap_data, create_comparisons_data
from bambi.plots.effects import comparisons
from bambi.plots.plot_types import plot_numeric, plot_categoric
from bambi.plots.utils import identity, contrast_dtype


def plot_cap(
    model,
    idata,
    covariates,
    target="mean",
    pps=False,
    use_hdi=True,
    hdi_prob=None,
    transforms=None,
    legend=True,
    ax=None,
    fig_kwargs=None,
):
    """Plot Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    covariates : list or dict
        A sequence of between one and three names of variables or a dict of length between one
        and three.
        If a sequence, the first variable is taken as the main variable,
        mapped to the horizontal axis. If present, the second name is a coloring/grouping variable,
        and the third is mapped to different plot panels.
        If a dictionary, keys must be taken from ("horizontal", "color", "panel") and the values
        are the names of the variables.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    pps: bool, optional
        Whether to plot the posterior predictive samples. Defaults to ``False``.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    hdi_prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A matplotlib axes object or a sequence of them. If None, this function instantiates a
        new axes object. Defaults to ``None``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        A tuple with the figure and the axes.

    Raises
    ------
    ValueError
        When ``level`` is not within 0 and 1.
        When the main covariate is not numeric or categoric.
    """

    covariate_kinds = ("horizontal", "color", "panel")
    if not isinstance(covariates, dict):
        covariates = listify(covariates)
        covariates = dict(zip(covariate_kinds, covariates))
    else:
        assert covariate_kinds[0] in covariates
        assert set(covariates).issubset(set(covariate_kinds))

    assert 1 <= len(covariates) <= 3

    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]

    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")

    cap_data = create_cap_data(model, covariates)

    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response_transform = transforms.get(response_name, identity)

    if pps:
        idata = model.predict(idata, data=cap_data, inplace=False, kind="pps")
        y_hat = response_transform(idata.posterior_predictive[response_name])
        y_hat_mean = y_hat.mean(("chain", "draw"))
    else:
        idata = model.predict(idata, data=cap_data, inplace=False)
        y_hat = response_transform(idata.posterior[f"{response_name}_{target}"])
        y_hat_mean = y_hat.mean(("chain", "draw"))

    if use_hdi and pps:
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[response_name].T
    elif use_hdi:
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{response_name}_{target}"].T
    else:
        lower_bound = round((1 - hdi_prob) / 2, 4)
        upper_bound = 1 - lower_bound
        y_hat_bounds = y_hat.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))

    if ax is None:
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        panel = covariates.get("panel", None)
        panels_n = len(np.unique(cap_data[panel])) if panel else 1
        rows, cols = default_grid(panels_n)
        fig, axes = create_axes_grid(panels_n, rows, cols, backend_kwargs=fig_kwargs)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if isinstance(axes[0], np.ndarray):
            fig = axes[0][0].get_figure()
        else:
            fig = axes[0].get_figure()

    main = covariates.get("horizontal")
    if is_numeric_dtype(cap_data[main]):
        # axes = _plot_cap_numeric(
        #     covariates, cap_data, y_hat_mean, y_hat_bounds, transforms, legend, axes
        # )
        axes = plot_numeric(
            covariates, cap_data, y_hat_mean, y_hat_bounds, transforms, legend, axes
        )
    elif is_categorical_dtype(cap_data[main]) or is_string_dtype(cap_data[main]):
        # axes = _plot_cap_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, axes)
        axes = plot_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, axes)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    ylabel = response_name if target == "mean" else target
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=main, ylabel=ylabel)

    return fig, axes


def plot_comparison(
        model: bmb.Model,
        idata: az.InferenceData,
        contrast_predictor: Union[str, dict, list],
        conditional: Union[str, dict, list],
        #comparison: str = "diff",
        target: str = "mean",
        use_hdi: bool = True,
        hdi_prob=None,
        transforms= None,
        legend: bool =True,
        ax=None,
        fig_kwargs=None
):    
    """Plot Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    contrast_predictor : str, dict, list
        The predictor name whose contrast we would like to compare.
    conditional : str, dict, list
        The covariates we would like to condition on.
    comparison : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    hdi_prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A matplotlib axes object or a sequence of them. If None, this function instantiates a
        new axes object. Defaults to ``None``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        A tuple with the figure and the axes.

    Raises
    ------
    ValueError
        When ``level`` is not within 0 and 1.
        When the main covariate is not numeric or categoric.
    """

    comparisons_df, contrast_df, idata = comparisons(
        model=model,
        idata=idata,
        contrast_predictor=contrast_predictor,
        conditional=conditional,
        target=target,
        use_hdi=use_hdi,
        hdi_prob=hdi_prob,
        transforms=transforms,
    )
    
    covariate_kinds = ("horizontal", "color", "panel")
    # if not dict, then user did not pass values to condition on
    if not isinstance(conditional, dict):
        conditional = listify(conditional)
        conditional = dict(zip(covariate_kinds, conditional))
    # if dict, user passed values to condition on
    elif isinstance(conditional, dict):
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))
    
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)

    if ax is None:
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        panel = conditional.get("panel", None)
        panels_n = len(np.unique(contrast_df[panel])) if panel else 1
        rows, cols = default_grid(panels_n)
        fig, axes = create_axes_grid(panels_n, rows, cols, backend_kwargs=fig_kwargs)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if isinstance(axes[0], np.ndarray):
            fig = axes[0][0].get_figure()
        else:
            fig = axes[0].get_figure()
    
    main = conditional.get("horizontal")

    y_hat_bounds = np.transpose(
        contrast_df[["contrast_comparison_lower", "contrast_comparison_upper"]].values
    )

    if is_numeric_dtype(contrast_df[main]):
        # main condition variable can be numeric, but only a few values
        # so it is treated as categoric
        if np.unique(contrast_df[main]).shape[0] <= 5:
            axes = plot_categoric(
                conditional,
                contrast_df,
                contrast_df["contrast_comparison"],
                y_hat_bounds,
                legend,
                axes
            )
        else:
            axes = plot_numeric(
                conditional, 
                contrast_df,
                contrast_df["contrast_comparison"],
                y_hat_bounds,
                transforms,
                legend,
                axes
            )

    elif is_categorical_dtype(contrast_df[main]) or is_string_dtype(contrast_df[main]):
        axes = plot_categoric(
            conditional, 
            contrast_df,
            contrast_df["contrast_comparison"],
            y_hat_bounds,
            legend,
            axes
        )
    else:
        raise ValueError("Main covariate must be numeric or categoric.")
    
    response_name = get_aliased_name(model.response_component.response_term)
    ylabel = response_name if target == "mean" else target
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=main, ylabel=ylabel)

    return fig, axes, comparisons_df, contrast_df, idata
