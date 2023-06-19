# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from typing import Union
import warnings

import arviz as az
import numpy as np

from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

import bambi as bmb
from bambi.utils import listify, get_aliased_name
from bambi.plots.effects import predictions, comparisons
from bambi.plots.plot_types import plot_numeric, plot_categoric
from bambi.plots.utils import get_covariates


def plot_cap(
    model: bmb.Model,
    idata: az.InferenceData,
    covariates: Union[str, list],
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    hdi_prob=None,
    transforms=None,
    legend: bool = True,
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

    if transforms is None:
        transforms = {}

    cap_data = predictions(
        model,
        idata,
        covariates,
        target=target,
        pps=pps,
        use_hdi=use_hdi,
        hdi_prob=hdi_prob,
        transforms=transforms,
    )

    response_name = get_aliased_name(model.response_component.response_term)
    covariates = get_covariates(covariates)
    main, group, panel = covariates.main, covariates.group, covariates.panel

    if ax is None:
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
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

    if is_numeric_dtype(cap_data[main]):
        axes = plot_numeric(covariates, cap_data, transforms, legend, axes)
    elif is_categorical_dtype(cap_data[main]) or is_string_dtype(cap_data[main]):
        axes = plot_categoric(covariates, cap_data, legend, axes)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    ylabel = response_name if target == "mean" else target
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=main, ylabel=ylabel)

    return fig, axes


def plot_comparison(
    model: bmb.Model,
    idata: az.InferenceData,
    contrast: Union[str, dict, list],
    conditional: Union[str, dict, list],
    comparison_type: str = "diff",
    target: str = "mean",
    use_hdi: bool = True,
    hdi_prob=None,
    transforms=None,
    legend: bool = True,
    ax=None,
    fig_kwargs=None,
    subplot_kwargs=None,
):
    """Plot Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    contrast : str, dict, list
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
    fig_kwargs : optional
        Keyword arguments passed to the matplotlib figure function as a dict. For example,
        ``fig_kwargs=dict(figsize=(11, 8)), sharey=True`` would make the figure 11 inches wide
        by 8 inches high and would share the y-axis values.
    subplot_kwargs : optional
        Keyword arguments passed to the matplotlib subplot function as a dict. This allows you
        to determine the covariates used for the horizontal, color, and panel axes. For example,
        ``subplot_kwargs=dict(horizontal="x", color="y", panel="z")`` would plot the horizontal
        axis as ``x``, the color axis as ``y``, and the panel axis as ``z``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        A tuple with the figure and the axes.

    Raises
    ------
    Warning
        When number of ``contrast_values`` is greater than 2.
    """

    if isinstance(contrast, dict):
        contrast_name, contrast_level = next(iter(contrast.items()))
        if len(contrast_level) > 2:
            warnings.warn(
                f"Attempting to plot when contrast {contrast_name} has {len(contrast_level)} values."
            )

    contrast_df = comparisons(
        model=model,
        idata=idata,
        contrast=contrast,
        conditional=conditional,
        comparison_type=comparison_type,
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

    covariates = get_covariates(conditional)

    if subplot_kwargs:
        for key, value in subplot_kwargs.items():
            setattr(covariates, key, value)

    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)

    if ax is None:
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        panels_n = len(np.unique(contrast_df[covariates.panel])) if covariates.panel else 1
        rows, cols = default_grid(panels_n)
        fig, axes = create_axes_grid(panels_n, rows, cols, backend_kwargs=fig_kwargs)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if isinstance(axes[0], np.ndarray):
            fig = axes[0][0].get_figure()
        else:
            fig = axes[0].get_figure()

    if is_numeric_dtype(contrast_df[covariates.main]):
        # main condition variable can be numeric but at the same time only
        # a few values, so it is treated as categoric
        if np.unique(contrast_df[covariates.main]).shape[0] <= 5:
            axes = plot_categoric(covariates, contrast_df, legend, axes)
        else:
            axes = plot_numeric(covariates, contrast_df, transforms, legend, axes)
    elif is_categorical_dtype(contrast_df[covariates.main]) or is_string_dtype(
        contrast_df[covariates.main]
    ):
        axes = plot_categoric(covariates, contrast_df, legend, axes)
    else:
        raise TypeError("Main covariate must be numeric or categoric.")

    response_name = get_aliased_name(model.response_component.response_term)
    ylabel = response_name if target == "mean" else target
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=covariates.main, ylabel=ylabel)
    return fig, axes
