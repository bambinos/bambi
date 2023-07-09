# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from typing import Union

import arviz as az
from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
import numpy as np
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi.models import Model
from bambi.plots.effects import comparisons, predictions
from bambi.plots.plot_types import plot_categoric, plot_numeric
from bambi.plots.utils import get_covariates, ConditionalInfo
from bambi.utils import get_aliased_name, listify


def plot_cap(
    model: Model,
    idata: az.InferenceData,
    covariates: Union[str, list],
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob=None,
    transforms=None,
    legend: bool = True,
    ax=None,
    fig_kwargs=None,
    subplot_kwargs=None,
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
        A sequence of between one and three names of variables in the model.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    pps: bool, optional
        Whether to plot the posterior predictive samples. Defaults to ``False``.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
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
        Keyword arguments used to determine the covariates used for the horizontal, group,
        and panel axes. For example, ``subplot_kwargs=dict(main="x", group="y", panel="z")`` would
        plot the horizontal axis as ``x``, the color (hue) as ``y``, and the panel axis as ``z``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        A tuple with the figure and the axes.

    Raises
    ------
    ValueError
        When ``level`` is not within 0 and 1.
        When the main covariate is not numeric or categoric.

    TypeError
        When ``covariates`` is not a string or a list of strings.
    """

    covariate_kinds = ("main", "group", "panel")
    if isinstance(covariates, dict):
        raise TypeError("covariates must be a string or a list of strings.")

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
        prob=prob,
        transforms=transforms,
    )

    response_name = get_aliased_name(model.response_component.response_term)
    covariates = get_covariates(covariates)

    if subplot_kwargs:
        for key, value in subplot_kwargs.items():
            setattr(covariates, key, value)

    if ax is None:
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        panels_n = len(np.unique(cap_data[covariates.panel])) if covariates.panel else 1
        rows, cols = default_grid(panels_n)
        fig, axes = create_axes_grid(panels_n, rows, cols, backend_kwargs=fig_kwargs)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if isinstance(axes[0], np.ndarray):
            fig = axes[0][0].get_figure()
        else:
            fig = axes[0].get_figure()

    if is_numeric_dtype(cap_data[covariates.main]):
        axes = plot_numeric(covariates, cap_data, transforms, legend, axes)
    elif is_categorical_dtype(cap_data[covariates.main]) or is_string_dtype(
        cap_data[covariates.main]
    ):
        axes = plot_categoric(covariates, cap_data, legend, axes)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    ylabel = response_name if target == "mean" else target
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=covariates.main, ylabel=ylabel)

    return fig, axes


def plot_comparison(
    model: Model,
    idata: az.InferenceData,
    contrast: Union[str, dict, list],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list] = None,
    comparison_type: str = "diff",
    use_hdi: bool = True,
    prob=None,
    legend: bool = True,
    transforms=None,
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
    average_by: str, list, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. Defaults to ``None``.
    comparison_type : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
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
        Keyword arguments used to determine the covariates used for the horizontal, group,
        and panel axes. For example, ``subplot_kwargs=dict(main="x", group="y", panel="z")`` would
        plot the horizontal axis as ``x``, the color (hue) as ``y``, and the panel axis as ``z``.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        A tuple with the figure and the axes.

    Raises
    ------
    ValueError
        If ``conditional`` and ``average_by`` are both ``None``.
        If length of ``conditional`` is greater than 3 and ``average_by`` is ``None``.

    Warning
        If length of ``contrast`` is greater than 2.
    """
    if conditional is None and average_by is None:
        raise ValueError("Must specify at least one of 'conditional' or 'average_by'.")
    if conditional is not None:
        if not isinstance(conditional, str):
            if len(conditional) > 3 and average_by is None:
                raise ValueError(
                    "Must specify a covariate to 'average_by' when number of covariates"
                    "passed to 'conditional' is greater than 3."
                )
    if average_by is True:
        raise ValueError(
            "Plotting when 'average_by = True' is not possible as 'True' marginalizes "
            "over all covariates resulting in a single comparison estimate. "
            "Please specify a covariate(s) to 'average_by'."
        )

    if isinstance(contrast, dict):
        contrast_name, contrast_level = next(iter(contrast.items()))
        if len(contrast_level) > 2:
            raise ValueError(
                f"Plotting when 'contrast' has > 2 values is not supported. "
                f"{contrast_name} has {len(contrast_level)} values."
            )

    contrast_df = comparisons(
        model=model,
        idata=idata,
        contrast=contrast,
        conditional=conditional,
        average_by=average_by,
        comparison_type=comparison_type,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
    )

    conditional_info = ConditionalInfo(model, conditional)

    if (subplot_kwargs and not average_by) or (subplot_kwargs and average_by):
        for key, value in subplot_kwargs.items():
            conditional_info.covariates.update({key: value})
        covariates = get_covariates(conditional_info.covariates)
    elif average_by and not subplot_kwargs:
        if not isinstance(average_by, list):
            average_by = listify(average_by)
        covariate_kinds = ("main", "group", "panel")
        average_by = dict(zip(covariate_kinds, average_by))
        covariates = get_covariates(average_by)
    else:
        covariates = get_covariates(conditional_info.covariates)

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
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=covariates.main, ylabel=response_name)

    return fig, axes
