from typing import Union

import arviz as az
from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi.models import Model
from bambi.interpret.effects import comparisons, slopes, predictions
from bambi.interpret.plot_types import plot_categoric, plot_numeric
from bambi.interpret.utils import get_covariates, ConditionalInfo
from bambi.utils import get_aliased_name, listify


def _plot_differences(
    model: Model,
    conditional_info: ConditionalInfo,
    summary_df: pd.DataFrame,
    average_by: Union[str, list, None] = None,
    transforms=None,
    legend: bool = True,
    ax=None,
    fig_kwargs=None,
    subplot_kwargs=None,
):
    """
    Common function used for both 'plot_comparisons' and 'plot_slopes'.
    """
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
        panels_n = len(np.unique(summary_df[covariates.panel])) if covariates.panel else 1
        rows, cols = default_grid(panels_n)
        fig, axes = create_axes_grid(panels_n, rows, cols, backend_kwargs=fig_kwargs)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
        if isinstance(axes[0], np.ndarray):
            fig = axes[0][0].get_figure()
        else:
            fig = axes[0].get_figure()

    if is_numeric_dtype(summary_df[covariates.main]):
        # main condition variable can be numeric but at the same time only
        # a few values, so it is treated as categoric
        if np.unique(summary_df[covariates.main]).shape[0] <= 5:
            axes = plot_categoric(covariates, summary_df, legend, axes)
        else:
            axes = plot_numeric(covariates, summary_df, transforms, legend, axes)
    elif is_categorical_dtype(summary_df[covariates.main]) or is_string_dtype(
        summary_df[covariates.main]
    ):
        axes = plot_categoric(covariates, summary_df, legend, axes)
    else:
        raise TypeError("Main covariate must be numeric or categoric.")

    response_name = get_aliased_name(model.response_component.response_term)
    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=covariates.main, ylabel=response_name)

    return fig, axes


def plot_predictions(
    model: Model,
    idata: az.InferenceData,
    conditional: Union[str, list, dict, None] = None,
    average_by: Union[str, list, None] = None,
    target: str = "mean",
    sample_new_groups: bool = False,
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
    conditional : str, list, dict, optional
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.
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
        If ``conditional`` and ``average_by`` are both ``None``.
        If length of ``conditional`` is greater than 3 and ``average_by`` is ``None``.
        If main covariate is not numeric or categoric.
    """
    if conditional is None and average_by is None:
        raise ValueError("Must specify at least one of 'conditional' or 'average_by'.")

    if isinstance(conditional, dict):
        conditional = {
            key: np.array(sorted(listify(value))).flatten() for key, value in conditional.items()
        }
    elif isinstance(conditional, str):
        conditional = listify(conditional)

    if conditional is not None and len(conditional) > 3 and average_by is None:
        raise ValueError(
            "Must specify a covariate to 'average_by' when number of covariates "
            "passed to 'conditional' is greater than 3."
        )

    if average_by is True:
        raise ValueError(
            "Plotting when 'average_by = True' is not possible as 'True' marginalizes "
            "over all covariates resulting in a single prediction estimate. "
            "Please pass a covariate(s) to 'average_by'."
        )

    cap_data = predictions(
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

    conditional_info = ConditionalInfo(model, conditional)
    transforms = transforms if transforms is not None else {}

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

    response_name = get_aliased_name(model.response_component.response_term)

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


def plot_comparisons(
    model: Model,
    idata: az.InferenceData,
    contrast: Union[str, dict, list],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, None] = None,
    comparison_type: str = "diff",
    sample_new_groups: bool = False,
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
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. Defaults to ``None``.
    comparison_type : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.
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
        If the number of contrast levels is greater than 2 and ``average_by`` is ``None``.
        If ``conditional`` and ``average_by`` are both ``None``.
        If length of ``conditional`` is greater than 3 and ``average_by`` is ``None``.
        If ``average_by`` is ``True``.
        If main covariate is not numeric or categoric.
    """
    contrast_name = contrast
    if isinstance(contrast, dict):
        contrast_name, contrast_levels = next(iter(contrast.items()))
        if len(contrast_levels) > 2 and average_by is None:
            raise ValueError(
                "When plotting with more than 2 values for 'contrast', you must "
                "pass a covariate to 'average_by'. "
                f"{contrast_name} has {len(contrast_levels)} values."
            )

    if not isinstance(contrast, dict):
        if is_categorical_dtype(model.data[contrast_name]) or is_string_dtype(
            model.data[contrast_name]
        ):
            contrast_levels = len(model.data[contrast_name].unique())
            if contrast_levels > 2 and average_by is None:
                raise ValueError(
                    "When plotting with more than 2 values for 'contrast', you must "
                    f"pass a covariate to 'average_by'. {contrast_name} has "
                    f"{contrast_levels} values."
                )

    if conditional is None and average_by is None:
        raise ValueError("Must specify at least one of 'conditional' or 'average_by'.")

    if isinstance(conditional, dict):
        conditional = {key: sorted(listify(value)) for key, value in conditional.items()}
    elif conditional is not None:
        conditional = listify(conditional)
        if len(conditional) > 3 and average_by is None:
            raise ValueError(
                "Must specify a covariate to 'average_by' when number of covariates"
                "passed to 'conditional' is greater than 3."
            )

    if average_by is True:
        raise ValueError(
            "Plotting when 'average_by = True' is not possible as 'True' marginalizes "
            "over all covariates resulting in a single prediction estimate. "
            "Please pass a covariate(s) to 'average_by'."
        )

    conditional_info = ConditionalInfo(model, conditional)

    contrast_summary = comparisons(
        model=model,
        idata=idata,
        contrast=contrast,
        conditional=conditional,
        average_by=average_by,
        comparison_type=comparison_type,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
        sample_new_groups=sample_new_groups,
    )

    return _plot_differences(
        model=model,
        conditional_info=conditional_info,
        summary_df=contrast_summary,
        average_by=average_by,
        transforms=transforms,
        legend=legend,
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
    )


def plot_slopes(
    model: Model,
    idata: az.InferenceData,
    wrt: Union[str, dict],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list] = None,
    eps: float = 1e-4,
    slope: str = "dydx",
    sample_new_groups: bool = False,
    use_hdi: bool = True,
    prob=None,
    transforms=None,
    legend: bool = True,
    ax=None,
    fig_kwargs=None,
    subplot_kwargs=None,
):
    """Plot Conditional Adjusted Slopes

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    wrt : str, dict
        The slope of the regression with respect to (wrt) this predictor will be computed.
        If 'wrt' is numeric, the derivative is computed, else if string or categorical,
        'comparisons' is called to compute difference in group means.
    conditional : str, dict, list
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    eps : float, optional
        To compute the slope, 'wrt' is evaluated at wrt +/- 'eps'. The rate of change is then
        computed as the difference between the two values divided by 'eps'. Defaults to 1e-4.
    slope: str, optional
        The type of slope to compute. Defaults to 'dydx'.
        'dydx' represents a unit increase in 'wrt' is associated with an n-unit change in
        the response.
        'eyex' represents a percentage increase in 'wrt' is associated with an n-percent
        change in the response.
        'eydx' represents a unit increase in 'wrt' is associated with an n-percent
        change in the response.
        'dyex' represents a percent change in 'wrt' is associated with a unit increase
        in the response.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
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
        If the number of ``wrt`` values is greater than 2 and ``average_by`` is ``None``.
        If ``conditional`` and ``average_by`` are both ``None``.
        If length of ``conditional`` is greater than 3 and ``average_by`` is ``None``.
        If ``average_by`` is ``True``.
        If ``slope`` is not one of ('dydx', 'dyex', 'eyex', 'eydx').
        If main covariate is not numeric or categoric.
    """
    wrt_name = wrt
    if isinstance(wrt, dict):
        wrt_name, wrt_value = next(iter(wrt.items()))

        if not isinstance(wrt_value, (list, np.ndarray)):
            wrt_value = [wrt_value]

        if len(wrt_value) > 2 and average_by is None:
            raise ValueError(
                "When plotting with more than 2 values for 'wrt', you must "
                "pass a covariate to 'average_by'"
            )

    if not isinstance(wrt, dict):
        if is_categorical_dtype(model.data[wrt_name]) or is_string_dtype(model.data[wrt_name]):
            num_values = len(model.data[wrt_name].unique())
            if num_values > 2 and average_by is None:
                raise ValueError(
                    "When plotting with more than 2 values for 'wrt', you must "
                    f"pass a covariate to 'average_by'. {wrt_name} has {num_values} values."
                )

    if conditional is None and average_by is None:
        raise ValueError("Must specify at least one of 'conditional' or 'average_by'.")

    if isinstance(conditional, dict):
        conditional = {key: sorted(listify(value)) for key, value in conditional.items()}
    elif conditional is not None:
        conditional = listify(conditional)
        if len(conditional) > 3 and average_by is None:
            raise ValueError(
                "Must specify a covariate to 'average_by' when number of covariates"
                "passed to 'conditional' is greater than 3."
            )

    if average_by is True:
        raise ValueError(
            "Plotting when 'average_by = True' is not possible as 'True' marginalizes "
            "over all covariates resulting in a single prediction estimate. "
            "Please pass a covariate(s) to 'average_by'."
        )

    if slope not in ("dydx", "dyex", "eyex", "eydx"):
        raise ValueError("'slope' must be one of ('dydx', 'dyex', 'eyex', 'eydx')")

    conditional_info = ConditionalInfo(model, conditional)

    slopes_summary = slopes(
        model=model,
        idata=idata,
        wrt=wrt,
        conditional=conditional,
        average_by=average_by,
        eps=eps,
        slope=slope,
        use_hdi=use_hdi,
        prob=prob,
        transforms=transforms,
        sample_new_groups=sample_new_groups,
    )

    return _plot_differences(
        model=model,
        conditional_info=conditional_info,
        summary_df=slopes_summary,
        average_by=average_by,
        transforms=transforms,
        legend=legend,
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
    )
