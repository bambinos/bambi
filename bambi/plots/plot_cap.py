# pylint: disable = protected-access
# pylint: disable = too-many-function-args
from statistics import mode

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi.utils import listify
from bambi.plots.utils import get_group_offset, get_unique_levels


def create_cap_data(model, covariates, grid_n=200, groups_n=5):
    """Create data for a Conditional Adjusted Predictions plot

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    covariates : list
        A sequence of one or two names of variables. The first variable is taken as the main
        variable. If present, the second variable is a grouping variable.
    grid_n : int, optional
        The number of points used to evaluate the main covariate. Defaults to 200.
    groups_n : int, optional
        The number of groups to create when the grouping variable is numeric. Groups are based on
        equally spaced points. Defaults to 5.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Predictions plot.

    Raises
    ------
    ValueError
        When the number of covariates is larger than 2.
        When either the main or the group covariates are not numeric or categoric.
    """
    data = model.data
    covariates = listify(covariates)

    if len(covariates) not in [1, 2]:
        raise ValueError(f"The number of covariates must be 1 or 2. It's {len(covariates)}.")

    main = covariates[0]

    # If available, take the name of the grouping variable
    if len(covariates) == 1:
        group = None
    else:
        group = covariates[1]

    # Obtain data for main variable
    data_main = data[main]
    if is_numeric_dtype(data_main):
        main_values = np.linspace(np.min(data_main), np.max(data_main), grid_n)
    elif is_string_dtype(data_main) or is_categorical_dtype(data_main):
        main_values = np.unique(data_main)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    # If available, obtain groups for grouping variable
    if group:
        group_data = data[group]
        if is_string_dtype(group_data) or is_categorical_dtype(group_data):
            group_values = np.unique(group_data)
        elif is_numeric_dtype(group_data):
            group_values = np.quantile(group_data, np.linspace(0, 1, groups_n))
        else:
            raise ValueError("Group covariate must be numeric or categoric.")

        # Reshape accordingly
        group_n = len(group_values)
        main_n = len(main_values)
        main_values = np.tile(main_values, group_n)
        group_values = np.repeat(group_values, main_n)
        data_dict = {main: main_values, group: group_values}
    else:
        data_dict = {main: main_values}

    # Construct dictionary of terms that are in the model
    terms = {}
    if model._design.common:
        terms.update(model._design.common.terms)

    if model._design.group:
        terms.update(model._design.group.terms)

    # Get default values for each variable in the model
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                name = component.name
                if name not in data_dict:
                    # For numeric predictors, select the mean.
                    if component.kind == "numeric":
                        data_dict[name] = np.mean(data[name])
                    # For categoric predictors, select the most frequent level.
                    elif component.kind == "categoric":
                        data_dict[name] = mode(data[name])

    cap_data = pd.DataFrame(data_dict)

    # Make sure new types are same types than the original columns
    for column in cap_data:
        cap_data[column] = cap_data[column].astype(data[column].dtype)
    return cap_data


def plot_cap(model, idata, covariates, use_hdi=True, hdi_prob=None, legend=True, ax=None):
    """Plot Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    covariates : list
        A sequence of one or two names of variables. The first variable is taken as the main
        variable. If present, the second variable is a grouping variable.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    hdi_prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A matplotlib axes object. If None, this function instantiates a new axes object.
        Defaults to ``None``.

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

    covariates = listify(covariates)
    assert len(covariates) in [1, 2]

    cap_data = create_cap_data(model, covariates)
    idata = model.predict(idata, data=cap_data, inplace=False)

    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]

    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")

    y_hat = idata.posterior[f"{model.response.name}_mean"]
    y_hat_mean = y_hat.mean(("chain", "draw"))

    if use_hdi:
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{model.response.name}_mean"].T
    else:
        lower_bound = round((1 - hdi_prob) / 2, 4)
        upper_bound = 1 - lower_bound
        y_hat_bounds = y_hat.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    main = covariates[0]
    if is_numeric_dtype(cap_data[main]):
        ax = _plot_cap_numeric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, ax)
    elif is_categorical_dtype(cap_data[main]) or is_string_dtype(cap_data[main]):
        ax = _plot_cap_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, ax)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    ax.set(xlabel=main, ylabel=model.response.name)
    return fig, ax


def _plot_cap_numeric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, ax):
    if len(covariates) == 1:
        main = covariates[0]
        ax.plot(cap_data[main], y_hat_mean, solid_capstyle="butt")
        ax.fill_between(cap_data[main], y_hat_bounds[0], y_hat_bounds[1], alpha=0.5)
    else:
        main, group = covariates
        groups = get_unique_levels(cap_data[group])

        for i, grp in enumerate(groups):
            idx = (cap_data[group] == grp).values
            ax.plot(cap_data.loc[idx, main], y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
            ax.fill_between(
                cap_data.loc[idx, main],
                y_hat_bounds[0][idx],
                y_hat_bounds[1][idx],
                alpha=0.3,
                color=f"C{i}",
            )

        if legend:
            handles = [
                (
                    Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                    Patch(color=f"C{i}", alpha=0.3, lw=1),
                )
                for i in range(len(groups))
            ]
            ax.legend(
                handles,
                tuple(groups),
                title=group,
                handlelength=1.3,
                handleheight=1,
                bbox_to_anchor=(1.03, 0.5),
                loc="center left",
            )

    return ax


def _plot_cap_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, ax):
    main = covariates[0]
    main_levels = get_unique_levels(cap_data[main])
    main_levels_n = len(main_levels)
    idxs_main = np.arange(main_levels_n)

    if len(covariates) == 1:
        ax.scatter(idxs_main, y_hat_mean)
        ax.vlines(idxs_main, y_hat_bounds[0], y_hat_bounds[1])
    else:
        group = covariates[1]
        group_levels = get_unique_levels(cap_data[group])
        group_levels_n = len(group_levels)
        offset_bounds = get_group_offset(group_levels_n)
        offset_groups = np.linspace(-offset_bounds, offset_bounds, group_levels_n)

        for i, grp in enumerate(group_levels):
            idx = (cap_data[group] == grp).values
            idxs = idxs_main + offset_groups[i]
            ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
            ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")

        if legend:
            handles = [
                Line2D([], [], c=f"C{i}", marker="o", label=level)
                for i, level in enumerate(group_levels)
            ]
            ax.legend(handles=handles, title=group, bbox_to_anchor=(1.03, 0.5), loc="center left")

    ax.set_xticks(idxs_main)
    ax.set_xticklabels(main_levels)
    return ax
