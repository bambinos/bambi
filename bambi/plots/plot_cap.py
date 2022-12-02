# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from statistics import mode

import arviz as az
import numpy as np
import pandas as pd

from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
from formulae.terms.call import Call
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

    main = covariates.get("horizontal")
    group = covariates.get("color", None)
    panel = covariates.get("panel", None)

    # Obtain data for main variable
    main_values = make_main_values(data[main], grid_n)
    main_n = len(main_values)

    # If available, obtain groups for grouping variable
    if group:
        group_values = make_group_values(data[group], groups_n)
        group_n = len(group_values)

    # If available, obtain groups for panel variable. Same logic than grouping applies
    if panel:
        panel_values = make_group_values(data[panel], groups_n)
        panel_n = len(panel_values)

    data_dict = {main: main_values}

    if group and not panel:
        main_values = np.tile(main_values, group_n)
        group_values = np.repeat(group_values, main_n)
        data_dict.update({main: main_values, group: group_values})
    elif not group and panel:
        main_values = np.tile(main_values, panel_n)
        panel_values = np.repeat(panel_values, main_n)
        data_dict.update({main: main_values, panel: panel_values})
    elif group and panel:
        if group == panel:
            main_values = np.tile(main_values, group_n)
            group_values = np.repeat(group_values, main_n)
            data_dict.update({main: main_values, group: group_values})
        else:
            main_values = np.tile(np.tile(main_values, group_n), panel_n)
            group_values = np.tile(np.repeat(group_values, main_n), panel_n)
            panel_values = np.repeat(panel_values, main_n * group_n)
            data_dict.update({main: main_values, group: group_values, panel: panel_values})

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
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]

                for name in names:
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


def plot_cap(
    model,
    idata,
    covariates,
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

    cap_data = create_cap_data(model, covariates)
    idata = model.predict(idata, data=cap_data, inplace=False)

    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]

    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")

    if transforms is None:
        transforms = {}

    response_transform = transforms.get(model.response.name, identity)

    y_hat = response_transform(idata.posterior[f"{model.response.name}_mean"])
    y_hat_mean = y_hat.mean(("chain", "draw"))

    if use_hdi:
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{model.response.name}_mean"].T
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
        fig = axes[0].get_figure()

    main = covariates.get("horizontal")
    if is_numeric_dtype(cap_data[main]):
        axes = _plot_cap_numeric(
            covariates, cap_data, y_hat_mean, y_hat_bounds, transforms, legend, axes
        )
    elif is_categorical_dtype(cap_data[main]) or is_string_dtype(cap_data[main]):
        axes = _plot_cap_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, axes)
    else:
        raise ValueError("Main covariate must be numeric or categoric.")

    for ax in axes.ravel():  # pylint: disable = redefined-argument-from-local
        ax.set(xlabel=main, ylabel=model.response.name)

    return fig, axes


def _plot_cap_numeric(covariates, cap_data, y_hat_mean, y_hat_bounds, transforms, legend, axes):
    main = covariates.get("horizontal")
    transform_main = transforms.get(main, identity)

    if len(covariates) == 1:
        ax = axes[0]
        values_main = transform_main(cap_data[main])
        ax.plot(values_main, y_hat_mean, solid_capstyle="butt")
        ax.fill_between(values_main, y_hat_bounds[0], y_hat_bounds[1], alpha=0.5)
    elif "color" in covariates and not "panel" in covariates:
        ax = axes[0]
        color = covariates.get("color")
        colors = get_unique_levels(cap_data[color])
        for i, clr in enumerate(colors):
            idx = (cap_data[color] == clr).to_numpy()
            values_main = transform_main(cap_data.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
            ax.fill_between(
                values_main,
                y_hat_bounds[0][idx],
                y_hat_bounds[1][idx],
                alpha=0.5,
                color=f"C{i}",
            )
    elif not "color" in covariates and "panel" in covariates:
        panel = covariates.get("panel")
        panels = get_unique_levels(cap_data[panel])
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (cap_data[panel] == pnl).to_numpy()
            values_main = transform_main(cap_data.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], solid_capstyle="butt")
            ax.fill_between(values_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx], alpha=0.5)
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in covariates and "panel" in covariates:
        color = covariates.get("color")
        panel = covariates.get("panel")
        colors = get_unique_levels(cap_data[color])
        panels = get_unique_levels(cap_data[panel])
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (cap_data[panel] == pnl).to_numpy()
                values_main = transform_main(cap_data.loc[idx, main])
                ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
                ax.fill_between(
                    values_main,
                    y_hat_bounds[0][idx],
                    y_hat_bounds[1][idx],
                    alpha=0.5,
                    color=f"C{i}",
                )
                ax.set(title=f"{panel} = {pnl}")
        else:
            for ax, pnl in zip(axes.ravel(), panels):
                for i, clr in enumerate(colors):
                    idx = ((cap_data[panel] == pnl) & (cap_data[color] == clr)).to_numpy()
                    values_main = transform_main(cap_data.loc[idx, main])
                    ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
                    ax.fill_between(
                        values_main,
                        y_hat_bounds[0][idx],
                        y_hat_bounds[1][idx],
                        alpha=0.5,
                        color=f"C{i}",
                    )
                    ax.set(title=f"{panel} = {pnl}")

    if "color" in covariates and legend:
        handles = [
            (
                Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                Patch(color=f"C{i}", alpha=0.5, lw=1),
            )
            for i in range(len(colors))
        ]
        for ax in axes.ravel():
            ax.legend(
                handles, tuple(colors), title=color, handlelength=1.3, handleheight=1, loc="best"
            )
    return axes


def _plot_cap_categoric(covariates, cap_data, y_hat_mean, y_hat_bounds, legend, axes):
    main = covariates.get("horizontal")
    main_levels = get_unique_levels(cap_data[main])
    main_levels_n = len(main_levels)
    idxs_main = np.arange(main_levels_n)

    if "color" in covariates:
        color = covariates.get("color")
        colors = get_unique_levels(cap_data[color])
        colors_n = len(colors)
        offset_bounds = get_group_offset(colors_n)
        colors_offset = np.linspace(-offset_bounds, offset_bounds, colors_n)

    if "panel" in covariates:
        panel = covariates.get("panel")
        panels = get_unique_levels(cap_data[panel])

    if len(covariates) == 1:
        ax = axes[0]
        ax.scatter(idxs_main, y_hat_mean)
        ax.vlines(idxs_main, y_hat_bounds[0], y_hat_bounds[1])
    elif "color" in covariates and not "panel" in covariates:
        ax = axes[0]
        for i, clr in enumerate(colors):
            idx = (cap_data[color] == clr).to_numpy()
            idxs = idxs_main + colors_offset[i]
            ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
            ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
    elif not "color" in covariates and "panel" in covariates:
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (cap_data[panel] == pnl).to_numpy()
            ax.scatter(idxs_main, y_hat_mean[idx])
            ax.vlines(idxs_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx])
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in covariates and "panel" in covariates:
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (cap_data[panel] == pnl).to_numpy()
                idxs = idxs_main + colors_offset[i]
                ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                ax.set(title=f"{panel} = {pnl}")
        else:
            for ax, pnl in zip(axes.ravel(), panels):
                for i, clr in enumerate(colors):
                    idx = ((cap_data[panel] == pnl) & (cap_data[color] == clr)).to_numpy()
                    idxs = idxs_main + colors_offset[i]
                    ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                    ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                    ax.set(title=f"{panel} = {pnl}")

    if "color" in covariates and legend:
        handles = [
            Line2D([], [], c=f"C{i}", marker="o", label=level) for i, level in enumerate(colors)
        ]
        for ax in axes.ravel():
            ax.legend(handles=handles, title=color, loc="best")

    for ax in axes.ravel():
        ax.set_xticks(idxs_main)
        ax.set_xticklabels(main_levels)

    return axes


def identity(x):
    return x


def make_main_values(x, grid_n):
    if is_numeric_dtype(x):
        return np.linspace(np.min(x), np.max(x), grid_n)
    elif is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    raise ValueError("Main covariate must be numeric or categoric.")


def make_group_values(x, groups_n):
    if is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    elif is_numeric_dtype(x):
        return np.quantile(x, np.linspace(0, 1, groups_n))
    raise ValueError("Group covariate must be numeric or categoric.")
