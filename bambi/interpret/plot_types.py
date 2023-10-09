from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from bambi.interpret.utils import Covariates, get_group_offset, identity


def plot_numeric(
    covariates: Covariates,
    plot_data: pd.DataFrame,
    transforms: dict,
    legend: bool = True,
    axes=None,
):
    """Plotting of numeric data types.

    Parameters
    ----------
    covariates : Covariates
        Covariates callable with attributes main, group, panel.
    plot_data : pd.DataFrame
        The data created by the `create_cap_data` or `create_comparisons_data`
        function.
    transforms : dict
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to `None`.
    legend : bool, optional
        Whether to include a legend in the plot. Default to `True`.
    axes : np.ndarray, optional
        Array of axes. Defaults to `None`.

    Returns
    -------
    axes : np.ndarray
        Array of axes.
    """

    main, color, panel = covariates.main, covariates.group, covariates.panel
    covariates = {k: v for k, v in vars(covariates).items() if v is not None}
    transform_main = transforms.get(main, identity)
    y_hat_mean = plot_data["estimate"]
    y_hat_bounds = np.transpose(plot_data[plot_data.columns[-2:]].values)

    # if "estimate_dim" column exists, then model predictions has multiple dimensions
    if "estimate_dim" in plot_data.columns:
        y_hat_dims = plot_data["estimate_dim"].unique()
        y_hat_ndim = len(y_hat_dims)
    else:
        y_hat_ndim = 1

    if len(covariates) == 1:
        ax = axes[0]
        if y_hat_ndim > 1:
            for i, clr in enumerate(y_hat_dims):
                idx = plot_data["estimate_dim"] == clr
                values_main = transform_main(plot_data.loc[idx, main])
                ax.plot(
                    values_main, y_hat_mean[idx], color=f"C{i}", label=clr, solid_capstyle="butt"
                )
                ax.fill_between(
                    values_main,
                    y_hat_bounds[0][idx],
                    y_hat_bounds[1][idx],
                    alpha=0.4,
                    color=f"C{i}",
                )
        else:
            values_main = transform_main(plot_data[main])
            ax.plot(values_main, y_hat_mean, solid_capstyle="butt", color="C0")
            ax.fill_between(values_main, y_hat_bounds[0], y_hat_bounds[1], alpha=0.4)
    elif "group" in covariates and not "panel" in covariates:
        ax = axes[0]
        colors = np.unique(plot_data[color])
        for i, clr in enumerate(colors):
            idx = (plot_data[color] == clr).to_numpy()
            values_main = transform_main(plot_data.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
            ax.fill_between(
                values_main,
                y_hat_bounds[0][idx],
                y_hat_bounds[1][idx],
                alpha=0.4,
                color=f"C{i}",
            )
    elif not "group" in covariates and "panel" in covariates:
        panels = np.unique(plot_data[panel])
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (plot_data[panel] == pnl).to_numpy()
            values_main = transform_main(plot_data.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], solid_capstyle="butt")
            ax.fill_between(values_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx], alpha=0.4)
            ax.set(title=f"{panel} = {pnl}")
    elif "group" in covariates and "panel" in covariates:
        colors = np.unique(plot_data[color])
        panels = np.unique(plot_data[panel])
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (plot_data[panel] == pnl).to_numpy()
                values_main = transform_main(plot_data.loc[idx, main])
                ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
                ax.fill_between(
                    values_main,
                    y_hat_bounds[0][idx],
                    y_hat_bounds[1][idx],
                    alpha=0.4,
                    color=f"C{i}",
                )
                ax.set(title=f"{panel} = {pnl}")
        else:
            for ax, pnl in zip(axes.ravel(), panels):
                for i, clr in enumerate(colors):
                    idx = ((plot_data[panel] == pnl) & (plot_data[color] == clr)).to_numpy()
                    values_main = transform_main(plot_data.loc[idx, main])
                    ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
                    ax.fill_between(
                        values_main,
                        y_hat_bounds[0][idx],
                        y_hat_bounds[1][idx],
                        alpha=0.4,
                        color=f"C{i}",
                    )
                    ax.set(title=f"{panel} = {pnl}")

    if y_hat_ndim > 1:
        if "group" not in covariates and legend:
            handles = [
                (
                    Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                    Patch(color=f"C{i}", alpha=0.4, lw=1),
                )
                for i in range(len(y_hat_dims))
            ]
            for ax in axes.ravel():
                ax.legend(
                    handles,
                    tuple(y_hat_dims),
                    title="estimate_dim",
                    handlelength=1.3,
                    handleheight=1,
                    loc="best",
                )

    if "group" in covariates and legend:
        handles = [
            (
                Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                Patch(color=f"C{i}", alpha=0.4, lw=1),
            )
            for i in range(len(colors))
        ]
        for ax in axes.ravel():
            ax.legend(
                handles, tuple(colors), title=color, handlelength=1.3, handleheight=1, loc="best"
            )

    return axes


def plot_categoric(covariates: Covariates, plot_data: pd.DataFrame, legend: bool = True, axes=None):
    """Plotting of categorical data types.

    Parameters
    ----------
    covariates : Covariates
        Covariates callable with attributes main, gro up, panel.
    plot_data : pd.DataFrame
        The data created by the `create_cap_data` or `create_comparisons_data`
        function.
    legend : bool, optional
        Whether to include a legend in the plot. Default to `True`.
    axes : np.ndarray, optional
        Array of axes. Defaults to `None`.

    Returns
    -------
    axes : np.ndarray
        Array of axes.
    """

    main, color, panel = covariates.main, covariates.group, covariates.panel
    covariates = {k: v for k, v in vars(covariates).items() if v is not None}
    main_levels = np.unique(plot_data[main])
    main_levels_n = len(main_levels)
    idxs_main = np.arange(main_levels_n)
    y_hat_mean = plot_data["estimate"]
    y_hat_bounds = np.transpose(plot_data[plot_data.columns[-2:]].values)

    # if "estimate_dim" column exists, then model predictions has multiple dimensions
    if "estimate_dim" in plot_data.columns:
        y_hat_dims = plot_data["estimate_dim"].unique()
        y_hat_ndim = len(y_hat_dims)
    else:
        y_hat_ndim = 1

    if "group" in covariates:
        colors = np.unique(plot_data[color])
        colors_n = len(colors)
        offset_bounds = get_group_offset(colors_n)
        colors_offset = np.linspace(-offset_bounds, offset_bounds, colors_n)

    if "panel" in covariates:
        panels = np.unique(plot_data[panel])

    if len(covariates) == 1:
        ax = axes[0]
        if y_hat_ndim > 1:
            offset_bounds = get_group_offset(y_hat_ndim)
            colors_offset = np.linspace(-offset_bounds, offset_bounds, y_hat_ndim)
            for i, clr in enumerate(y_hat_dims):
                idx = plot_data["estimate_dim"] == clr
                idxs = idxs_main + colors_offset[i]
                ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
        else:
            ax.scatter(idxs_main, y_hat_mean, color="C0")
            ax.vlines(idxs_main, y_hat_bounds[0], y_hat_bounds[1], color="C0")
    elif "group" in covariates and not "panel" in covariates:
        ax = axes[0]
        for i, clr in enumerate(colors):
            idx = (plot_data[color] == clr).to_numpy()
            idxs = idxs_main + colors_offset[i]
            ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
            ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
    elif not "group" in covariates and "panel" in covariates:
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (plot_data[panel] == pnl).to_numpy()
            ax.scatter(idxs_main, y_hat_mean[idx])
            ax.vlines(idxs_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx])
            ax.set(title=f"{panel} = {pnl}")
    elif "group" in covariates and "panel" in covariates:
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (plot_data[panel] == pnl).to_numpy()
                idxs = idxs_main + colors_offset[i]
                ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                ax.set(title=f"{panel} = {pnl}")
        else:
            for ax, pnl in zip(axes.ravel(), panels):
                for i, clr in enumerate(colors):
                    idx = ((plot_data[panel] == pnl) & (plot_data[color] == clr)).to_numpy()
                    idxs = idxs_main + colors_offset[i]
                    ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                    ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                    ax.set(title=f"{panel} = {pnl}")

    if y_hat_ndim > 1:
        if "group" not in covariates and legend:
            handles = [
                (
                    Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                    Patch(color=f"C{i}", alpha=0.4, lw=1),
                )
                for i in range(len(y_hat_dims))
            ]
            for ax in axes.ravel():
                ax.legend(
                    handles,
                    tuple(y_hat_dims),
                    title="estimate_dim",
                    handlelength=1.3,
                    handleheight=1,
                    loc="best",
                )

    if "group" in covariates and legend:
        handles = [
            Line2D([], [], c=f"C{i}", marker="o", label=level) for i, level in enumerate(colors)
        ]
        for ax in axes.ravel():
            ax.legend(handles=handles, title=color, loc="best")

    for ax in axes.ravel():
        ax.set_xticks(idxs_main)
        ax.set_xticklabels(main_levels)

    return axes
