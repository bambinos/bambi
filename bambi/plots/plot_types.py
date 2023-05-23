from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from bambi.plots.utils import identity, get_unique_levels, get_group_offset



# TO DO: instead of str: str, it should be bmb.response_distribution
# and Callable
pps_plot_kinds = {
    "Bernoulli": "bar",
    "Binomial": "bar",
    "Beta": "line",
    "Exponential": "line",
    "Gamma": "line",
    "Normal": "line",
    "NegativeBinomial": "bar",
    "Poisson": "bar",
    "StudentT": "line",
    "VonMises": "line",
    "InverseGaussian": "line",
    "Categorical": "bar"
}


def plot_numeric(
        covariates, 
        plot_data, 
        y_hat_mean, 
        y_hat_bounds, 
        transforms, 
        legend, 
        axes
):
    """Plotting of numeric data types.

    Parameters
    ----------
    covariates : dict
        A dictionary with the covariates to plot.
    plot_data : pd.DataFrame
        The data created by the `create_plot_data`, `create_comparisons_data`
        function.
    """
    main = covariates.get("horizontal")
    transform_main = transforms.get(main, identity)

    if len(covariates) == 1:
        ax = axes[0]
        values_main = transform_main(plot_data[main])
        ax.plot(values_main, y_hat_mean, solid_capstyle="butt")
        ax.fill_between(values_main, y_hat_bounds[0], y_hat_bounds[1], alpha=0.4)
    elif "color" in covariates and not "panel" in covariates:
        ax = axes[0]
        color = covariates.get("color")
        colors = get_unique_levels(plot_data[color])
        for i, clr in enumerate(colors):
            idx = (plot_data[color] == clr).to_numpy()
            values_main = transform_main(plot_data.loc[idx, main])
            ax.plot(
                values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt"
            )
            ax.fill_between(
                values_main,
                y_hat_bounds[0][idx],
                y_hat_bounds[1][idx],
                alpha=0.4,
                color=f"C{i}",
            )
    elif not "color" in covariates and "panel" in covariates:
        panel = covariates.get("panel")
        panels = get_unique_levels(plot_data[panel])
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (plot_data[panel] == pnl).to_numpy()
            values_main = transform_main(plot_data.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], solid_capstyle="butt")
            ax.fill_between(
                values_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx], alpha=0.4
            )
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in covariates and "panel" in covariates:
        color = covariates.get("color")
        panel = covariates.get("panel")
        colors = get_unique_levels(plot_data[color])
        panels = get_unique_levels(plot_data[panel])
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (plot_data[panel] == pnl).to_numpy()
                values_main = transform_main(plot_data.loc[idx, main])
                ax.plot(
                    values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt"
                )
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
                    ax.plot(
                        values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt"
                    )
                    ax.fill_between(
                        values_main,
                        y_hat_bounds[0][idx],
                        y_hat_bounds[1][idx],
                        alpha=0.4,
                        color=f"C{i}",
                    )
                    ax.set(title=f"{panel} = {pnl}")

    if "color" in covariates and legend:
        handles = [
            (
                Line2D([], [], color=f"C{i}", solid_capstyle="butt"),
                Patch(color=f"C{i}", alpha=0.4, lw=1),
            )
            for i in range(len(colors))
        ]
        for ax in axes.ravel():
            ax.legend(
                handles, 
                tuple(colors), 
                title=color, 
                handlelength=1.3, 
                handleheight=1, 
                loc="best"
            )
    return axes


def plot_categoric(covariates, plot_data, y_hat_mean, y_hat_bounds, legend, axes):
    """Plotting of categorical data types.

    Parameters
    ----------
    covariates : dict
        A dictionary with the covariates to plot.
    plot_data : pd.DataFrame
        The data created by the `create_plot_data`, `create_comparisons_data`
        function.
    
    """
    main = covariates.get("horizontal")
    main_levels = get_unique_levels(plot_data[main])
    main_levels_n = len(main_levels)
    idxs_main = np.arange(main_levels_n)

    if "color" in covariates:
        color = covariates.get("color")
        colors = get_unique_levels(plot_data[color])
        colors_n = len(colors)
        offset_bounds = get_group_offset(colors_n)
        colors_offset = np.linspace(-offset_bounds, offset_bounds, colors_n)

    if "panel" in covariates:
        panel = covariates.get("panel")
        panels = get_unique_levels(plot_data[panel])

    if len(covariates) == 1:
        ax = axes[0]
        ax.scatter(idxs_main, y_hat_mean)
        ax.vlines(idxs_main, y_hat_bounds[0], y_hat_bounds[1])
    elif "color" in covariates and not "panel" in covariates:
        ax = axes[0]
        for i, clr in enumerate(colors):
            idx = (plot_data[color] == clr).to_numpy()
            idxs = idxs_main + colors_offset[i]
            ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
            ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
    elif not "color" in covariates and "panel" in covariates:
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (plot_data[panel] == pnl).to_numpy()
            ax.scatter(idxs_main, y_hat_mean[idx])
            ax.vlines(idxs_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx])
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in covariates and "panel" in covariates:
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
