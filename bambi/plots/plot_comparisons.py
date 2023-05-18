# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from statistics import mode
from typing import Union

import arviz as az
import numpy as np
import pandas as pd

from arviz.plots.backends.matplotlib import create_axes_grid
from arviz.plots.plot_utils import default_grid
from formulae.terms.call import Call
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi.utils import listify, get_aliased_name, clean_formula_lhs
from bambi.plots.utils import get_group_offset, get_unique_levels
from bambi.plots.utils import CreateData


def plot_comparison(
        model,
        idata,
        contrast_predictor: Union[str, dict, list],
        conditional: Union[str, dict, list],
        target="mean",
        use_hdi=True,
        hdi_prob=None,
        transforms=None,
        legend=True,
        ax=None,
        fig_kwargs=None
):
    """
    TO DO: create parent class to inherit args. from b/c they are common
    to all plot functions.
    # """
    
    comparisons_df, contrast_df = comparison(
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
        conditional_values = conditional
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))

    # contrast_name, contrast = next(iter(contrast_predictor.items()))
    
    # print(f"orig. conditional: {conditional_values}")
    # print(f"conditional: {conditional}")
    # #print(f"contrast_name: {contrast_name}, contrast: {contrast}")
    # print(f"contrast_predictor: {contrast_predictor}")

    # create = CreateData(model, conditional)
    # comparisons_df = create.comparisons_data(
    #     contrast_predictor=contrast_predictor,
    #     conditional=conditional_values
    #     )
    
    # if hdi_prob is None:
    #     hdi_prob = az.rcParams["stats.hdi_prob"]
    
    # if not 0 < hdi_prob < 1:
    #     raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")
    
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response_transform = transforms.get(response_name, identity)
    response_preds_term = f"{response_name}_{target}_preds"

    # idata = model.predict(idata, data=comparisons_df, inplace=False)
    # y_hat = response_transform(idata.posterior[f"{response_name}_{target}"])
    # y_hat_mean = y_hat.mean(("chain", "draw"))
    # comparisons_df[response_preds_term] = y_hat_mean

    # if use_hdi:
    #      y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{response_name}_{target}"].T

    # # rename using 95% HDI
    # lower = f"{response_preds_term}_lower"
    # upper = f"{response_preds_term}_upper"
    # comparisons_df[lower] = y_hat_bounds[0]
    # comparisons_df[upper] = y_hat_bounds[1]


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
        # axes = _plot_cap_numeric(
        #     conditional, 
        #     contrast_df,
        #     contrast_df["contrast_comparison"],
        #     y_hat_bounds,
        #     transforms,
        #     legend,
        #     axes
        # )
        axes = _plot_comparison_categoric(
            conditional, 
            contrast_df,
            contrast_df["contrast_comparison"],
            y_hat_bounds,
            legend,
            axes
        )
    elif is_categorical_dtype(contrast_df[main]) or is_string_dtype(contrast_df[main]):
        axes = _plot_comparison_categoric(
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

    return (fig, axes), contrast_df


def _plot_cap_numeric(
        conditional, 
        contrast_comparison, 
        y_hat_mean, 
        y_hat_bounds, 
        transforms, 
        legend, 
        axes
    ):
    main = conditional.get("horizontal")
    transform_main = transforms.get(main, identity)

    if len(conditional) == 1:
        ax = axes[0]
        values_main = transform_main(contrast_comparison[main])
        ax.plot(values_main, y_hat_mean, solid_capstyle="butt")
        ax.fill_between(values_main, y_hat_bounds[0], y_hat_bounds[1], alpha=0.4)
    elif "color" in conditional and not "panel" in conditional:
        ax = axes[0]
        color = conditional.get("color")
        colors = get_unique_levels(contrast_comparison[color])
        for i, clr in enumerate(colors):
            idx = (contrast_comparison[color] == clr).to_numpy()
            values_main = transform_main(contrast_comparison.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
            ax.fill_between(
                values_main,
                y_hat_bounds[0][idx],
                y_hat_bounds[1][idx],
                alpha=0.4,
                color=f"C{i}",
            )
    elif not "color" in conditional and "panel" in conditional:
        panel = conditional.get("panel")
        panels = get_unique_levels(contrast_comparison[panel])
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (contrast_comparison[panel] == pnl).to_numpy()
            values_main = transform_main(contrast_comparison.loc[idx, main])
            ax.plot(values_main, y_hat_mean[idx], solid_capstyle="butt")
            ax.fill_between(values_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx], alpha=0.4)
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in conditional and "panel" in conditional:
        color = conditional.get("color")
        panel = conditional.get("panel")
        colors = get_unique_levels(contrast_comparison[color])
        panels = get_unique_levels(contrast_comparison[panel])
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (contrast_comparison[panel] == pnl).to_numpy()
                values_main = transform_main(contrast_comparison.loc[idx, main])
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
                    idx = ((contrast_comparison[panel] == pnl) & (contrast_comparison[color] == clr)).to_numpy()
                    values_main = transform_main(contrast_comparison.loc[idx, main])
                    ax.plot(values_main, y_hat_mean[idx], color=f"C{i}", solid_capstyle="butt")
                    ax.fill_between(
                        values_main,
                        y_hat_bounds[0][idx],
                        y_hat_bounds[1][idx],
                        alpha=0.4,
                        color=f"C{i}",
                    )
                    ax.set(title=f"{panel} = {pnl}")

    if "color" in conditional and legend:
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



def _plot_comparison_categoric(
        conditional, 
        contrast_comparison,
        y_hat_mean, 
        y_hat_bounds,
        legend, 
        axes
    ):
    
    main = conditional.get("horizontal")
    main_levels = get_unique_levels(contrast_comparison[main])
    main_levels_n = len(main_levels)
    idxs_main = np.arange(main_levels_n)

    if "color" in conditional:
        color = conditional.get("color")
        colors = get_unique_levels(contrast_comparison[color])
        colors_n = len(colors)
        offset_bounds = get_group_offset(colors_n)
        colors_offset = np.linspace(-offset_bounds, offset_bounds, colors_n)

    if "panel" in conditional:
        panel = conditional.get("panel")
        panels = get_unique_levels(contrast_comparison[panel])

    if len(conditional) == 1:
        ax = axes[0]
        ax.scatter(idxs_main, y_hat_mean)
        ax.vlines(idxs_main, y_hat_bounds[0], y_hat_bounds[1])
    elif "color" in conditional and not "panel" in conditional:
        ax = axes[0]
        for i, clr in enumerate(colors):
            idx = (contrast_comparison[color] == clr).to_numpy()
            idxs = idxs_main + colors_offset[i]
            ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
            ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
    elif not "color" in conditional and "panel" in conditional:
        for ax, pnl in zip(axes.ravel(), panels):
            idx = (contrast_comparison[panel] == pnl).to_numpy()
            ax.scatter(idxs_main, y_hat_mean[idx])
            ax.vlines(idxs_main, y_hat_bounds[0][idx], y_hat_bounds[1][idx])
            ax.set(title=f"{panel} = {pnl}")
    elif "color" in conditional and "panel" in conditional:
        if color == panel:
            for i, (ax, pnl) in enumerate(zip(axes.ravel(), panels)):
                idx = (conditional[panel] == pnl).to_numpy()
                idxs = idxs_main + colors_offset[i]
                ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                ax.set(title=f"{panel} = {pnl}")
        else:
            for ax, pnl in zip(axes.ravel(), panels):
                for i, clr in enumerate(colors):
                    idx = ((conditional[panel] == pnl) & (conditional[color] == clr)).to_numpy()
                    idxs = idxs_main + colors_offset[i]
                    ax.scatter(idxs, y_hat_mean[idx], color=f"C{i}")
                    ax.vlines(idxs, y_hat_bounds[0][idx], y_hat_bounds[1][idx], color=f"C{i}")
                    ax.set(title=f"{panel} = {pnl}")

    if "color" in conditional and legend:
        handles = [
            Line2D([], [], c=f"C{i}", marker="o", label=level) for i, level in enumerate(colors)
        ]
        for ax in axes.ravel():
            ax.legend(handles=handles, title=color, loc="best")

    for ax in axes.ravel():
        ax.set_xticks(idxs_main)
        ax.set_xticklabels(main_levels)
    
    return axes



def comparison(
        model,
        idata,
        contrast_predictor: Union[str, dict, list],
        conditional: Union[str, dict, list],
        target="mean",
        use_hdi=True,
        hdi_prob=None,
        transforms=None,
    ) -> pd.DataFrame:
    """

    """
    create = CreateData(model, conditional)

    covariate_kinds = ("horizontal", "color", "panel")
    # if not dict, then user did not pass values to condition on
    if not isinstance(conditional, dict):
        conditional = listify(conditional)
        conditional = dict(zip(covariate_kinds, conditional))
        comparisons_df = create.comparisons_data(
            contrast_predictor=contrast_predictor,
            conditional=conditional,
            user_passed=False
        )
    # if dict, user passed values to condition on
    elif isinstance(conditional, dict):
        comparisons_df = create.comparisons_data(
            contrast_predictor=contrast_predictor,
            conditional=conditional,
            user_passed=True
        )
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))
    
    contrast_name, contrast = next(iter(contrast_predictor.items()))
    
    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]
    
    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")
    
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response_transform = transforms.get(response_name, identity)
    response_preds_term = f"{response_name}_{target}_preds"

    idata = model.predict(idata, data=comparisons_df, inplace=False)
    y_hat = response_transform(idata.posterior[f"{response_name}_{target}"])
    y_hat_mean = y_hat.mean(("chain", "draw"))
    comparisons_df[response_preds_term] = y_hat_mean

    if use_hdi:
         y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{response_name}_{target}"].T

    # TO DO: rename using more informative names
    lower = f"{response_preds_term}_lower"
    upper = f"{response_preds_term}_upper"
    comparisons_df[lower] = y_hat_bounds[0]
    comparisons_df[upper] = y_hat_bounds[1]

    model_covariates = list(
        comparisons_df.columns[~comparisons_df.columns
                               .isin([contrast_name, response_preds_term, lower, upper])]
                               )
    
    # compute difference between contrast predictions
    contrast_comparison = pd.DataFrame((comparisons_df
                           .groupby(model_covariates)[[response_preds_term, lower, upper]]
                           .diff()
                           .dropna()
                           .reset_index(drop=True)
                           ))
    
    main = conditional.get("horizontal")
    group = conditional.get("color")
    panel = conditional.get("panel")

    if np.unique(comparisons_df[main]).shape[0] == 1:
        contrast_comparison[main] = np.repeat(
            np.unique(comparisons_df[main]), contrast_comparison.shape[0]
        )
    else:
        contrast_comparison[main] = np.unique(comparisons_df[main])

    # TO DO: create a utility function for this
    if group is not None:
        if np.unique(comparisons_df[group]).shape[0] == 1:
            contrast_comparison[group] = np.repeat(
                np.unique(comparisons_df[group]), contrast_comparison.shape[0]
            )
        else:
            contrast_comparison[group] = np.unique(comparisons_df[group])
    elif panel is not None:
        if np.unique(comparisons_df[panel]).shape[0] == 1:
            contrast_comparison[panel] = np.repeat(
                np.unique(comparisons_df[panel]), contrast_comparison.shape[0]
            )
        else:
            contrast_comparison[panel] = np.unique(comparisons_df[panel])

    contrast_comparison = contrast_comparison.rename(
        columns={
            f"{response_preds_term}": "contrast_comparison",
            f"{lower}": "contrast_comparison_lower",
            f"{upper}": "contrast_comparison_upper"
        }
    )

    return comparisons_df, contrast_comparison


def identity(x):
    return x
