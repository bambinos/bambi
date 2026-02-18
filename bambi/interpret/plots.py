# pylint: disable=redefined-outer-name
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from typing import Any, Mapping

import pandas as pd
import seaborn.objects as so
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype
from seaborn.objects import Plot


@dataclass(frozen=True)
class FigureConfig:
    """Configuration for customizing the appearance of a Seaborn figure."""

    sharex: bool = True
    sharey: bool = True
    alpha: float = 0.3
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    wrap: int | None = None
    theme: dict[str, Any] | None = None

    @staticmethod
    def from_kwargs(kwargs: dict[str, Any] | None = None) -> FigureConfig:
        """Create a FigureConfig from a dictionary of keyword arguments.

        Parameters
        ----------
        kwargs : dict or None
            A dictionary of figure-level arguments.

        Returns
        -------
        FigureConfig
        """
        return FigureConfig(**(kwargs or {}))


@dataclass(frozen=True)
class SubplotConfig:
    """Configuration for specifying the content to plot from an
    interpret summary results DataFrame.

     Parameters
     ----------
     main : str
         The primary variable to plot on the x-axis.
     group : str or None
         Optional grouping variable for color differentiation.
     panel : str or None
         Optional faceting variable for creating subplots.
    """

    main: str
    group: str | None = None
    panel: str | None = None

    @staticmethod
    def from_params(
        var_names: list[str], overrides: Mapping[str, str] | None = None
    ) -> SubplotConfig:
        """Create a SubplotConfig from variable names or overrides.

        Parameters
        ----------
        var_names : list[str]
            List of variable names to create the plot configuration from.
        overrides : Mapping[str, str] or None
            Dictionary to override default plotting sequence. Valid keys
            are 'main', 'group', and 'panel'.

        Returns
        -------
        SubplotConfig

        Raises
        ------
        ValueError
            If no variable names are provided, if more than 3 variables are
            provided, if 'main' key is missing from overrides, or if invalid
            keys are in overrides.
        """
        match (overrides, var_names):
            # User explicitly controls layout via overrides
            case (dict(), _):
                allowed = {"main", "group", "panel"}
                invalid = set(overrides) - allowed
                if invalid:
                    raise ValueError(
                        f"Invalid keys in subplot_kwargs: {invalid}. "
                        f"Only 'main', 'group', and 'panel' are allowed."
                    )
                if "main" not in overrides:
                    raise ValueError(
                        "'subplot_kwargs' must contain 'main' key when overriding "
                        "default plotting sequence."
                    )
                return SubplotConfig(
                    main=overrides["main"],
                    group=overrides.get("group"),
                    panel=overrides.get("panel"),
                )
            # No overrides, no variables (cannot determine plot axes)
            case (None, []):
                raise ValueError(
                    "Unable to determine plotting variable(s) when 'conditional' is 'None'. "
                    "Either pass variable(s) to `conditional` or specify the layout via "
                    "'subplot_kwargs'."
                )
            # No overrides, too many variables (cannot auto-assign to 3 aesthetics)
            case (None, names) if len(names) > 3:
                raise ValueError(
                    f"Cannot automatically plot more than 3 conditional variables "
                    f"(received {len(names)}). Either reduce the number of variables, "
                    f"pass `average_by` to reduce dimensionality, or use "
                    f"`subplot_kwargs` to explicitly assign variables to plot axes."
                )
            # No overrides, 1-3 variables (auto-assign positionally)
            case (None, names):
                return SubplotConfig(
                    main=names[0],
                    group=names[1] if len(names) > 1 else None,
                    panel=names[2] if len(names) > 2 else None,
                )


@dataclass(frozen=True)
class PlottingConfig:
    subplot: SubplotConfig
    figure: FigureConfig

    @staticmethod
    def from_params(
        var_names: list[str],
        subplot_kwargs: Mapping[str, str] | None = None,
        fig_kwargs: dict[str, Any] | None = None,
    ) -> PlottingConfig:
        """Create a PlottingConfig from variable names and optional overrides.

        Parameters
        ----------
        var_names : list[str]
            List of variable names for the subplot configuration.
        subplot_kwargs : Mapping[str, str] or None
            Overrides for the subplot configuration.
        fig_kwargs : dict or None
            Keyword arguments for figure customization.

        Returns
        -------
        PlottingConfig
        """
        return PlottingConfig(
            subplot=SubplotConfig.from_params(var_names, subplot_kwargs),
            figure=FigureConfig.from_kwargs(fig_kwargs),
        )


def _get_interval_pairs(data: DataFrame, base_alpha: float) -> list[tuple[str, str, float, float]]:
    """Extract lower/upper column pairs and compute alpha and linewidth for each interval.

    Pairs are sorted by interval width (widest first) so that wider bands
    are drawn first (behind narrower intervals). Alpha ramps linearly from
    `base_alpha / n` (widest) to `base_alpha` (narrowest). Linewidth ramps
    linearly from `1` (widest) to `2` (narrowest).

    Parameters
    ----------
    data : DataFrame
        Summary DataFrame containing `lower_*` and `upper_*` columns.
    base_alpha : float
        Maximum alpha value (used for the narrowest interval).

    Returns
    -------
    list[tuple[str, str, float, float]]
        Each tuple is ``(lower_col, upper_col, alpha, linewidth)``.
    """
    lower_cols = [col for col in data.columns if col.startswith("lower_")]
    upper_cols = [col for col in data.columns if col.startswith("upper_")]

    # NOTE: Implicit assumption is that the percentages in the column names
    # are indeed the correct intervals
    def _pct(col: str) -> float:
        """Parse the percentage from column names to pair them"""
        m = re.search(r"([\d.]+)%", col)
        return float(m.group(1)) if m else 0.0

    # Sort lower cols by percentage ascending (smallest % = widest interval)
    lower_cols.sort(key=_pct)
    # Sort upper cols by percentage descending (largest % = widest interval)
    upper_cols.sort(key=_pct, reverse=True)

    n = len(lower_cols)
    pairs = []
    for i, (lo, hi) in enumerate(zip(lower_cols, upper_cols)):
        alpha = base_alpha * (i + 1) / n  # 0.15 = 0.3 * (0 + 1) / 2
        linewidth = 1 + i / max(n - 1, 1)  # 1.0 = 1 + 0 / max(2 - 1, 1)
        pairs.append((lo, hi, alpha, linewidth))

    return pairs


def _add_main_layer(plot: Plot, data: DataFrame, config: PlottingConfig) -> Plot:
    intervals = _get_interval_pairs(data, config.figure.alpha)

    match data[config.subplot.main].dtype:
        # Strip plot if categorical or integer dtype
        case dtype if isinstance(dtype, pd.CategoricalDtype) or is_integer_dtype(dtype):
            plot = plot.add(so.Dot(), so.Dodge())
            plot = reduce(
                # Repeatedly add a the uncertainty Range to the plot never indexing into
                # the alpha element because this is a strip plot
                lambda p, iv: p.add(so.Range(linewidth=iv[3]), so.Dodge(), ymin=iv[0], ymax=iv[1]),
                intervals,
                plot,
            )
        # Line plot if numeric dtype
        case dtype if is_float_dtype(dtype):
            plot = plot.add(so.Line())
            # Repeatedly add the uncertainty Band to the plot never indexing into
            # the linewidth element because this is a line plot
            plot = reduce(
                lambda p, iv: p.add(so.Band(alpha=iv[2]), ymin=iv[0], ymax=iv[1]),
                intervals,
                plot,
            )
        case _:
            raise TypeError(f"Unsupported data type: {data[config.subplot.main].dtype}")

    return plot


def plot(
    data: DataFrame,
    config: PlottingConfig,
) -> Plot:
    """Declaratively plot data according to a plotting configuration.

    Parameters
    ----------
    data : DataFrame
        An interpret summary DataFrame containing estimate, lower, and upper columns
        along with variable columns specified in the config.
    config : PlottingConfig
        A plotting configuration used to build and customize the appearance of a Seaborn
        objects plotting specification.

    Returns
    -------
    Plot
        A Seaborn objects Plot displaying the information of an `interpret` summary DataFrame.

    Raises
    ------
    TypeError
        If the main variable has an unsupported data type.
    """
    # Base plot (must include x-y axis)
    plot = so.Plot(data, x=config.subplot.main, y="estimate", color=config.subplot.group)
    # Force color cycle to nominal instead of gradient
    plot = plot.scale(color=so.Nominal())
    # Add a facet layer (only adds if panel is not None)
    plot = plot.facet(col=config.subplot.panel, wrap=config.figure.wrap)
    # Share x-y axis labels
    plot = plot.share(x=config.figure.sharex, y=config.figure.sharey)
    # Add a main layer (line or stripplot based on dtype)
    plot = _add_main_layer(plot, data, config)

    # Adjust figure labels
    plot = plot.label(
        x=config.figure.xlabel or config.subplot.main,
        y=config.figure.ylabel or "estimate",
        title=config.figure.title,
    )
    # Set plot theme (dict of matplotlib rc parameters)
    if config.figure.theme:
        plot = plot.theme(config.figure.theme)

    return plot
