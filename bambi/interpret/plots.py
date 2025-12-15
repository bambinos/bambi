from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd
import seaborn.objects as so
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype
from seaborn.objects import Plot


@dataclass
class PlotConfig:
    """Configuration for plotting interpret results.

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


def create_plot_config(
    var_names: list[str], overrides: Optional[Mapping[str, str]] = None
) -> PlotConfig:
    """Create a PlotConfig from variable names or overrides.

    Parameters
    ----------
    var_names : list[str]
        List of variable names to create the plot configuration from.
    overrides : Mapping[str, str] or None
        Dictionary to override default plotting sequence. Valid keys are 'main', 'group', and 'panel'.

    Returns
    -------
    PlotConfig
        A PlotConfig object with main, group, and panel assignments.
        Default behavior assigns variables in the order of: main, group, panel.
        If overrides is provided, uses those mappings instead.

    Raises
    ------
    ValueError
        If no variable names are provided, if more than 3 variables are provided,
        if 'main' key is missing from overrides, or if invalid keys are in overrides.
    """
    match overrides:
        # Pattern match on valid subplot_kwarg structure
        case {"main": main, "group": group, "panel": panel}:
            return PlotConfig(main=main, group=group, panel=panel)

        case {"main": main, "group": group}:
            return PlotConfig(main=main, group=group)

        case {"main": main, "panel": panel}:
            return PlotConfig(main=main, panel=panel)

        case {"main": main}:
            return PlotConfig(main=main)

        # Invalid: subplot_kwargs provided but missing 'main' or has extra keys
        case dict() as override_dict if override_dict:
            provided_keys = set(override_dict.keys())
            allowed_keys = {"main", "group", "panel"}

            if "main" not in provided_keys:
                raise ValueError(
                    "'subplot_kwargs' must contain 'main' key when overriding default plotting sequence."
                )

            invalid_keys = provided_keys - allowed_keys
            raise ValueError(
                f"Invalid keys in subplot_kwargs: {invalid_keys}. "
                f"Only 'main', 'group', and 'panel' are allowed."
            )

        # Default plotting sequence
        case _:
            if not var_names:
                raise ValueError("At least one variable name must be provided")
            if len(var_names) > 3:
                raise ValueError(
                    f"Cannot create plot config with more than 3 variables. Received: {len(var_names)}"
                )

            return PlotConfig(
                main=var_names[0],
                group=var_names[1] if len(var_names) > 1 else None,
                panel=var_names[2] if len(var_names) > 2 else None,
            )


def plot(data: DataFrame, config: PlotConfig, theme: dict[str, Any]) -> Plot:
    """Declaratively plot data according to a plot configuration.

    Parameters
    ----------
    data : DataFrame
        An interpret summary DataFrame containing estimate, lower, and upper columns
        along with variable columns specified in the config.
    config : PlotConfig
        A plotting configuration used to build a Seaborn objects plotting specification.
        Specifies the main variable (x-axis), optional grouping variable (color),
        and optional panel variable (facets).
    theme : dict or None
        A dictionary of 'matplotlib rc' parameters.

    Returns
    -------
    Plot
        A Seaborn objects Plot with appropriate marks (Dot/Line) and bands (Range/Band)
        based on the data types of the variables. Categorical and integer types use
        strip plots with error bars, while float types use line plots with bands.

    Raises
    ------
    TypeError
        If the main variable has an unsupported data type.
    """
    estimate_dim = list(filter(lambda col: "dim" in col, data.columns))
    if estimate_dim:
        print(f"Detected an estimate dimension column: {estimate_dim}")

    # Plotting specification labels
    ymin = next(filter(lambda col: "lower" in col, data.columns))
    ymax = next(filter(lambda col: "upper" in col, data.columns))

    # Plotting customization
    # - ticks
    # - sharex and sharey
    # -

    # Base figure (must include x-y axis)
    plot = so.Plot(data, x=config.main, y="estimate", color=config.group)

    # Add a facet layer (only adds if config.panel is not None)
    plot = plot.facet(col=config.panel)
    # TODO
    # plot = plot.share(x=config.sharex, y=config.sharey)

    # Add a "main" layer
    match data[config.main].dtype:
        # Strip plot if categorical or integer dtype
        case dtype if pd.CategoricalDtype() or is_integer_dtype(dtype):
            plot = plot.add(so.Dot(), so.Dodge())
            plot = plot.add(
                so.Range(),
                so.Dodge(),
                ymin=ymin,
                ymax=ymax,
            )
        # Line plot if numeric or integer dtype
        case dtype if is_float_dtype(dtype):
            plot = plot.add(so.Line())
            plot = plot.add(
                so.Band(alpha=0.3),
                ymin=ymin,
                ymax=ymax,
            )
        case _:
            raise TypeError(f"Unsupported data type: {data[config.main].dtype}")

    # Add theme dictionary
    plot = plot.theme(theme)

    plot.show()

    return plot
