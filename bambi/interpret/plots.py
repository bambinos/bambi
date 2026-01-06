from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd
import seaborn.objects as so
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype
from seaborn.objects import Plot


@dataclass
class FigConfig:
    """Configuration for customizing the appearance of a Seaborn figure."""

    sharex: bool = True
    sharey: bool = True
    # legend: bool = True # TODO
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    wrap: Optional[int] = None


@dataclass
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


@dataclass
class PlottingConfig:
    plot: SubplotConfig
    figure: FigConfig


def create_figure_config(kwargs: Optional[dict[str, Any]] = None) -> FigConfig:
    """Create a `FigConfig` to alter the default Seaborn figure-level
    appearance.

    Parameters
    ----------
    kwargs : dict or None
        A dictionary of figure-level arguments.

    Returns
    -------
    FigConfig
        A `FigConfig` object with attributes used to alter a Seaborn figure
        appearance.
    """
    return FigConfig(**(kwargs or {}))


def create_subplot_config(
    var_names: list[str], overrides: Optional[Mapping[str, str]] = None
) -> SubplotConfig:
    """Create a `SubplotConfig` from variable names or overrides.

    Parameters
    ----------
    var_names : list[str]
        List of variable names to create the plot configuration from.
    overrides : Mapping[str, str] or None
        Dictionary to override default plotting sequence. Valid keys are 'main', 'group', and 'panel'.

    Returns
    -------
    SubplotConfig
        A SubplotConfig object with main, group, and panel assignments.
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
            return SubplotConfig(main=main, group=group, panel=panel)
        case {"main": main, "group": group}:
            return SubplotConfig(main=main, group=group)
        case {"main": main, "panel": panel}:
            return SubplotConfig(main=main, panel=panel)
        case {"main": main}:
            return SubplotConfig(main=main)
        # Invalid subplot_kwargs provided but missing 'main' or has extra keys
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

            return SubplotConfig(
                main=var_names[0],
                group=var_names[1] if len(var_names) > 1 else None,
                panel=var_names[2] if len(var_names) > 2 else None,
            )


def plot(
    data: DataFrame,
    config: PlottingConfig,
    theme: dict[str, Any],
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
    theme : dict or None
        A dictionary of 'matplotlib rc' parameters.

    Returns
    -------
    Plot
        A Seaborn objects Plot displaying the information of an `interpret` summary DataFrame.

    Raises
    ------
    TypeError
        If the main variable has an unsupported data type.
    """
    estimate_dim = list(filter(lambda col: "dim" in col, data.columns))
    if estimate_dim:
        print(f"Detected an estimate dimension column: {estimate_dim}")

    # Plot labels
    ymin = next(filter(lambda col: "lower" in col, data.columns))
    ymax = next(filter(lambda col: "upper" in col, data.columns))

    # Base plot (must include x-y axis)
    plot = so.Plot(data, x=config.plot.main, y="estimate", color=config.plot.group)
    # Add a facet layer (only adds if config.panel is not None)
    plot = plot.facet(col=config.plot.panel, wrap=config.figure.wrap)
    # Share axis labels
    plot = plot.share(x=config.figure.sharex, y=config.figure.sharey)
    # Add a "main" layer
    match data[config.plot.main].dtype:
        # Strip plot if categorical or integer dtype
        case dtype if pd.CategoricalDtype() or is_integer_dtype(dtype):
            plot = plot.add(so.Dot(), so.Dodge())
            plot = plot.add(
                so.Range(),
                so.Dodge(),
                ymin=ymin,
                ymax=ymax,
            )
        # Line plot if numeric dtype
        case dtype if is_float_dtype(dtype):
            plot = plot.add(so.Line())
            plot = plot.add(
                so.Band(alpha=0.3),
                ymin=ymin,
                ymax=ymax,
            )
        case _:
            raise TypeError(f"Unsupported data type: {data[config.plot.main].dtype}")

    # Adjust figure labels
    plot = plot.label(
        x=config.figure.xlabel,
        y=config.figure.ylabel,
        title=config.figure.title,
        # legend=config.figure.legend, # TODO
    )
    # Set plot theme (matplotlib rc parameters)
    plot = plot.theme(theme)

    plot.show()

    return plot
