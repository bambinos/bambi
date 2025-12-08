from dataclasses import dataclass, fields
from typing import Any, Optional

import pandas as pd
import seaborn.objects as so
from pandas import DataFrame
from pandas.api.types import (
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.api import is_float_dtype
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


def plot(data: DataFrame, config: PlotConfig) -> Plot:
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
        print(f"Detected estimate dimension column: {estimate_dim}")

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
    # plot = plot.theme(config.theme)

    plot.show()

    return plot
