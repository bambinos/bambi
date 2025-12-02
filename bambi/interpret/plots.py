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
    "Product type"

    main: str
    group: str | None = None
    panel: str | None = None
    # sharex: bool = True
    # sharey: bool = True
    # theme: Optional[dict[Any, Any]] = None


def plot(data: DataFrame, config: PlotConfig) -> Plot:
    """Declaratively plot `data` according to a plot `config`.

    Parameters
    ----------
    data : DataFrame
        A `interpret` summary dataframe.
    config : PlotConfig
        A plotting config used to build a Seaborn objects plotting specification.

    Returns
    -------
    A Seaborn objects `Plot`.
    """
    estimate_dim = list(filter(lambda col: "dim" in col, data.columns))
    if estimate_dim:
        print(f"Detected estimate dimension column: {estimate_dim}")

    # Base figure (must include x-y axis)
    plot = so.Plot(data, x=config.main, y="estimate", color=config.group)

    # Add a facet layer (only adds if config.panel is not None)
    plot = plot.facet(col=config.panel)
    # TODO
    # plot = plot.share(x=config.sharex, y=config.sharey)

    # Add a "main" layer
    match data[config.main].dtype:
        # Strip plot if categorical or integer dtype
        case dtype if pd.CategoricalDtype() or dtype if is_integer_dtype(dtype):
            plot = plot.add(so.Dot(), so.Dodge())
            plot = plot.add(
                so.Range(),
                so.Dodge(),
                ymin="lower_0.03%",
                ymax="upper_0.97%",
            )
        # Line plot if numeric or integer dtype
        case dtype if is_float_dtype(dtype):
            plot = plot.add(so.Line())
            plot = plot.add(
                so.Band(alpha=0.3),
                ymin="lower_0.03%",
                ymax="upper_0.97%",
            )
        case _:
            raise TypeError(f"Unsupported data type: {data[config.main].dtype}")

    # Add theme dictionary
    # plot = plot.theme(config.theme)

    plot.show()

    return plot
