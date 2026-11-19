"""Plots of posterior causal estimands and their uncertainty intervals."""

import pandas as pd
import seaborn.objects as so

from bambi.causal.types import CausalResult
from bambi.interpret.plots import PlottingConfig, plot


def plot_effects(result, *, main, group=None, panel=None, fig_kwargs=None):
    """Plot actual estimand summaries using explicit axes and optional facets.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`g_computation`.
    main : str
        Summary column used for the horizontal axis.
    group, panel : str, optional
        Summary columns used for color grouping and faceting.
    fig_kwargs : dict, optional
        Figure options accepted by Bambi's interpret plotting configuration.

    Returns
    -------
    seaborn.objects.Plot
        Declarative plot, supporting ``show()``. Differences have a zero
        reference and risk or odds ratios have a one reference.
    """
    if not isinstance(result, CausalResult):
        raise ValueError("result must be a CausalResult.")
    data = result.summary.copy(deep=True)
    keys = list(dict.fromkeys(key for key in (main, group, panel) if key is not None))
    if any(key not in data for key in keys):
        raise ValueError("Plot axes must name result summary columns.")
    if not pd.api.types.is_numeric_dtype(data[main]):
        data[main] = pd.Categorical(data[main], categories=pd.unique(data[main]))
    config = PlottingConfig.from_params(
        keys,
        subplot_kwargs={"main": main, "group": group, "panel": panel},
        fig_kwargs=fig_kwargs,
    )
    figure = plot(data, config)
    comparisons = set(data["comparison"])
    if comparisons == {"mean"}:
        return figure
    if len(comparisons) != 1:
        raise ValueError("A plot must contain one comparison scale.")
    baseline = 0 if comparisons == {"difference"} else 1
    reference = data[keys].drop_duplicates().copy()
    reference["__reference__"] = baseline
    return figure.add(so.Line(color="0.5", linestyle="--"), data=reference, y="__reference__").add(
        so.Dash(color="0.5"), data=reference, y="__reference__"
    )
