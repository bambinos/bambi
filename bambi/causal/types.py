"""Results of posterior standardization over an explicit reference population."""

from typing import NamedTuple

import pandas as pd
import xarray as xr


class CausalResult(NamedTuple):
    """Posterior estimands, their summaries, and the assumptions supplied by the user.

    Attributes
    ----------
    summary : pandas.DataFrame
        One row per intervention and reference-population stratum.
    draws : xarray.DataTree
        ``effects/estimate`` contains the actual estimand draws, with dimensions
        ``chain``, ``draw``, and ``__obs__``. ``standardized/mean`` contains the
        standardized outcome means for each intervention and stratum.
    metadata : dict
        Reference-population provenance and causal assumptions. These records do
        not establish exchangeability, positivity, or consistency.
    """

    summary: pd.DataFrame
    draws: xr.DataTree
    metadata: dict
