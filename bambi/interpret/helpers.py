from functools import partial
from itertools import combinations
from typing import Any, Callable

import numpy as np
import xarray as xr
from arviz import InferenceData
from pandas import DataFrame
from xarray import DataArray

from .types import Contrast, Variable


def create_inference_data(
    preds_idata: InferenceData, preds_data: DataFrame
) -> InferenceData:
    """Create a new InferenceData object by replacing the observed_data group with the
    'preds_data'.

    Parameters
    ----------
    preds_idata : InferenceData
        The InferenceData object containing posterior samples.
    preds_data : DataFrame
        The DataFrame to use as the new observed_data group.

    Returns
    -------
    InferenceData
        A new InferenceData object with the observed_data group replaced by preds_data.

    Raises
    ------
    ValueError
        If the InferenceData object does not contain an 'observed_data' group.
    NotImplementedError
        If the InferenceData object has more than one coordinate.
    """
    new_grid_idata = preds_idata.copy()
    xr_df = xr.Dataset.from_dataframe(preds_data)

    if "observed_data" in new_grid_idata.groups():
        coordinate_name = list(new_grid_idata["observed_data"].coords)
        # Delete the Pandas-based observed_data group and add the preds xr.Dataset
        del new_grid_idata.observed_data
        new_grid_idata.add_groups(data=xr_df)
    else:
        raise ValueError(
            "InferenceData object does not contain a 'data' or 'observed_data' group."
        )

    if len(coordinate_name) > 1:
        raise NotImplementedError("Only one coordinate is currently supported.")
    coordinate_name = coordinate_name[0]

    # Rename index to match coordinate name in other InferenceData groups
    new_grid_idata.data = new_grid_idata.data.rename({"index": coordinate_name})

    return new_grid_idata


def filter_draws(
    val: Any, idata: InferenceData, group: str, target: str, variable: Variable
) -> DataArray:
    """Filter draws from an InferenceData group based on variable values.

    Parameters
    ----------
    val : Any
        The value to filter by.
    idata : InferenceData
        The InferenceData object containing the draws.
    group : str
        The name of the group to filter from (e.g., 'posterior', 'posterior_predictive').
    target : str
        The target variable name within the group.
    variable : Variable
        The variable (pandas Series) to use for filtering.

    Returns
    -------
    DataArray
        An xarray DataArray containing the filtered draws.
    """
    coordinate_name = list(idata["data"].coords)[0]

    # Get indices where condition is true
    # np.logical_and.reduce is useful if multiple conditions (contrast vals)
    idx = np.where(np.logical_and.reduce([idata["data"][variable.name] == val]))[0]
    draws = idata[group].isel({coordinate_name: idx})[target]

    # In the case of main and or parent parameters (e.g., distributional models)
    if coordinate_name in draws.coords:
        new_coords = np.arange(len(idx))
        draws = draws.assign_coords({coordinate_name: new_coords})

    return draws


def compare(
    idata: InferenceData,
    contrast: Contrast,
    target: str,
    group: str,
    comparison_fn: Callable,
) -> dict[str, DataArray]:
    """Compare samples in an InferenceData group given the Contrast variables.

    Parameters
    ----------
    idata : InferenceData
        The InferenceData object containing the samples to compare.
    contrast : Contrast
        The Contrast object specifying the variable to create contrasts for.
    target : str
        The target variable name to compare within the group.
    group : str
        The name of the group to compare (e.g., 'posterior', 'posterior_predictive').
    comparison_fn : Callable
        The comparison function to apply to pairs of draws (e.g., difference, ratio).

    Returns
    -------
    dict[str, DataArray]
        A dictionary mapping comparison labels (e.g., "1_vs_2") to DataArrays
        containing the comparison results.
    """
    filter_fn = partial(
        filter_draws,
        idata=idata,
        group=group,
        target=target,
        variable=contrast.variable,
    )

    # Apply filter_draws over all contrast variable values
    filtered_draws = list(map(filter_fn, contrast.variable))
    # Generate unique pairs for each draw
    paired_draws = combinations(enumerate(filtered_draws), r=2)
    # Apply a comparison function to each pair
    res = {
        f"{contrast.variable[i]}_vs_{contrast.variable[j]}": comparison_fn(a, b)
        for (i, a), (j, b) in paired_draws
    }

    return res
