# pylint: disable = too-many-nested-blocks
from typing import Any, Callable, Optional

import numpy as np
import xarray as xr
from arviz import InferenceData
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from bambi.models import Model
from bambi.utils import get_aliased_name


def create_inference_data(
    preds_idata: InferenceData, preds_data: DataFrame
) -> InferenceData:
    """Create a new InferenceData object by replacing the observed_data group with the
    `preds_data`.

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


def get_response_and_target(model: Model, target: str) -> tuple[str, str | None]:
    """Get the response name and target parameter from the model.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    target : str
        Target model parameter (e.g., 'mean', or a distributional component name).

    Returns
    -------
    tuple[str, str or None]
        A tuple containing the response name and the target parameter name.
        If target is 'mean', returns the response name and the parent parameter.
        Otherwise, returns the component alias (or response name) and the target (or None).
    """
    match target:
        case "mean":
            return (
                get_aliased_name(model.response_component.term),
                model.family.likelihood.parent,
            )
        case _:
            component = model.components[target]
            return (
                get_aliased_name(component)
                if component.alias
                else get_aliased_name(model.response_component.term),
                None if component.alias else target,
            )


def aggregate(
    data: DataFrame,
    by: Optional[str | list[str]],
    agg_fn: Callable[
        [DataFrame | Series | DataFrameGroupBy | SeriesGroupBy], DataFrame
    ] = lambda df: df.mean(),
) -> DataFrame:
    """Group data by variable(s) and apply an aggregation function.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to aggregate.
    by : str or list[str] or None
        Column name(s) to group by. If None, returns data unchanged.
    agg_fn : Callable[[DataFrame], DataFrame]
        Aggregation function to apply to each group. Default is mean.

    Returns
    -------
    DataFrame
        The aggregated DataFrame with summary statistics.
    """
    keywords = ["estimate", "lower", "upper"]
    # Lower and upper columns can have different names
    # For example, lower_0.03% or lower_0.05%
    selector_cols = [
        col for col in data.columns if any(keyword in col for keyword in keywords)
    ]

    match by:
        case None:
            return data
        case "all":
            return agg_fn(data[selector_cols]).to_frame().transpose()
        case _:
            return agg_fn(
                data.groupby(by=by, observed=True)[selector_cols]
            ).reset_index()


def get_model_terms(model: Model) -> dict:
    """Loop through the distributional components of a Bambi model and return terms.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.

    Returns
    -------
    dict
        A dictionary containing all terms from the model's distributional components.
    """
    terms = {}
    for component in model.distributional_components.values():
        if component.design.common:
            terms.update(component.design.common.terms)

        if component.design.group:
            terms.update(component.design.group.terms)

    return terms


def get_model_covariates(model: Model) -> np.ndarray:
    """Return covariates specified in the model.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.

    Returns
    -------
    np.ndarray
        An array of unique covariate names present in the model.
    """
    terms = get_model_terms(model)
    covariates = []
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, look for relevant argument names
                if isinstance(component, Call):
                    # Add variable names passed as unnamed arguments
                    covariates.extend(
                        arg.name
                        for arg in component.call.args
                        if isinstance(arg, LazyVariable)
                    )
                    # Add variable names passed as named arguments
                    covariates.extend(
                        kwarg_value.name
                        for kwarg_value in component.call.kwargs.values()
                        if isinstance(kwarg_value, LazyVariable)
                    )
                else:
                    covariates.append(component.name)
        elif hasattr(term, "factor"):
            covariates.extend(list(term.var_names))

    # Don't include non-covariate names (#797)
    covariates = [name for name in covariates if name in model.data]

    return np.unique(covariates)


def identity(x: Any) -> Any:
    """Identity function that returns its input unchanged.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    Any
        The same value as the input.
    """
    return x
