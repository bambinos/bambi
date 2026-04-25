# pylint: disable = too-many-nested-blocks
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import xarray as xr
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

from bambi.models import Model
from bambi.utils import get_aliased_name


class TargetInfo(NamedTuple):
    """Information regarding which type of prediction is required based on a `target`.

    `interpret` allows users to plot target quantities such as posterior parameters,
    and or the posterior predictive.

    Parameters
    ----------
    response_name : str
        Key for transforms dict lookup
    var_name : str
        Variable name to extract from idata[group]
    group : str
        `posterior` or `posterior_predictive`
    predict_kind : str
        `response_params` or `response` — passed to model.predict()
    """

    response_name: str
    var_name: str
    group: str
    predict_kind: str


def create_datatree(preds_idata: xr.DataTree, preds_data: DataFrame) -> xr.DataTree:
    """Create a new DataTree object by replacing the observed_data group with the
    `preds_data`.

    Parameters
    ----------
    preds_idata : DataTree
        The DataTree object containing posterior samples.
    preds_data : DataFrame
        The DataFrame to use as the new observed_data group.

    Returns
    -------
    DataTree
        A new DataTree object with the observed_data group replaced by preds_data.

    Raises
    ------
    ValueError
        If the DataTree object does not contain an 'observed_data' group.
    NotImplementedError
        If the DataTree object has more than one coordinate.
    """
    new_grid_idata = preds_idata.copy()
    xr_df = xr.Dataset.from_dataframe(preds_data)

    if "data" in new_grid_idata.children:
        coordinate_name = list(new_grid_idata["data"].coords)
        # Delete the pandas-based data group and add the preds xr.Dataset
        del new_grid_idata["data"]
        new_grid_idata["data"] = xr_df
    elif "observed_data" in new_grid_idata.children:
        coordinate_name = list(new_grid_idata["observed_data"].coords)
        # Delete the pandas-based observed_data group and add the preds xr.Dataset
        del new_grid_idata["observed_data"]
        new_grid_idata["observed_data"] = xr_df
        new_grid_idata["data"] = new_grid_idata["observed_data"].ds
    else:
        raise ValueError("DataTree object does not contain a 'data' or 'observed_data' group.")

    if len(coordinate_name) > 1:
        raise NotImplementedError("Only one coordinate is currently supported.")
    coordinate_name = coordinate_name[0]

    # Rename index to match coordinate name in other DataTree groups.
    data_group = new_grid_idata["data"].ds
    if "index" in data_group.dims and coordinate_name != "index":
        new_grid_idata["data"] = data_group.rename({"index": coordinate_name})
        if "observed_data" in new_grid_idata.children:
            new_grid_idata["observed_data"] = new_grid_idata["data"].ds

    return new_grid_idata


def resolve_target(model: Model, target: str) -> TargetInfo:
    """Resolve the target parameter into the arguments required to pass to the predict
    method of a Bambi model.

    Parameters
    ----------
    model : Model
        The fitted Bambi model.
    target : str
        Which quantity to extract. `"mean"` for the posterior of the parent
        parameter (e.g. `"mu"`). Pass the response variable name (e.g. `"mpg"`) for
        posterior predictive samples. Pass a distributional component name (e.g.
        `"sigma"`) for the posterior of that component.

    Returns
    -------
    TargetInfo
        A named tuple with `response_name`, `var_name`, `group`, and `predict_kind`.
    """
    response_name = get_aliased_name(model.response_component.term)
    match target:
        case "mean":
            return TargetInfo(
                response_name,
                model.family.likelihood.parent,
                "posterior",
                "response_params",
            )
        case t if t == response_name:
            return TargetInfo(response_name, response_name, "posterior_predictive", "response")
        case _:
            component = model.components[target]
            if component.alias:
                alias = get_aliased_name(component)
                return TargetInfo(alias, alias, "posterior", "response_params")
            else:
                return TargetInfo(response_name, target, "posterior", "response_params")


def aggregate(
    data: DataFrame,
    by: Optional[str | list[str]],
    agg_fn: Callable[
        [DataFrame | Series | DataFrameGroupBy | SeriesGroupBy], DataFrame
    ] = lambda df: df.mean(),
    preserve: Optional[list[str]] = None,
) -> DataFrame:
    """Group data by variable(s) and apply an aggregation function.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to aggregate.
    by : str or list[str] or None
        Column name(s) to group by. If None, returns data unchanged.
    agg_fn : Callable[[DataFrame | Series | DataFrameGroupBy | SeriesGroupBy], DataFrame],
    optional
        Aggregation function to apply to each group. Default is mean.
    preserve : list[str] or None
        Column names that must survive aggregation by being included as groupby keys.

    Returns
    -------
    DataFrame
        The aggregated DataFrame with summary statistics.
    """
    keywords = ["estimate", "lower", "upper"]
    # Lower and upper columns can have different names
    # For example, lower_0.03% or lower_0.05%
    selector_cols = [col for col in data.columns if any(keyword in col for keyword in keywords)]
    preserve = preserve or []

    match by:
        case None:
            return data
        case "all":
            if preserve:
                return agg_fn(data.groupby(preserve, observed=True)[selector_cols]).reset_index()
            return agg_fn(data[selector_cols]).to_frame().transpose()
        case _:
            by = [by] if isinstance(by, str) else list(by)
            all_groups = list(dict.fromkeys(preserve + by))
            return agg_fn(data.groupby(by=all_groups, observed=True)[selector_cols]).reset_index()


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
                        arg.name for arg in component.call.args if isinstance(arg, LazyVariable)
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
