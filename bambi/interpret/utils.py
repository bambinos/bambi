# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from typing import Any, Callable, Optional

import numpy as np
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable
from pandas import DataFrame

from bambi import Model
from bambi.utils import get_aliased_name


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
    agg_fn: Callable[[DataFrame], DataFrame] = lambda df: df.mean(),
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
    stat_cols = [
        col for col in data.columns if any(keyword in col for keyword in keywords)
    ]

    match by:
        case None:
            return data
        case _:
            return agg_fn(data.groupby(by=by, observed=True)[stat_cols]).reset_index()


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
