# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable
from pandas import DataFrame

from bambi import Model
from bambi.interpret.logs import log_interpret_defaults
from bambi.utils import get_aliased_name

from .plots import PlotConfig


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
    """Aggregate data by grouping variables.

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
                # if the component is a function call, look for relevant argument names
                if isinstance(component, Call):
                    # Add variable names passed as unnamed arguments
                    covariates.append(
                        [
                            arg.name
                            for arg in component.call.args
                            if isinstance(arg, LazyVariable)
                        ]
                    )
                    # Add variable names passed as named arguments
                    covariates.append(
                        [
                            kwarg_value.name
                            for kwarg_value in component.call.kwargs.values()
                            if isinstance(kwarg_value, LazyVariable)
                        ]
                    )
                else:
                    covariates.append([component.name])
        elif hasattr(term, "factor"):
            covariates.append(list(term.var_names))

    flatten_covariates = [item for sublist in covariates for item in sublist]

    # Don't include non-covariate names (#797)
    flatten_covariates = [name for name in flatten_covariates if name in model.data]

    return np.unique(flatten_covariates)


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
