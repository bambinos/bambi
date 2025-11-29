# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass, fields
from typing import Mapping, Optional, Sequence

import numpy as np
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable

from bambi import Model
from bambi.interpret.logs import log_interpret_defaults
from bambi.utils import get_aliased_name, listify

from .plots import PlotConfig


def create_plot_config(
    var_names: Sequence[str], overrides: Optional[Mapping[str, str]] = None
) -> PlotConfig:
    """
    Create a 'PlotConfig' from 'var_names' or 'overrides'.

    Default behavior assigns variables in the order of: main, group, panel. If overrides is
    provided uses those mappings instead.
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
    """
    Loops through the distributional components of a bambi model and
    returns a dictionary of terms.
    """
    terms = {}
    for component in model.distributional_components.values():
        if component.design.common:
            terms.update(component.design.common.terms)

        if component.design.group:
            terms.update(component.design.group.terms)

    return terms


def get_model_covariates(model: Model) -> np.ndarray:
    """
    Return covariates specified in the model.
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


def get_response_and_target(model: Model, target: str):
    """
    Parameters
    ----------
    target : str
        Target model parameter...
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


def identity(x):
    return x
