from dataclasses import dataclass
from statistics import mode
import itertools
from typing import Callable, Union, Tuple, Any

import numpy as np
import pandas as pd
from formulae.terms.call import Call
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype


@dataclass
class Covariates:
    main: str
    group: Union[str, None]
    panel: Union[str, None]


def get_model_terms(model) -> dict:
    """
    """
    terms = {}
    for component in model.distributional_components.values():
        if component.design.common:
            terms.update(component.design.common.terms)

        if component.design.group:
            terms.update(component.design.group.terms)

    return terms


def get_covariates(covariates: dict) -> Covariates:
    """
    """
    covariate_kinds = ("horizontal", "color", "panel")
    if any(key in covariate_kinds for key in covariates.keys()):
        # default if user did not pass their own conditional dict
        main = covariates.get("horizontal")
        group = covariates.get("color", None)
        panel = covariates.get("panel", None)
    else:
        # assign main, group, panel based on the number of variables
        # passed by the user in their conditional dict
        length = len(covariates.keys())
        if length == 1:
            main = covariates.keys()
            group = None
            panel = None
        elif length == 2:
            main, group = covariates.keys()
            panel = None
        elif length == 3:
            main, group, panel = covariates.keys()

    return Covariates(main, group, panel)


def enforce_dtypes(data, df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce dtypes of the original data to the new data.
    """
    observed_dtypes = data.dtypes
    for col in df.columns:
        if col in observed_dtypes.index:
            df[col] = df[col].astype(observed_dtypes[col])
    return df


def contrast_dtype(model, contrast_predictor):

    if isinstance(contrast_predictor, list):
        contrast_predictor = " ".join(contrast_predictor)

    terms = get_model_terms(model)

    if contrast_predictor in terms.keys():
        term = terms.get(contrast_predictor)
        if hasattr(term, "components"):
            for component in term.components:
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]
                for name in names:
                    if name == contrast_predictor:
                        return component.kind
                        

def make_group_panel_values(
        data,
        data_dict, 
        main, 
        group, 
        panel, 
        kind,
        groups_n: int = 5
    ):
    """
    """
    
    # If available, obtain groups for grouping variable
    if group:
        group_values = make_group_values(data[group], groups_n)
        group_n = len(group_values)

    # If available, obtain groups for panel variable. Same logic than grouping applies
    if panel:
        panel_values = make_group_values(data[panel], groups_n)
        panel_n = len(panel_values)

    main_values = data_dict[main]
    main_n = len(main_values)

    # TO DO: is there a more concise way than logic and passing of
    # kind = ... argument?
    if kind == 'predictions':
        if group and not panel:
            main_values = np.tile(main_values, group_n)
            group_values = np.repeat(group_values, main_n)
            data_dict.update({main: main_values, group: group_values})
        elif not group and panel:
            main_values = np.tile(main_values, panel_n)
            panel_values = np.repeat(panel_values, main_n)
            data_dict.update({main: main_values, panel: panel_values})
        elif group and panel:
            if group == panel:
                main_values = np.tile(main_values, group_n)
                group_values = np.repeat(group_values, main_n)
                data_dict.update({main: main_values, group: group_values})
            else:
                main_values = np.tile(np.tile(main_values, group_n), panel_n)
                group_values = np.tile(np.repeat(group_values, main_n), panel_n)
                panel_values = np.repeat(panel_values, main_n * group_n)
                data_dict.update({main: main_values, group: group_values, panel: panel_values})
    else:
        if group and not panel:
            data_dict.update({group: group_values})

    return data_dict


def set_default_values(model, data, data_dict: dict, kind: str) -> pd.DataFrame:
    """
    """
    terms = get_model_terms(model)

    # Get default values for each variable in the model
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]
                for name in names:
                    if name not in data_dict:
                        # For numeric predictors, select the mean.
                        if component.kind == "numeric":
                            data_dict[name] = np.mean(data[name])
                        # For categoric predictors, select the most frequent level.
                        elif component.kind == "categoric":
                            data_dict[name] = mode(data[name])

    if kind == 'comparison':
        # if value in dict is not a list then convert to a list
        for key, value in data_dict.items():
            if not isinstance(value, (list, np.ndarray)):
                data_dict[key] = [value]
        return data_dict
    elif kind == 'predictions':
        return data_dict


def set_default_contrast_values(model, data, contrast_predictor):
    """
    """

    def _numeric_difference(x, kind: str = 'centered'):
        """
        """
        return [x - 0.5, x + 0.5]

    def _categoric_difference(x: np.ndarray, kind: str = 'pairwise'):
        """
        """
        return list(itertools.combinations(x, 2))
    

    terms = get_model_terms(model)

    # Get default values for each variable in the model
    # if contrast_predictor in terms.keys():
    #     term = terms.get(contrast_predictor)
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]
                for name in names:
                    if name == contrast_predictor:
                        # For numeric predictors, select the mean.
                        if component.kind == "numeric":
                            contrast = _numeric_difference(np.mean(data[name]))
                        # For categoric predictors, select the most frequent level.
                        elif component.kind == "categoric":
                            # contrast = _categoric_difference(
                            #     get_unique_levels(data[name])
                            # )
                            contrast = get_unique_levels(data[name])

    return contrast


def make_main_values(x, grid_n: int = 200, groups_n: int = 5) -> np.ndarray:
    """
    """
    if is_numeric_dtype(x):
        return np.linspace(np.min(x), np.max(x), grid_n)
    elif is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    raise ValueError("Main covariate must be numeric or categoric.")


def make_group_values(x, groups_n: int = 5) -> np.ndarray:
    """
    """
    if is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    elif is_numeric_dtype(x):
        return np.quantile(x, np.linspace(0, 1, groups_n))
    raise ValueError("Group covariate must be numeric or categoric.")


def get_unique_levels(x) -> Union[list, np.ndarray]:
    """
    """
    if hasattr(x, "dtype") and hasattr(x.dtype, "categories"):
        levels = list(x.dtype.categories)
    else:
        levels = np.unique(x)
    return levels


def get_group_offset(n, lower: float = 0.05, upper: float = 0.4) -> np.ndarray:
    # Complementary log log function, scaled.
    # See following code to have an idea of how this function looks like
    # lower, upper = 0.05, 0.4
    # x = np.linspace(2, 9)
    # y = get_group_offset(x, lower, upper)
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(x, y)
    # ax.axvline(2, color="k", ls="--")
    # ax.axhline(lower, color="k", ls="--")
    # ax.axhline(upper, color="k", ls="--")
    intercept, slope = 3.25, 1
    return lower + np.exp(-np.exp(intercept - slope * n)) * (upper - lower)


def identity(x):
    return x
