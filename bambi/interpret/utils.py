# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass, field
import re
from statistics import mode
from typing import Union

import numpy as np
from formulae.terms.call import Call
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
import xarray as xr

from bambi import Model
from bambi.utils import listify


@dataclass
class VariableInfo:
    """
    Stores information about the variable (covariate) passed into the 'contrast'
    or 'wrt' argument for comparisons and slopes. Depending on the effect type
    ('slopes' or 'comparisons'), the values attribute is either: (1) the values
    passed with the 'contrast' or 'wrt' argument, or (2) default are values
    computed by calling 'set_default_variable_values()'.

    'VariableInfo' is used to create 'slopes' and 'comparisons' data as well as
    computing the estimates for the 'slopes' and 'comparisons' effects.

    Parameters
    ----------
    model : Model
        The bambi Model object.
    variable : str, dict, or list
        The variable of interest passed by the user. `contrast` if 'comparisons'
        and `wrt` if 'slopes'.
    kind : str
        The effect type. Either 'slopes' or 'comparisons'.
    grid : bool or None, optional
        Whether a grid of pairwise values should be used as the data for generating
        predictions. Defaults to False.
    eps : float or None, optional
        The epsilon value used to add to the variable of interest's values when
        computing the 'slopes' effect. Defualts to None.
    user_passed : bool, optional
        Whether the user passed their own values for the variable of interest.
        Defaults to False.
    """

    model: Model
    variable: Union[str, dict, list]
    kind: str
    grid: Union[bool, None] = False
    eps: Union[float, None] = None
    user_passed: bool = False
    name: str = field(init=False)
    values: Union[int, float] = field(init=False)
    passed_values: int = field(init=False)

    def __post_init__(self):
        """
        Sets the name and values attributes based on the the effect type
        ('slopes' or 'comparisons'), if the user provided their own values,
        and dtype of the 'variable'.
        """
        if isinstance(self.variable, dict):
            self.user_passed = True
            self.passed_values = np.array(list(self.variable.values())[0])
            self.values = self.passed_values
            if self.kind == "slopes":
                self.values = self.epsilon_difference(self.passed_values, self.eps)
                if self.values.ndim > 1:
                    self.values = self.values.flatten()
            self.name = list(self.variable.keys())[0]
        elif isinstance(self.variable, (list, str)):
            self.name = self.variable
            if isinstance(self.variable, list):
                self.name = " ".join(self.variable)
            self.values = self.set_default_variable_values()
        elif not isinstance(self.variable, (list, dict, str)):
            raise TypeError("`variable` must be a list, dict, or string")

    def centered_difference(self, x, eps, dtype) -> np.ndarray:
        return np.array([x - eps, x + eps], dtype=dtype)

    def epsilon_difference(self, x, eps) -> np.ndarray:
        return np.array([x, x + eps])

    def set_default_variable_values(self) -> np.ndarray:
        """
        Returns default values for the variable of interest ('contrast' and 'wrt')
        for the 'slopes' and 'comparisons' effects depending on the dtype of the
        variable of interest, effect type, and if self.grid is True. The scenarios
        are described below:

        If numeric dtype and kind is 'comparisons', the returned value is a
        centered difference based on the mean of `variable'.

        If numeric dtype and kind is 'slopes', the returned value is an epsilon
        difference based on the mean of `variable'.

        If categoric dtype the returned value is the unique levels of `variable'.
        """
        terms = get_model_terms(self.model)
        # get default values for each variable in the model
        for term in terms.values():
            if hasattr(term, "components"):
                for component in term.components:
                    # if the component is a function call, use the argument names
                    if isinstance(component, Call):
                        names = [arg.name for arg in component.call.args]
                    else:
                        names = [component.name]
                    for name in names:
                        if name == self.name:
                            predictor_data = self.model.data[name]
                            dtype = predictor_data.dtype
                            if component.kind == "numeric":
                                if self.grid or self.kind == "comparisons":
                                    predictor_data = np.mean(predictor_data)
                                if self.kind == "slopes":
                                    values = self.epsilon_difference(predictor_data, self.eps)
                                elif self.kind == "comparisons":
                                    values = self.centered_difference(
                                        predictor_data, self.eps, dtype
                                    )
                            elif component.kind == "categoric":
                                values = np.unique(predictor_data)

        return values


@dataclass
class ConditionalInfo:
    """
    Stores information about the conditional (covariates) passed into the
    'conditional' argument for 'comparisons' and 'slopes' effects.

    'ConditionalInfo' is used to create 'slopes' and 'comparisons' data as well
    as computing the estimates for the 'slopes' and 'comparisons' effects.

    Parameters
    ----------
    model : bambi.Model
        The bambi model object.
    conditional : str, dict, or list
        The covariate(s) specified by the user to condition on.
    """

    model: Model
    conditional: Union[str, dict, list, None]
    covariates: dict = field(init=False)
    user_passed: bool = field(init=False)

    def __post_init__(self):
        """
        Sets the covariates attributes based on if the user passed a dictionary
        or not.
        """
        covariate_kinds = ("main", "group", "panel")

        if not isinstance(self.conditional, dict):
            self.covariates = listify(self.conditional)
            self.covariates = dict(zip(covariate_kinds, self.covariates))
            self.user_passed = False
        elif isinstance(self.conditional, dict):
            self.covariates = dict(zip(covariate_kinds, self.conditional))
            self.user_passed = True


@dataclass
class Covariates:
    """
    Stores the 'main', 'group', and 'panel' covariates from the 'conditional'
    argument in 'slopes' and 'comparisons'.
    """

    main: str
    group: Union[str, None]
    panel: Union[str, None]


def average_over(data: pd.DataFrame, covariate: Union[str, list]) -> pd.DataFrame:
    """
    Average estimates by specified covariate in the model. data.columns[-3:] are
    the columns: 'estimate', 'lower', and 'upper'.
    """
    if covariate == "all":
        return pd.DataFrame(data[data.columns[-3:]].mean()).T
    else:
        return data.groupby(covariate, as_index=False)[data.columns[-3:]].mean()


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
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    covariates.append([arg.name for arg in component.call.args])
                else:
                    covariates.append([component.name])
        elif hasattr(term, "factor"):
            covariates.append(list(term.var_names))

    flatten_covariates = [item for sublist in covariates for item in sublist]

    return np.unique(flatten_covariates)


def get_covariates(covariates: dict) -> Covariates:
    """
    Obtain the main, group, and panel covariates from the user's
    conditional dict.
    """
    covariate_kinds = ("main", "group", "panel")
    if any(key in covariate_kinds for key in covariates.keys()):
        # default if user did not pass their own conditional dict
        main = covariates.get("main")
        group = covariates.get("group", None)
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


def enforce_dtypes(
    observed_df: pd.DataFrame, new_df: pd.DataFrame, except_col=None
) -> pd.DataFrame:
    """
    Enforce dtypes of the observed data to the new data.
    """
    observed_dtypes = observed_df.dtypes
    for col in new_df.columns:
        if col in observed_dtypes.index and not except_col:
            if observed_dtypes[col] == "category":
                # explicitly converts to category dtype
                new_df[col] = new_df[col].astype("category")
            else:
                # casts the original dtype to the new data
                new_df[col] = new_df[col].astype(observed_dtypes[col])

    return new_df


def make_group_panel_values(
    data: pd.DataFrame,
    data_dict: dict,
    main: str,
    group: Union[str, None],
    panel: Union[str, None],
    kind: str,
    groups_n: int = 5,
) -> dict:
    """
    Compute group and panel values based on original data.
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

    if kind == "predictions":
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
    elif kind in ("comparisons", "slopes"):
        # for comparisons and slopes, we need unique values for numeric and categorical
        # group/panel covariates since we iterate over pairwise combinations of values
        if group and not panel:
            data_dict.update({group: np.unique(group_values)})
        elif group and panel:
            data_dict.update({group: np.unique(group_values), panel: np.unique(panel_values)})

    return data_dict


def set_default_values(model: Model, data_dict: dict, kind: str) -> dict:
    """
    Set default values for each variable in the model if the user did not
    pass them in the data_dict.
    """
    assert kind in (
        "comparisons",
        "predictions",
        "slopes",
    ), "kind must be either 'comparisons', 'slopes', or 'predictions'"

    unique_covariates = get_model_covariates(model)
    for name in unique_covariates:
        if name not in data_dict:
            dtype = str(model.data[name].dtype)
            if re.match(r"float*|int*", dtype):
                data_dict[name] = np.mean(model.data[name])
            elif dtype in ("category", "dtype"):
                data_dict[name] = mode(model.data[name])

    if kind in ("comparisons", "slopes"):
        # if value in dict is not a list then convert to a list
        for key, value in data_dict.items():
            if not isinstance(value, (list, np.ndarray)):
                data_dict[key] = [value]
        return data_dict

    return data_dict


def make_main_values(x: np.ndarray, grid_n: int = 50) -> np.ndarray:
    """
    Compute main values based on original data using a grid of evenly spaced
    values for numeric predictors and unique levels for categoric predictors.
    """
    if is_numeric_dtype(x):
        return np.linspace(np.min(x), np.max(x), grid_n)
    elif is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    raise ValueError("Main covariate must be numeric or categoric.")


def make_group_values(x: np.ndarray, groups_n: int = 5) -> np.ndarray:
    """
    Compute group values based on original data using unique levels for
    categoric predictors and quantiles for numeric predictors.
    """
    if is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    elif is_numeric_dtype(x):
        return np.quantile(x, np.linspace(0, 1, groups_n))
    raise ValueError("Group covariate must be numeric or categoric.")


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


def merge(y_hat_mean: xr.DataArray, y_hat_bounds: xr.DataArray, data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predictions ('y_hat_mean' and 'y_hat_bounds') into dataframes and join
    with the original data used to perform predictions. This will "duplicate" the
    data to ensure that the original data is aligned with each response dimension
    (level).
    """

    idx_names = y_hat_mean.to_dataframe().index.names

    yhat_df = y_hat_mean.to_dataframe().reset_index().set_index(idx_names)
    lower_df = y_hat_bounds.sel(hdi="lower").to_dataframe().reset_index().set_index(idx_names)
    higher_df = y_hat_bounds.sel(hdi="higher").to_dataframe().reset_index().set_index(idx_names)
    bounds_df = pd.merge(left=lower_df, right=higher_df, left_index=True, right_index=True)
    preds_df = (
        pd.merge(left=yhat_df, right=bounds_df, left_index=True, right_index=True)
        .reset_index()
        .set_index(idx_names[0])
    )

    summary_df = pd.merge(left=data, right=preds_df, left_index=True, right_index=True)

    return summary_df.drop(columns=["hdi_x", "hdi_y"])
