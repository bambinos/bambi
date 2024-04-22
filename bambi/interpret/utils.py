# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable

from bambi import Model
from bambi.utils import listify
from bambi.interpret.logs import log_interpret_defaults


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
    values: Union[int, float, np.ndarray] = field(init=False)
    passed_values: Union[int, float, np.ndarray] = field(init=False)

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

    @log_interpret_defaults
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
        values = None  # Otherwise pylint complains
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
                                if self.kind == "comparisons":
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
        Maps covariates to 'main', 'group', and 'panel' in the order they are passed
        to the 'conditional' argument.

        By default, the first three elements (covariates) are mapped to 'main', 'group',
        and 'panel'. If the user passes more than three covariates, the remaining
        are mapped to 'covariate_4', 'covariate_5', etc. to ensure they are
        not dropped due to non-unique keys.
        """
        covariate_kinds = ("main", "group", "panel")

        if not isinstance(self.conditional, dict):
            self.conditional = listify(self.conditional)
            covariate_names = self.conditional
            self.user_passed = False
        elif isinstance(self.conditional, dict):
            covariate_names = list(self.conditional.keys())
            for key, value in self.conditional.items():
                if not isinstance(value, (list, np.ndarray)):
                    self.conditional[key] = listify(value)

            # sort values b/c of matplotlib plotting behavior when calling `plot_categorical`
            self.conditional = {key: sorted(value) for key, value in self.conditional.items()}
            self.user_passed = True

        self.covariates = dict(zip(covariate_kinds, self.conditional))

        # adds unique keys to the covariates dict if the user passed more than three covariates
        extra_covariates = covariate_names[len(covariate_kinds) :]
        if extra_covariates:
            for index, extra in enumerate(extra_covariates, start=1):
                self.covariates[f"covariate_{index}"] = extra


@dataclass
class Covariates:
    """
    Stores the 'main', 'group', and 'panel' covariates from the 'conditional'
    argument in 'plot_comparisons', 'plot_predictions', 'plot_slopes'.
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
                # if the component is a function call, look for relevant argument names
                if isinstance(component, Call):
                    # Add variable names passed as unnamed arguments
                    covariates.append(
                        [arg.name for arg in component.call.args if isinstance(arg, LazyVariable)]
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


def get_group_offset(n, lower: float = 0.05, upper: float = 0.4) -> np.ndarray:
    """
    When plotting categorical variables, this function computes the offset of the
    stripplot points based on the number of groups ``n``.
    """
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
