import itertools
from typing import Union

import numpy as np
import pandas as pd

from bambi.interpret.utils import (
    ConditionalInfo,
    enforce_dtypes,
    get_covariates,
    get_model_covariates,
    make_group_panel_values,
    make_main_values,
    set_default_values,
    VariableInfo,
)


def _pairwise_grid(data_dict: dict) -> pd.DataFrame:
    """Creates a pairwise grid (cartesian product) of data by using the
    key-values of the dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary containing the covariates as keys and their values as the
        values.

    Returns
    -------
    pd.DataFrame
        A dataframe containing values used as input to the fitted Bambi model to
        generate predictions.
    """
    keys, values = zip(*data_dict.items())
    data_grid = pd.DataFrame([dict(zip(keys, v)) for v in itertools.product(*values)])
    return data_grid


def _grid_level(
    condition_info: ConditionalInfo,
    variable_info: Union[VariableInfo, None],
    user_passed: bool,
    kind: str,
) -> pd.DataFrame:
    """Creates a "grid" of data by using the covariates passed into the
    `conditional` argument. Values for the grid are either: (1) computed
    using a equally spaced grid, mean, and or mode (depending on the
    covariate dtype), and (2) a user specified value or range of values.

    Parameters
    ----------
    condition_info : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.
    variable_info : VariableInfo, optional
        Information about the variable of interest. This is `contrast` for
        'comparisons', `wrt` for 'slopes', and `None` for 'predictions'.
    user_passed : bool
        Whether the user passed a value(s) for the `conditional` argument.
    kind : str
        The kind of effect being computed. Either "comparisons", "predictions",
        or "slopes".

    Returns
    -------
    pd.DataFrame
        A dataframe containing values used as input to the fitted Bambi model to
        generate predictions.
    """
    covariates = get_covariates(condition_info.covariates)

    if kind == "predictions":
        # Compute pairwise grid of values if the user passed a dict.
        if user_passed:
            data_dict = {**condition_info.conditional}
            data_dict = set_default_values(condition_info.model, data_dict, kind=kind)
            for key, value in data_dict.items():
                if not isinstance(value, (list, np.ndarray)):
                    data_dict[key] = [value]
            data_grid = _pairwise_grid(data_dict)
        else:
            # Compute a grid of values
            main_values = make_main_values(condition_info.model.data[covariates.main])
            data_dict = {covariates.main: main_values}
            data_dict = make_group_panel_values(
                condition_info.model.data,
                data_dict,
                covariates.main,
                covariates.group,
                covariates.panel,
                kind=kind,
            )
            data_dict = set_default_values(condition_info.model, data_dict, kind=kind)
            data_grid = pd.DataFrame(data_dict)
    else:
        # Compute pairwise grid of values if the user passed a dict.
        if user_passed:
            data_dict = {**condition_info.conditional}
        else:
            # Compute a grid of values
            main_values = make_main_values(condition_info.model.data[covariates.main])
            data_dict = {covariates.main: main_values}
            data_dict = make_group_panel_values(
                condition_info.model.data,
                data_dict,
                covariates.main,
                covariates.group,
                covariates.panel,
                kind=kind,
            )

        data_dict[variable_info.name] = variable_info.values
        data_dict = set_default_values(condition_info.model, data_dict, kind=kind)
        data_grid = _pairwise_grid(data_dict)

    # Can't enforce dtype on numeric 'wrt' for 'slopes 'as it may remove floating point epsilons
    except_col = None if kind in ("comparisons", "predictions") else {variable_info.name}
    data_grid = enforce_dtypes(condition_info.model.data, data_grid, except_col)

    # After computing default values, fractional values may have been computed.
    # Enforcing the dtype of "int" may create duplicate rows as it will round
    # the fractional values.
    data_grid = data_grid.drop_duplicates()

    return data_grid.reset_index(drop=True)


def _differences_unit_level(variable_info: VariableInfo, kind: str) -> pd.DataFrame:
    """Creates the data for unit-level contrasts by using the observed (empirical)
    data. All covariates in the model are included in the data, except for the
    contrast predictor. The contrast predictor is replaced with either: (1) the
    default contrast value, or (2) the user specified contrast value.

    Parameters
    ----------
    variable_info : VariableInfo
        Information about the variable of interest. This is `contrast` for
        'comparisons' and `wrt` for 'slopes'.
    kind : str
        The kind of effect being computed. Either "comparisons" or "slopes".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the unit-level data for the variable of interest
        value. This dataframe is the same length as the data used to fit the
        Bambi model.
    """
    covariates = get_model_covariates(variable_info.model)
    df = variable_info.model.data[covariates].drop(labels=variable_info.name, axis=1)

    variable_vals = variable_info.values

    if kind == "comparisons":
        variable_vals = np.array(variable_info.values)[..., None]
        variable_vals = np.repeat(variable_vals, variable_info.model.data.shape[0], axis=1)

    unit_level_df_dict = {}
    for idx, value in enumerate(variable_vals):
        unit_level_df_dict[f"contrast_{idx}"] = df.copy()
        unit_level_df_dict[f"contrast_{idx}"][variable_info.name] = value

    return pd.concat(unit_level_df_dict.values())


def create_differences_data(
    condition_info: ConditionalInfo, variable_info: VariableInfo, user_passed: bool, kind: str
) -> pd.DataFrame:
    """Creates either unit level or grid level data for 'comparisons' and 'slopes'
    depending if the user passed covariate values.

    Parameters
    ----------
    condition_info : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.
    variable_info : VariableInfo
        Information about the variable of interest. This is `contrast` for
        'comparisons' and `wrt` for 'slopes'.
    user_passed : bool
        Whether the user passed a value(s) for the `conditional` argument.
    kind : str
        The kind of effect being computed. Either "comparisons" or "slopes".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data used to generate predictions. If no
        covariates were passed, then the original data used to fit the model
        is returned. Otherwise, a grid of values is created using the covariates
        passed into the `conditional` argument.
    """

    if not condition_info.covariates:
        return _differences_unit_level(variable_info, kind)

    return _grid_level(condition_info, variable_info, user_passed, kind)


def create_predictions_data(condition_info: ConditionalInfo, user_passed: bool) -> pd.DataFrame:
    """Creates either unit level or grid level data for 'predictions' depending
    if the user passed covariates.

    Parameters
    ----------
    condition_info : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.
    user_passed : bool
        Whether the user passed a value(s) for the `conditional` argument.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data used to generate predictions. If no
        covariates were passed, then the original data used to fit the model
        is returned. Otherwise, a grid of values is created using the covariates
        passed into the `conditional` argument.
    """
    # Unit level data used the observed (empirical) data
    if not condition_info.covariates:
        covariates = get_model_covariates(condition_info.model)
        return condition_info.model.data[covariates]

    return _grid_level(condition_info, None, user_passed, "predictions")
