import itertools

import numpy as np
import pandas as pd

from bambi.models import Model
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


def _grid_level(
    condition_info: ConditionalInfo, variable_info: VariableInfo, user_passed: bool, kind: str
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
        A dataframe containing a pairwise grid of values used as input to the
        fitted Bambi model to generate predictions.
    """
    covariates = get_covariates(condition_info.covariates)

    if user_passed:
        data_dict = {**condition_info.conditional}
    else:
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
    comparison_data = set_default_values(condition_info.model, data_dict, kind=kind)

    # use cartesian product (cross join) to create pairwise grid
    keys, values = zip(*comparison_data.items())
    pairwise_grid = pd.DataFrame([dict(zip(keys, v)) for v in itertools.product(*values)])

    # can't enforce dtype on numeric 'wrt' as it may remove floating point epsilons
    except_col = None if kind == "comparisons" else {variable_info.name}
    pairwise_grid = enforce_dtypes(condition_info.model.data, pairwise_grid, except_col)

    # After computing default values, fractional values may have been computed.
    # Enforcing the dtype of "int" may create duplicate rows as it will round
    # the fractional values.
    pairwise_grid = pairwise_grid.drop_duplicates()

    return pairwise_grid


# TODO: rename to _differences_unit_level???
def _unit_level(variable_info: VariableInfo, kind: str) -> pd.DataFrame:
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
        A dataframe containing the data used to generate predictions.
    """

    if not condition_info.covariates:
        return _unit_level(variable_info, kind)

    return _grid_level(condition_info, variable_info, user_passed, kind)


def create_predictions_data(condition_info: ConditionalInfo, model: Model) -> pd.DataFrame:
    """Creates either unit level or grid level data for 'predictions' depending
    if the user passed covariates.

    Parameters
    ----------
    condition_info : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.
    model : Model
        A fitted Bambi model.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data used to generate predictions.
    """

    if not condition_info.covariates:
        return model.data

    # TODO: move to _predictions_unit_level???
    data = model.data
    covariates = get_covariates(condition_info.covariates)
    main, group, panel = covariates.main, covariates.group, covariates.panel

    data_dict = {main: make_main_values(data[main])}
    data_dict.update(
        make_group_panel_values(data, data_dict, main, group, panel, kind="predictions")
    )
    data_dict = set_default_values(model, data_dict, kind="predictions")

    return enforce_dtypes(data, pd.DataFrame(data_dict))
