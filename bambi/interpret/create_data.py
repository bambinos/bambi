import itertools
from statistics import mode

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_categorical_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from bambi import Model
from bambi.interpret.utils import (
    ConditionalInfo,
    enforce_dtypes,
    get_model_covariates,
    VariableInfo,
)

from bambi.interpret.logs import log_interpret_defaults


def create_grid(condition, variable, **kwargs) -> pd.DataFrame:
    """Creates a grid of data by using the covariates passed into the
    'conditional' and 'variable' argument.

    Values for the grid are either:
        1.) computed using an equally spaced grid (`np.linspace`), mean, and or mode depending on
            the covariate dtype.
        2.) a user specified value or range of values.

    Parameters
    ----------
    condition : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.
    variable : VariableInfo, optional
        Information about the variable of interest. This is 'contrast' for
        'comparisons', 'wrt' for 'slopes', and 'None' for 'predictions'.
    **kwargs : dict
        Optional keywords specifying the type of grid to create 'grid_type'
        and or the effect type 'effect_type' being computed.

    Returns
    -------
    pd.DataFrame
        A dataframe containing pairwise combinations of values.
    """
    model, observed_data = condition.model, condition.model.data

    if condition.user_passed:
        # TODO: FIX THIS!!!
        # data_dict = {**condition.covariates}
        data_dict = {**condition.conditional}
    else:
        data_dict = {}
        # TODO: FIX THIS!!!
        # for covariate in condition.covariates:
        for covariate in condition.covariates.values():
            x = observed_data[covariate]

            if is_numeric_dtype(x) or is_float_dtype(x):
                values = np.linspace(np.min(x), np.max(x), 50)
            elif is_integer_dtype(x):
                values = np.quantile(x, np.linspace(0, 1, 5))
            elif is_categorical_dtype(x) or is_string_dtype(x) or is_object_dtype(x):
                values = np.unique(x)
            else:
                raise TypeError(
                    f"Unsupported data type of {x.dtype} for covariate '{covariate.name}'"
                )

            data_dict[covariate] = values

    if variable:
        data_dict[variable.name] = variable.values

    # Set typical values as defaults for unspecified covariates
    data_dict = set_default_values(model, data_dict)

    # TODO: expand() for 'predictions' so predictions data is not a pairwise grid
    # if grid_type == "expand":
    #     data_grid = _expand_grid()
    # else:
    data_grid = _pairwise_grid(data_dict)

    # Can't enforce dtype on 'with respect to' variable for 'slopes' as it
    # may remove floating point in the epsilon
    effect_kind = kwargs.get("effect_kind", None)
    if effect_kind == "slopes":
        except_col = variable.name
    else:
        except_col = None

    data_grid = enforce_dtypes(observed_data, data_grid, except_col)

    # After computing default values, fractional values may have been computed.
    # Enforcing the dtype of "int" may create duplicate rows as it will round
    # the fractional values.
    data_grid = data_grid.drop_duplicates()

    return data_grid.reset_index(drop=True)


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
    cross_joined_data = pd.DataFrame([dict(zip(keys, v)) for v in itertools.product(*values)])
    return cross_joined_data


def _differences_unit_level(variable_info: VariableInfo, effect_type: str) -> pd.DataFrame:
    """Creates the data for unit-level contrasts by using the observed (empirical)
    data. All covariates in the model are included in the data, except for the
    contrast predictor. The contrast predictor is replaced with either: (1) the
    default contrast value, or (2) the user specified contrast value.

    Parameters
    ----------
    variable_info : VariableInfo
        Information about the variable of interest. This is `contrast` for
        'comparisons' and `wrt` for 'slopes'.
    effect_type : str
        The type of effect being computed. Either "comparisons" or "slopes".

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

    if effect_type == "comparisons":
        variable_vals = np.array(variable_info.values)[..., None]
        variable_vals = np.repeat(variable_vals, variable_info.model.data.shape[0], axis=1)

    unit_level_df_dict = {}
    for idx, value in enumerate(variable_vals):
        unit_level_df_dict[f"contrast_{idx}"] = df.copy()
        unit_level_df_dict[f"contrast_{idx}"][variable_info.name] = value

    # After inserting the variable of interest's values, duplicate rows may have
    # been introduced if that value was already present in the data. Dropping
    # duplicates ensures that the data is the same length as the original data.
    unit_level_df = pd.concat(unit_level_df_dict.values()).drop_duplicates().reset_index(drop=True)

    return unit_level_df


def create_differences_data(
    condition_info: ConditionalInfo, variable_info: VariableInfo, effect_type: str
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
    effect_type : str
        The type of effect being computed. Either "comparisons" or "slopes".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data used to generate predictions. If no
        covariates were passed, then the original data used to fit the model
        is returned. Otherwise, a grid of values is created using the covariates
        passed into the `conditional` argument.
    """
    if not condition_info.covariates:
        return _differences_unit_level(variable_info, effect_type)

    return create_grid(condition_info, variable_info, effect_type=effect_type)


def create_predictions_data(condition_info: ConditionalInfo) -> pd.DataFrame:
    """Creates either unit level or grid level data for 'predictions' depending
    if the user passed covariates.

    Parameters
    ----------
    condition_info : ConditionalInfo
        Information about the conditional argument passed into the plot
        function.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data used to generate predictions. If no
        covariates were passed, then the original data used to fit the model
        is returned. Otherwise, a grid of values is created using the covariates
        passed into the `conditional` argument.
    """
    # Unit level data uses the observed (empirical) data
    if not condition_info.covariates:
        covariates = get_model_covariates(condition_info.model)
        return condition_info.model.data[covariates]

    return create_grid(condition_info, None)


@log_interpret_defaults
def set_default_values(model: Model, data_dict: dict) -> dict:
    """
    Set default values for each variable in the model if the user did not
    pass them in the data_dict.
    """
    # Set unspecified covariates to "typical" values
    unique_covariates = get_model_covariates(model)
    for name in unique_covariates:
        if name not in data_dict:
            x = model.data[name]
            if is_numeric_dtype(x) or is_integer_dtype(x) or is_float_dtype(x):
                data_dict[name] = np.array([np.mean(x)])
            elif is_categorical_dtype(x) or is_string_dtype(x) or is_object_dtype(x):
                data_dict[name] = np.array([mode(x)])
            else:
                raise TypeError(f"Unsupported data type of {x.dtype} for covariate '{name}'")

    return data_dict
