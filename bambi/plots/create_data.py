from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd

from bambi.models import Model
from bambi.utils import clean_formula_lhs
from bambi.plots.utils import (
    ConditionalInfo,
    ContrastInfo,
    enforce_dtypes,
    get_covariates,
    get_model_covariates,
    make_group_panel_values,
    make_main_values,
    set_default_values,
)


def create_cap_data(model: Model, covariates: dict) -> pd.DataFrame:
    """Create data for a Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    covariates : dict
        A dictionary of length between one and three.
        Keys must be taken from ("horizontal", "color", "panel").
        The values indicate the names of variables.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Predictions dataframe and or
        plotting.
    """
    data = model.data
    covariates = get_covariates(covariates)
    main, group, panel = covariates.main, covariates.group, covariates.panel

    # Obtain data for main variable
    main_values = make_main_values(data[main])
    data_dict = {main: main_values}

    # Obtain data for group and panel variables if not None
    data_dict = make_group_panel_values(data, data_dict, main, group, panel, kind="predictions")
    data_dict = set_default_values(model, data_dict, kind="predictions")
    return enforce_dtypes(data, pd.DataFrame(data_dict))


def create_comparisons_data(
        condition: ConditionalInfo,
        contrast: ContrastInfo, 
        user_passed: bool = False
    ) -> pd.DataFrame:
    """Create data for a Conditional Adjusted Comparisons

    Parameters
    ----------
    comparisons : ComparisonInfo
        An dataclass instance containing the model, contrast, and conditional
        covariates to be used in the comparisons.
    user_passed : bool, optional
        Whether the user passed their own 'conditional' data to determine the 
        conditional data. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Comparisons dataframe and or
        plotting.
    """

    def _grid_level(condition: ConditionalInfo, contrast: ContrastInfo):
        """
        Creates the data for grid-level contrasts by using the covariates passed
        into the `conditional` arg. Values for the grid are either: (1) computed
        using a equally spaced grid, mean, and or mode (depending on the covariate
        dtype), and (2) a user specified value or range of values. 
        """
        covariates = get_covariates(condition.conditional)

        if user_passed:
            data_dict = {**condition.conditional}
        else:
            main_values = make_main_values(condition.model.data[covariates.main])
            data_dict = {covariates.main: main_values}
            data_dict = make_group_panel_values(
                condition.model.data, 
                data_dict, 
                covariates.main, 
                covariates.group, 
                covariates.panel, 
                kind="comparison"
            )

        data_dict[contrast.name] = contrast.values
        comparison_data = set_default_values(condition.model, data_dict, kind="comparison")
        # use cartesian product (cross join) to create contrasts
        keys, values = zip(*comparison_data.items())
        contrast_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return enforce_dtypes(condition.model.data, pd.DataFrame(contrast_dict))


    def _unit_level(comparisons: ConditionalInfo, contrast: ContrastInfo):
        """
        Creates the data for unit-level contrasts by using the observed (empirical)
        data. All covariates in the model are included in the data, except for the
        contrast predictor. The contrast predictor is replaced with either: (1) the
        default contrast value, or (2) the user specified contrast value.
        """
        covariates = get_model_covariates(contrast.model)
        df = contrast.model.data[covariates].drop(labels=contrast.name, axis=1)

        contrast_vals = np.array(contrast.values)[..., None]
        contrast_vals = np.repeat(contrast_vals, contrast.model.data.shape[0], axis=1)

        contrast_df_dict = {}
        for idx, value in enumerate(contrast_vals):
            contrast_df_dict[f"contrast_{idx}"] = df.copy()
            contrast_df_dict[f"contrast_{idx}"][contrast.name] = value

        return pd.concat(contrast_df_dict.values())


    if not condition.conditional:
        df = _unit_level(condition, contrast)
    else:
        df = _grid_level(condition, contrast)
    
    return df
 