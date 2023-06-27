from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd

from bambi.models import Model
from bambi.utils import clean_formula_lhs
from bambi.plots.utils import (
    ComparisonInfo,
    ContrastInfo,
    enforce_dtypes,
    get_covariates,
    get_model_covariates,
    make_group_panel_values,
    make_main_values,
    set_default_contrast_values,
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


def create_comparisons_data(comparisons: ComparisonInfo, user_passed: bool = False):
    """Create data for a Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    comparisons : ComparisonInfo
        The name of the predictor to be used in the comparisons.
    user_passed : bool, optional
        Whether the user passed data to the model. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Comparisons dataframe and or
        plotting.
    """

    def grid_level(comparisons: ComparisonInfo, contrast: ContrastInfo):
        """
        """
        covariates = get_covariates(comparisons.conditional)
        model_covariates = clean_formula_lhs(str(comparisons.model.formula.main)).strip()
        model_covariates = model_covariates.split(" ")

        # if user passed data, then only need to compute default values for
        # unspecified covariates in the model
        if user_passed:
            data_dict = {**comparisons.conditional}
        else:
            # if user did not pass data, then compute default values for the
            # covariates specified in the `conditional` arg.
            main_values = make_main_values(comparisons.model.data[covariates.main])
            data_dict = {covariates.main: main_values}
            data_dict = make_group_panel_values(
                comparisons.model.data, 
                data_dict, 
                covariates.main, 
                covariates.group, 
                covariates.panel, 
                kind="comparison"
            )

        data_dict[contrast.name] = contrast.values
        comparison_data = set_default_values(comparisons.model, data_dict, kind="comparison")
        # use cartesian product (cross join) to create contrasts
        keys, values = zip(*comparison_data.items())
        contrast_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return enforce_dtypes(comparisons.model.data, pd.DataFrame(contrast_dict))


    def unit_level(comparisons: ComparisonInfo, contrast: ContrastInfo):
        """
        """
        covariates = get_model_covariates(comparisons.model)
        df = comparisons.model.data[covariates].drop(labels=contrast.name, axis=1)

        contrast_vals = np.array(contrast.values)[..., None]
        contrast_vals = np.repeat(contrast_vals, comparisons.model.data.shape[0], axis=1)

        contrast_df_dict = {}
        for idx, value in enumerate(contrast_vals):
            contrast_df_dict[f"contrast_{idx}"] = df.copy()
            contrast_df_dict[f"contrast_{idx}"][contrast.name] = value

        return pd.concat(contrast_df_dict.values())  


    contrast = ContrastInfo(comparisons.contrast_predictor, comparisons.model)

    if not comparisons.conditional:
        df = unit_level(comparisons, contrast)
    else:
        df = grid_level(comparisons, contrast)
    
    return df
