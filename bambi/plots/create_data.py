from dataclasses import dataclass
from statistics import mode
from typing import Callable, Union

import numpy as np
import pandas as pd
import itertools
from formulae.terms.call import Call
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

import bambi as bmb
from bambi.utils import clean_formula_lhs
from bambi.plots.utils import enforce_dtypes, make_group_panel_values, \
    set_default_values, make_main_values, make_group_values, get_covariates, \
    get_unique_levels, get_group_offset 



def create_comparisons_data(
            model: bmb.Model,
            contrast_predictor: Union[list, dict, str], 
            conditional: Union[list, dict, str],
            user_passed: bool = False,
            grid_n: int = 200
        ):
        """
        """
        
        data = model.data
        covariates = get_covariates(conditional)
        main, group, panel = covariates.main, covariates.group, covariates.panel

        model_covariates = clean_formula_lhs(str(model.formula.main)).strip()
        model_covariates = model_covariates.split(" ")
        
        # if user passed data, then only need to compute default values for 
        # unspecified covariates in the model
        if user_passed:
            data_dict = {**conditional}
        else:
            # if user did not pass data, then compute default values
            main_values = make_main_values(data[main], grid_n)
            data_dict = {main: main_values}
            data_dict = make_group_panel_values(
                data, data_dict, main, group, panel, kind='comparison'
                )
        
        # TO DO: remove hard coding of index? (it seems to work though)
        if isinstance(contrast_predictor, dict):
            main_predictor = list(contrast_predictor.keys())[0] 
            contrast = list(contrast_predictor.values())[0]
            data_dict[main_predictor] = contrast
        elif isinstance(contrast_predictor, list):
            print("default")
        elif not isinstance(contrast_predictor, (list, dict, str)):
            raise TypeError("`contrast_predictor` must be a list, dict, or string")
        
        comparison_data = set_default_values(model, data, data_dict, kind='comparison')
        # use cartesian product (cross join) to create contrasts
        keys, values = zip(*comparison_data.items())
        contrast_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return enforce_dtypes(data, pd.DataFrame(contrast_dict))
