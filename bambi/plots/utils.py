from statistics import mode

import numpy as np
import pandas as pd
import itertools
from formulae.terms.call import Call
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi.utils import listify, get_aliased_name, clean_formula_lhs


def get_unique_levels(x):
    if hasattr(x, "dtype") and hasattr(x.dtype, "categories"):
        levels = list(x.dtype.categories)
    else:
        levels = np.unique(x)
    return levels


def get_group_offset(n, lower=0.05, upper=0.4):
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


class CreateData:
    """
    Class to create data for `plot_cap`, `plot_comparisons`, 
    and `plot_slopes`.
    """

    def __init__(self, model, covariates):
        self.model = model
        self.covariates = covariates
        self.data = model.data
    
    def _get_covariates(self, covariates: dict) -> tuple:

        main = covariates.get("horizontal")
        group = covariates.get("color", None)
        panel = covariates.get("panel", None)

        return (main, group, panel)


    def _make_main_values(self, x, grid_n):
        if is_numeric_dtype(x):
            return np.linspace(np.min(x), np.max(x), grid_n)
        elif is_string_dtype(x) or is_categorical_dtype(x):
            return np.unique(x)
        raise ValueError("Main covariate must be numeric or categoric.")
    

    def _set_default_values(self, data_dict: dict, type: str):
        """
        """

        terms = {}
        for component in self.model.distributional_components.values():
            if component.design.common:
                terms.update(component.design.common.terms)

            if component.design.group:
                terms.update(component.design.group.terms)

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
                                data_dict[name] = np.mean(self.data[name])
                            # For categoric predictors, select the most frequent level.
                            elif component.kind == "categoric":
                                data_dict[name] = mode(self.data[name])

        if type == 'comparison':
            # if value in dict is not a list then convert to a list
            for key, value in data_dict.items():
                if not isinstance(value, (list, np.ndarray)):
                    data_dict[key] = [value]
            return data_dict
        elif type == 'predictions':
            return pd.DataFrame(data_dict)
        else:
            raise ValueError("type must be 'comparison', 'predictions', or 'slopes'")
    
    
    def cap_data(self, main, group, panel):
        pass


    def comparisons_data(
            self,
            contrast_predictor, 
            conditional, 
            user_passed,
            grid_n=200
        ):
        """
        """

        main, group, panel = self._get_covariates(conditional)

        model_covariates = clean_formula_lhs(str(self.model.formula.main)).strip()
        model_covariates = model_covariates.split(" ")
        
        # check if contrast_predictor and conditional are instances of type dict
        if isinstance(contrast_predictor, dict) and isinstance(conditional, dict):
            if user_passed:
                data_dict = {**conditional}
            else:
                main_values = self._make_main_values(self.data[main], grid_n)
                data_dict = {main: main_values}
        
        # TO DO: remove hard coding of index
        if isinstance(contrast_predictor, dict):
            main_predictor = list(contrast_predictor.keys())[0] 
            contrast = list(contrast_predictor.values())[0]
            data_dict[main_predictor] = contrast
        elif isinstance(contrast_predictor, list):
            print("default")
        elif not isinstance(contrast_predictor, (list, dict, str)):
            raise TypeError("focal must be a list, dict, or string")
        

        comparison_data = self._set_default_values(data_dict, type='comparison')
        # Use cartesian product (cross join) to create contrasts
        keys, values = zip(*comparison_data.items())
        comparisons_df = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(comparisons_df)