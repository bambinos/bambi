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
    
    
    def enforce_dtypes(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Enforce dtypes of the original data to the new data.
        """
        
        observed_dtypes = self.model.data.dtypes
        for col in df.columns:
            if col in observed_dtypes.index:
                df[col] = df[col].astype(observed_dtypes[col])
        return df
    
    
    def _get_covariates(self, covariates: dict) -> tuple:

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

        return (main, group, panel)


    def _make_main_values(self, x, grid_n=200, groups_n=5):
        if is_numeric_dtype(x):
            return np.linspace(np.min(x), np.max(x), grid_n)
        elif is_string_dtype(x) or is_categorical_dtype(x):
            return np.unique(x)
        raise ValueError("Main covariate must be numeric or categoric.")
    

    def make_group_values(self, x, groups_n=5):
        if is_string_dtype(x) or is_categorical_dtype(x):
            return np.unique(x)
        elif is_numeric_dtype(x):
            return np.quantile(x, np.linspace(0, 1, groups_n))
        raise ValueError("Group covariate must be numeric or categoric.")
    
    
    def _make_group_panel_values(
            self, 
            data_dict, 
            main, 
            group, 
            panel, 
            kind,
            groups_n=5
        ):

        # If available, obtain groups for grouping variable
        if group:
            group_values = self.make_group_values(self.data[group], groups_n)
            group_n = len(group_values)

        # If available, obtain groups for panel variable. Same logic than grouping applies
        if panel:
            panel_values = self.make_group_values(self.data[panel], groups_n)
            panel_n = len(panel_values)

        main_values = data_dict[main]
        main_n = len(main_values)

        # TO DO: is there a more concise way than logic and passing of
        # kind = ... argument?
        if kind == 'prediction':
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
        """
        TO DO: implement this method
        """
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
        print(f"main: {main}, group: {group}, panel: {panel}")

        model_covariates = clean_formula_lhs(str(self.model.formula.main)).strip()
        model_covariates = model_covariates.split(" ")
        
        if isinstance(contrast_predictor, dict) and isinstance(conditional, dict):
            # if user passed data, then only to compute default values for 
            # unspecified covariates in the model
            if user_passed:
                data_dict = {**conditional}
            else:
                # if user did not pass data, then compute default values
                main_values = self._make_main_values(self.data[main], grid_n)
                data_dict = {main: main_values}
                data_dict = self._make_group_panel_values(
                    data_dict, main, group, panel, kind='comparison'
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
        
        comparison_data = self._set_default_values(data_dict, type='comparison')
        # use cartesian product (cross join) to create contrasts
        keys, values = zip(*comparison_data.items())
        contrast_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return self.enforce_dtypes(pd.DataFrame(contrast_dict))
