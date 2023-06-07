# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from typing import Union, Callable, Tuple, Any

import arviz as az
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import bambi as bmb
from bambi.utils import listify, get_aliased_name
from bambi.plots.create_data import create_cap_data, create_comparisons_data
from bambi.plots.utils import identity




def comparisons(
        model: bmb.Model,
        idata: az.InferenceData,
        contrast_predictor: Union[str, dict, list],
        conditional: Union[str, dict, list],
        target: str = "mean",
        use_hdi: bool = True,
        hdi_prob=None,
        transforms=None,
    ) -> pd.DataFrame:
    """
    
    """

    covariate_kinds = ("horizontal", "color", "panel")
    # if not dict, then user did not pass values to condition on
    if not isinstance(conditional, dict):
        conditional = listify(conditional)
        conditional = dict(zip(covariate_kinds, conditional))
        comparisons_df = create_comparisons_data(
            model,
            contrast_predictor,
            conditional,
            user_passed=False
        )
    # if dict, user passed values to condition on
    elif isinstance(conditional, dict):
        comparisons_df = create_comparisons_data(
            model,
            contrast_predictor,
            conditional,
            user_passed=True
        )
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))

    # RE DO THIS
    if isinstance(contrast_predictor, dict):
        contrast_name, contrast = next(iter(contrast_predictor.items()))
    elif isinstance(contrast_predictor, list):
        contrast_name = contrast_predictor[0]
    elif isinstance(contrast_predictor, str):
        contrast_name = contrast_predictor
    
    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]
    
    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")
    
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response_transform = transforms.get(response_name, identity)
    response_preds_term = f"{response_name}_{target}_preds"

    # Perform predictions on new data
    idata = model.predict(idata, data=comparisons_df, inplace=False)
    y_hat = response_transform(idata.posterior[f"{response_name}_{target}"])
    y_hat_mean = y_hat.mean(("chain", "draw"))
    comparisons_df[response_preds_term] = y_hat_mean

    if use_hdi:
         y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{response_name}_{target}"].T

    # TO DO: rename using more informative names
    lower = f"{response_preds_term}_lower"
    upper = f"{response_preds_term}_higher"
    comparisons_df[lower] = y_hat_bounds[0]
    comparisons_df[upper] = y_hat_bounds[1]

    # obtain covariaties used in the model to perform group by operations
    model_covariates = list(
        comparisons_df.columns[~comparisons_df.columns
                               .isin([contrast_name, response_preds_term, lower, upper])]
                               )
    # TO DO: allow different types of comparisons other than difference (ratio, etc.)
    # compute difference between contrast predictions
    contrast_comparison = pd.DataFrame((comparisons_df
                           .groupby(model_covariates)[[response_preds_term, lower, upper]]
                           .diff()
                           .dropna()
                           .reset_index(drop=True)
                           ))
    
    main = conditional.get("horizontal")
    group = conditional.get("color")
    panel = conditional.get("panel")

    print(f"model covariates: {model_covariates}")

    # TO DO: create a utility function for building contrasts dataframe
    N = contrast_comparison.shape[0]
    if np.unique(comparisons_df[main]).shape[0] == 1:
        number_repeats = N
        contrast_comparison[main] = np.repeat(
            np.unique(comparisons_df[main]), number_repeats
        )
    else:
        main_values = np.unique(comparisons_df[main])
        main_n = len(main_values)
        number_repeats = N // main_n
        if is_numeric_dtype(comparisons_df[main]):
            X_unique = (comparisons_df[model_covariates]
                        .drop_duplicates()
                        .reset_index(drop=True)
            )
            contrast_comparison[main] = X_unique[main]
        else:
            values = np.repeat(main_values, number_repeats)
            contrast_comparison[main] = values

    if group and not panel:
        group_values = np.unique(comparisons_df[group])
        group_n = len(group_values)
        number_repeats = N // group_n
        values = np.tile(group_values, number_repeats)
        contrast_comparison[group] = values
    elif group and panel:
        raise UserWarning("Not implemented: TO DO!!!")

    # TO DO: better names and descriptive statistics
    contrast_comparison = contrast_comparison.rename(
        columns={
            f"{response_preds_term}": "contrast_comparison",
            f"{lower}": "contrast_comparison_lower",
            f"{upper}": "contrast_comparison_upper"
        }
    )
    return comparisons_df, contrast_comparison, idata
