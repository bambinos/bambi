import pandas as pd
import itertools

import bambi as bmb
from bambi.utils import clean_formula_lhs
from bambi.plots.utils import enforce_dtypes, make_group_panel_values, \
    set_default_values, make_main_values, get_covariates, set_default_contrast_values, \
    Comparison


def create_cap_data(
          model: bmb.Model,
          covariates: dict,
          grid_n: int = 200       
) -> pd.DataFrame:
    """Create data for a Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    covariates : dict
        A dictionary of length between one and three.
        Keys must be taken from ("horizontal", "color", "panel").
        The values indicate the names of variables.
    grid_n : int, optional
        The number of points used to evaluate the main covariate. Defaults to 200.
    groups_n : int, optional
        The number of groups to create when the grouping variable is numeric. Groups are based on
        equally spaced points. Defaults to 5.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Predictions plot.

    Raises
    ------
    ValueError
        When the number of covariates is larger than 2.
        When either the main or the group covariates are not numeric or categoric.
    """
    data = model.data
    covariates = get_covariates(covariates)
    main, group, panel = covariates.main, covariates.group, covariates.panel

    # Obtain data for main variable
    main_values = make_main_values(data[main], grid_n)
    data_dict = {main: main_values}

    # Obtain data for group and panel variables if not None
    data_dict = make_group_panel_values(data, data_dict, main, group, panel, kind="predictions")
    data_dict = set_default_values(model, data, data_dict, kind="predictions")
    return enforce_dtypes(data, pd.DataFrame(data_dict))


def create_comparisons_data(
            comparisons: Comparison,
            user_passed: bool = False,
            grid_n: int = 200
) -> pd.DataFrame:
    """Create data for a Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    contrast_predictor : Union[list, dict, str]
        The name of the predictor to be used in the comparisons.
    conditional : Union[list, dict, str]
        A dictionary of length between one and three.
        Keys must be taken from ("horizontal", "color", "panel").
        The values indicate the names of variables.
    user_passed : bool, optional
        Whether the user passed data to the model. Defaults to False.
    grid_n : int, optional
        The number of points used to evaluate the main covariate. Defaults to 200.
    groups_n : int, optional
        The number of groups to create when the grouping variable is numeric. Groups are based on
        equally spaced points. Defaults to 5.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Predictions plot.

    Raises
    ------
    ValueError
        When the number of covariates is larger than 2.
        When either the main or the group covariates are not numeric or categoric.
    """

    model, contrast_predictor, conditional = (
        comparisons.model, 
        comparisons.contrast_predictor, 
        comparisons.conditional
    )
    
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
        # if user did not pass data, then compute default values for the
        # covariates specified in the `conditional` arg.
        main_values = make_main_values(data[main], grid_n)
        data_dict = {main: main_values}
        data_dict = make_group_panel_values(
            data, data_dict, main, group, panel, kind='comparison'
            )

    # use key. value pairs to specify the contrast name and value
    if isinstance(contrast_predictor, dict):
        contrast_name = list(contrast_predictor.keys())[0] 
        contrast_vals = list(contrast_predictor.values())[0]
        data_dict[contrast_name] = contrast_vals
    # obtain default values for the contrast predictor if list or str
    elif isinstance(contrast_predictor, (list, str)):
        if isinstance(contrast_predictor, list):
            contrast_predictor = ' '.join(contrast_predictor)
        data_dict[contrast_predictor] = set_default_contrast_values(
            model, data, contrast_predictor
        )
    elif not isinstance(contrast_predictor, (list, dict, str)):
        raise TypeError("`contrast_predictor` must be a list, dict, or string")
    
    comparison_data = set_default_values(model, data, data_dict, kind='comparison')
    # use cartesian product (cross join) to create contrasts
    keys, values = zip(*comparison_data.items())
    contrast_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return enforce_dtypes(data, pd.DataFrame(contrast_dict))
