import numpy as np
import pandas as pd
import xarray as xr

from bambi import Model
from bambi.interpret.create_data import create_grid
from bambi.interpret.utils import ConditionalInfo, VariableInfo


def data_grid(
    model: Model,
    conditional: str | list | dict,
    variable: str | dict | None = None,
    effect_type: str | None = None,
    eps: float | None = None,
    **kwargs,
):
    """Create a pairwise grid of data using the covariates passed to the 'conditional' and optional
    'variable' argument. Covariates not passed to 'conditional', but are terms in the Bambi model,
    are set to typical values (e.g., mean or mode).

    Parameters
    ----------
    model : Model
        Bambi Model object.
    conditional : str, list, dict
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    variable: str, dict, optional
        The variable of interest. This is 'contrast' for 'comparisons', 'wrt' for 'slopes', and
        'None' for 'predictions'. If dict, keys are the covariate names and values are the values.
    effect_type : str, optional
        The type of effect the data may be used for. This argument is useful for if the data
        will be used to compute 'comparisons' or 'slopes' and a parameter is passed to 'variable'
        as it determines the default 'eps' value. Defaults to None.
    eps : float, optional
        The epsilon value used to compute 'comparisons' or 'slopes'. If 'effect_type' is True,
        'comparisons' defaults to `0.5` and 'slopes' defaults to `1e-4`.
    **kwargs : dict
        Optional keywords arguments passed to 'create_grid' to determine the number of values `num`
        to return when computing a `np.linspace` grid for default values.

    Returns
    -------
    pd.DataFrame
        A dataframe containing pairwise combinations of values based on the parameters
        passed into 'conditional' and 'variable'.

    Raises
    ------
    ValueError
        If 'variable' and 'effect_type' not in ["comparisons", "slopes", "predictions"].
    TypeError
        If 'conditional' is a dict and the values are not of type int, float, list, or np.ndarray.
        If 'conditional' is a list and the elements are not of type str.
        If 'variable' is a dict and there is more than one key.

    """
    if variable and effect_type not in ["comparisons", "slopes", "predictions"]:
        raise ValueError(
            "'If passing an argument to 'variable', the parameter 'effect_type' must be either "
            f"'comparisons' or 'slopes'. Received: {effect_type}"
        )

    if isinstance(conditional, dict):
        for value in conditional.values():
            if not isinstance(value, (int, float, list, np.ndarray)):
                raise TypeError(
                    "Dictionary values must be of type int, float, list, or np.ndarray. "
                    f"Received: {type(value)}"
                )

    if isinstance(conditional, list):
        for value in conditional:
            if not isinstance(value, str):
                raise TypeError(f"Elements of list must be of type str. Received: {type(value)}")

    conditional = ConditionalInfo(model, conditional)
    kwargs["effect_type"] = effect_type

    if variable:
        if isinstance(variable, dict):
            if len(variable) > 1:
                raise ValueError("Variable dictionary must have only one key.")

        if not eps and effect_type == "comparisons":
            eps = 0.5
        elif not eps and effect_type == "slopes":
            eps = 1e-4

        grid = bool(conditional.covariates)
        variable = VariableInfo(model, variable, kind=effect_type, eps=eps, grid=grid)

    return create_grid(conditional, variable, **kwargs)


def _prepare_idata(idata: "InferenceData", data: xr.Dataset) -> "InferenceData":
    """Prepare InferenceData object for use in `select_draws` by removing the
    'observed_data' group and replacing it with another 'data' group that contains
    the data used to generate predictions.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object containing the inference data after performing `model.predict`.
    data : xr.Dataset
        The Dataset passed as 'data' to `model.predict` to generate predictions.

    Returns
    -------
    InferenceData
        A new InferenceData object with the 'observed_data' group removed and
        replaced with a 'data' group that contains the 'data' used to generate
        predictions.

    Raises
    ------
    ValueError
        If the InferenceData object does not contain a 'data' or 'observed_data' group.
    """

    if "observed_data" in idata.groups():
        coordinate_name = list(idata["observed_data"].coords)
        del idata.observed_data
        idata["data"] = data
    else:
        raise ValueError("InferenceData object does not contain a 'data' or 'observed_data' group.")

    if len(coordinate_name) > 1:
        raise NotImplementedError("Only one coordinate is currently supported.")
    coordinate_name = coordinate_name[0]

    # rename index to match coordinate name in other InferenceData groups
    idata.data = idata.data.rename({"index": coordinate_name})
    return idata


def select_draws(
    idata: "InferenceData",
    data: pd.DataFrame,
    condition: dict,
    data_var: str,
    group: str = "posterior",
) -> xr.DataArray:
    """Select posterior or posterior predictive draws conditioned on the observation
    that produced that draw by passing a `condition` dictionary.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object containing the inference data after performing `model.predict`.
    data : pd.DataFrame
        The Dataframe passed as 'data' to `model.predict` to generate predictions.
    condition : dict
        Dictionary of variable names and values used to select draws.
    data_var : str
        Name of data variable in the 'group' to select draws from.
    group : str, optional
        Whether to select draws from the posterior or posterior predictive group.
        Defaults to 'posterior'.

    Returns
    -------
    xr.DataArray
        A DataArray containing the selected draws.

    Raises
    ------
    ValueError
        If 'condition' is an empty dictionary.
        If 'group' is not 'posterior' or 'posterior_predictive'.
        If the InferenceData object does not contain a 'group' group.
    """
    if not condition:
        raise ValueError("'condition' cannot be empty an empty dictionary")

    if group not in ["posterior", "posterior_predictive"]:
        raise ValueError("'group' must be either 'posterior' or 'posterior_predictive'")
    if group not in idata.groups():
        raise ValueError(f"InferenceData object does not contain a '{group}' group.")

    for key, value in condition.items():
        if isinstance(value, (list, np.ndarray)):
            raise ValueError(f"{key} condition value cannot be an array or list")

    idata = idata.copy()
    xr_df = xr.Dataset.from_dataframe(data)
    idata_new = _prepare_idata(idata, xr_df)
    coordinate_name = list(idata_new["data"].coords)[0]

    # indices of draws that satisfy condition
    condition_idx = np.where(
        np.logical_and.reduce([idata_new["data"][key] == value for key, value in condition.items()])
    )[0]
    draws = idata_new[group].isel({f"{coordinate_name}": condition_idx})[data_var]

    # for main and or parent parameters (e.g., distributional models)
    if coordinate_name in draws.coords:
        new_coords = np.arange(len(condition_idx))
        draws = draws.assign_coords({coordinate_name: new_coords})

    return draws
