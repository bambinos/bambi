# pylint: disable=ungrouped-imports
from dataclasses import dataclass, field
import itertools
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype
import xarray as xr

from bambi.models import Model
from bambi.interpret.create_data import create_differences_data, create_predictions_data
from bambi.interpret.utils import (
    average_over,
    ConditionalInfo,
    enforce_dtypes,
    identity,
    merge,
    VariableInfo,
)
from bambi.utils import get_aliased_name


SUPPORTED_SLOPES = ("dydx", "eyex")
SUPPORTED_COMPARISONS = {
    "diff": lambda x, y: x - y,
    "ratio": lambda x, y: x / y,
}


@dataclass
class ResponseInfo:
    """Stores metadata about the response variable for indexing data in az.InferenceData,
    computing uncertainty intervals, and creating the summary dataframe in
    'PredictiveDifferences'.

    Parameters
    ----------
    model : Model
        The fitted bambi Model object.
    target : str
        The target of the response variable such as 'mean' or 'sigma'. Defaults to
        'mean'.
    lower_bound : float
        The percentile of the lower bound of the uncertainty interval. Defaults to 0.03.
    upper_bound : float
        The percentile of the upper bound of the uncertainty interval. Defaults to 0.97.
    """

    name: str
    target: Union[str, None] = None
    lower_bound: float = 0.03
    upper_bound: float = 0.97
    name_target: str = field(init=False)
    name_obs: str = field(init=False)
    lower_bound_name: str = field(init=False)
    upper_bound_name: str = field(init=False)

    def __post_init__(self):
        """
        Assigns commonly used f-strings for indexing and column names as attributes
        in building the summary dataframe in 'PredictiveDifferences'.
        """
        if self.target is None:
            self.name_target = self.name
        else:
            self.name_target = f"{self.name}_{self.target}"

        self.name_obs = f"{self.name}_obs"
        self.lower_bound_name = f"lower_{self.lower_bound * 100}%"
        self.upper_bound_name = f"upper_{self.upper_bound * 100}%"


@dataclass
class Estimate:
    """Stores the mean and bounds (uncertainty interval) of 'comparisons' and
    'effects' estimates. Used in 'PredictiveDifferences' to store typed data
    for the summary dataframe.

    Parameters
    ----------
    mean : Dict[str, xr.DataArray]
        The mean of the posterior distribution (chains and draws).
    bounds : Dict[str, xr.Dataset]
        The uncertainty interval of the posterior distribution (chains and draws).
    use_hdi : bool
        Whether to use the highest density interval (HDI) (True) or quantiles (False).
    """

    mean: Dict[str, xr.DataArray]
    bounds: Dict[str, xr.Dataset]
    use_hdi: bool
    bounds_list: list = field(init=False)
    lower: xr.DataArray = field(init=False)
    higher: xr.DataArray = field(init=False)

    def __post_init__(self):
        """
        Parses the mean and bounds dictionaries into arrays for inserting
        the 'mean', 'lower', and 'upper' columns into the summary dataframe.
        """
        self.bounds_list = [self.bounds[key] for key in self.bounds]
        self.mean = np.array(list(self.mean.values())).flatten()

        if self.use_hdi:
            data_var = list(self.bounds_list[0].data_vars)[0]
            self.lower = np.array(
                [self.bounds[key][data_var].sel(hdi="lower") for key in self.bounds]
            ).flatten()
            self.higher = np.array(
                [self.bounds[key][data_var].sel(hdi="higher") for key in self.bounds]
            ).flatten()
        else:
            lower = self.bounds_list[0].coords["quantile"].values[0]
            higher = self.bounds_list[0].coords["quantile"].values[1]
            self.lower = np.array(
                [self.bounds[key].sel(quantile=lower) for key in self.bounds]
            ).flatten()
            self.higher = np.array(
                [self.bounds[key].sel(quantile=higher) for key in self.bounds]
            ).flatten()


# pylint: disable=consider-iterating-dictionary
# pylint: disable=too-many-instance-attributes
@dataclass
class PredictiveDifferences:
    """Computes predictive differences and their uncertainty intervals for
    'comparisons' and 'slopes' effects and returns a summary dataframe of the
    results.

    Parameters
    ----------
    model : Model
        Bambi model object.
    preds_data : pd.DataFrame
        Dataframe used to generate predictions.
    variable : VariableInfo
        Variable of interest with its name and values.
    conditional : ConditionalInfo
        Conditional covariates with their names and values (if any).
    response : ResponseInfo
        Response variable with its name and target.
    use_hdi : bool
        Whether to use the highest density interval (HDI) (True) or quantiles (False).
    kind : str
        Type of effect to compute. Either 'comparisons' or 'slopes'.

    Returns
    -------
    summary_df : pd.DataFrame
        Dataframe with the data used to generate predictions, the effect kind
        and type, variable of interest name and value, and the mean and uncertainty
        intervals of the predictive difference estimate.
    """

    model: Model
    preds_data: pd.DataFrame
    variable: VariableInfo
    conditional: ConditionalInfo
    response: ResponseInfo
    use_hdi: bool
    kind: str
    estimate_name: str = field(init=False)
    estimate: Estimate = field(init=False)
    summary_df: pd.DataFrame = field(init=False)
    contrast_values: list = field(init=False)

    def set_variable_values(self, draws: dict) -> np.ndarray:
        """
        Obtain pairwise combinations of the 'draws' keys. The dictionary keys
        represent the variable of interest's values. If 'comparisons', then
        the keys are the contrast values. If 'slopes', then the keys are the
        values of the variable of interest and the values of the variable of
        interest plus 'eps'.
        """

        # obtain pairwise combinations of the variable of interest's values (keys)
        keys = np.array(list(draws.keys()))
        pairwise_variables = list(itertools.combinations(keys, 2))

        # if 'slopes' and user passed their own values, then need to index the
        # original data, and the original data plus 'eps'
        if self.kind == "slopes" and self.variable.user_passed:
            original_data, original_data_plus_eps = (
                keys[: self.variable.passed_values.size],
                keys[self.variable.passed_values.size :],
            )
            pairwise_variables = np.dstack((original_data, original_data_plus_eps))[0]
            self.variable.values = self.variable.values.reshape(2, self.variable.passed_values.size)

        return pairwise_variables

    def get_slope_estimate(
        self,
        predictive_difference: xr.DataArray,
        pair: tuple,
        draws: dict,
        slope: str,
        eps: float,
        wrt_x: xr.DataArray,
    ) -> xr.DataArray:
        """
        Computes the slope estimate for 'dydx', 'dyex', 'eyex', 'eydx'.
        """
        predictive_difference = (predictive_difference / eps).rename(self.response.name_target)

        if slope in ("eyex", "dyex"):
            wrt_x = xr.DataArray(
                wrt_x[pair[1]],
                coords={self.response.name_obs: np.arange(0, len(wrt_x[pair[1]]))},
                dims=[self.response.name_obs],
            )

        if slope in ("eyex", "eydx"):
            y_hat = draws[pair[1]]

        if slope == "eyex":
            predictive_difference = predictive_difference * (wrt_x / y_hat)
        elif slope == "eydx":
            predictive_difference = predictive_difference * (1 / y_hat)
        elif slope == "dyex":
            predictive_difference = predictive_difference * wrt_x

        return predictive_difference

    def get_estimate(
        self,
        idata: az.InferenceData,
        response_transforms: dict,
        comparison_type: str = "diff",
        slope: str = "dydx",
        eps: Union[float, None] = None,
        prob: float = 0.94,
    ):
        """Obtain the effect ('comparisons' or 'slopes') estimate and uncertainty
        interval using the posterior samples. First, the posterior samples are
        subsetted by the variable of interest's values. Then, the effect is
        computed for each pairwise combination of the variable of interest's
        values.

        Parameters
        ----------
        idata : az.InferenceData
            InferenceData object containing the posterior samples for the model.
        response_transforms : dict
            Dictionary with the response variable name as key and the
            transformation function as value.
        comparison_type : str
            Type of comparison to compute. Either 'diff' or 'ratio'. Defaults
            to 'diff'.
        slope : str
            Type of slope to compute. Either 'dydx', 'dyex', 'eyex', 'eydx'.
            Defaults to 'dydx'.
        eps : float
            Value to add to the variable of interest's values to compute the
            slope. Defaults to None.
        prob : float
            Probability for the uncertainty interval. Defaults to 0.94.

        Returns
        -------
        estimate : Estimate
            Estimate object with the effect estimate mean  and uncertainty
            interval.
        """
        assert self.kind in ("slopes", "comparisons")
        assert comparison_type in SUPPORTED_COMPARISONS.keys()

        function = SUPPORTED_COMPARISONS[comparison_type]

        if self.kind == "slopes":
            self.estimate_name = slope
        else:
            self.estimate_name = comparison_type

        if self.variable.values.ndim == 1:
            self.variable.values = np.array(self.variable.values).reshape(-1, 1)

        draws = {}
        variable_data = {}
        for idx, _ in enumerate(self.variable.values):
            mask = np.array(self.preds_data[self.variable.name].isin(self.variable.values[idx]))
            select_draw = response_transforms(
                idata.posterior[self.response.name_target].sel({self.response.name_obs: mask})
            )
            select_draw = select_draw.assign_coords(
                {self.response.name_obs: np.arange(len(select_draw.coords[self.response.name_obs]))}
            )
            draws[f"mask_{idx}"] = select_draw

            if slope in ("eyex", "dyex"):
                variable_data[f"mask_{idx}"] = self.preds_data[
                    self.preds_data[self.variable.name].isin(self.variable.values[idx])
                ][self.variable.name]

        pairwise_variables = self.set_variable_values(draws)

        difference_mean = {}
        difference_bounds = {}
        for idx, pair in enumerate(pairwise_variables):
            # comparisons effects
            predictive_difference = function(draws[pair[1]], draws[pair[0]])
            # slope effects
            if self.kind == "slopes":
                predictive_difference = self.get_slope_estimate(
                    predictive_difference, pair, draws, slope, eps, variable_data
                )

            difference_mean[f"estimate_{idx}"] = predictive_difference.mean(("chain", "draw"))

            if self.use_hdi:
                difference_bounds[f"estimate_{idx}"] = az.hdi(predictive_difference, prob)
            else:
                difference_bounds[f"estimate_{idx}"] = predictive_difference.quantile(
                    q=(self.response.lower_bound, self.response.upper_bound), dim=("chain", "draw")
                )

        self.estimate = Estimate(difference_mean, difference_bounds, self.use_hdi)

        return self

    def get_summary_df(self, response_dim: np.ndarray) -> pd.DataFrame:
        """
        Builds the summary dataframe for 'comparisons' and 'slopes' effects.
        There are four scenarios to consider:

            1.) If the effect kind is 'comparisons' and more than 2 values are being
            compared, then the entire 'preds' data is used.

            2.) If the model predictions have multiple response levels, then 'preds' data
            needs to be duplicated to match the number of response levels. E.g., 'preds'
            data has 100 rows and 3 response levels, then the summary dataframe will have
            300 rows since the model made a prediction for each response level for each
            sample in 'preds'.

            3.) If the effect kind is 'slopes' and more than 2 values are being compared, then
            only a subset of the 'preds' data is used to build the summary.

            4.) If the number of values passed for the variable of interest is less then 2
            for 'comparisons' and 'slopes', then a subset of the 'preds' data is used
            to build the summary.
        """
        # Scenario 1
        if len(self.variable.values) > 2 and self.kind == "comparisons":
            summary_df = self.preds_data.drop(columns=self.variable.name).drop_duplicates()
            covariates_cols = summary_df.columns
            contrast_values = list(itertools.combinations(self.variable.values.flatten(), 2))
            covariate_vals = np.tile(summary_df.T, len(contrast_values))
            summary_df = pd.DataFrame(data=covariate_vals.T, columns=covariates_cols)
            contrast_values = np.repeat(
                contrast_values, summary_df.shape[0] // len(contrast_values), axis=0
            )
            contrast_values = [tuple(elem) for elem in contrast_values]
        # Scenario 2
        elif len(response_dim) > 1:
            summary_df = self.preds_data.drop(columns=self.variable.name).drop_duplicates()
            covariates_cols = summary_df.columns
            contrast_values = self.variable.values.flatten()
            covariate_vals = np.repeat(summary_df.T, len(response_dim))
            summary_df = pd.DataFrame(data=covariate_vals.T, columns=covariates_cols)
            summary_df["estimate_dim"] = np.tile(
                response_dim, summary_df.shape[0] // len(response_dim)
            )
            contrast_values = [tuple(contrast_values)] * summary_df.shape[0]
        # Scenario 3 & 4
        else:
            wrt = {}
            for idx, _ in enumerate(self.variable.values):
                mask = np.array(self.preds_data[self.variable.name].isin(self.variable.values[idx]))
                wrt[f"draw_mask_{idx}"] = self.preds_data[mask][self.variable.name].reset_index(
                    drop=True
                )
                # only need to get "a" dataframe since remaining N dataframes are identical
                if idx == 0:
                    summary_df = (
                        self.preds_data[mask]
                        .drop(columns=self.variable.name)
                        .reset_index(drop=True)
                    )
            contrast_values = pd.concat(wrt.values(), axis=1).apply(tuple, axis=1)

        summary_df.insert(0, "term", self.variable.name)
        summary_df.insert(1, "estimate_type", self.estimate_name)
        summary_df.insert(2, "value", contrast_values)
        summary_df.insert(len(summary_df.columns), "estimate", self.estimate.mean)
        summary_df.insert(
            len(summary_df.columns), self.response.lower_bound_name, self.estimate.lower
        )
        summary_df.insert(
            len(summary_df.columns), self.response.upper_bound_name, self.estimate.higher
        )

        self.summary_df = summary_df
        self.contrast_values = contrast_values

        return self.summary_df

    def average_by(self, variable: Union[bool, str]) -> pd.DataFrame:
        """Uses the original 'summary_df' to perform a marginal (if 'variable=True')
        or group by average if covariate(s) are passed.

        Parameters
        ----------
        variable : Union[bool, str]
            If 'True', then average over all covariates. If a string
            is passed, then a group by average is performed.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the marginal or group by average.
        """
        if variable is True:
            contrast_df_avg = average_over(self.summary_df, "all")
            contrast_df_avg.insert(0, "term", self.variable.name)
            contrast_df_avg.insert(1, "estimate_type", self.estimate_name)
            if self.kind != "slopes" and len(self.variable.values) < 3:
                contrast_df_avg.insert(2, "value", self.contrast_values)
        else:
            contrast_df_avg = average_over(self.summary_df, variable)
            contrast_df_avg.insert(0, "term", self.variable.name)
            contrast_df_avg.insert(1, "estimate_type", self.estimate_name)
            if self.kind != "slopes" and len(self.variable.values) < 3:
                contrast_df_avg.insert(2, "value", self.contrast_values)

        return contrast_df_avg.reset_index(drop=True)


def predictions(
    model: Model,
    idata: az.InferenceData,
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, bool, None] = None,
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob=None,
    transforms=None,
    sample_new_groups=False,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    conditional : str, list, dict, optional
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    pps: bool, optional
        Whether to plot the posterior predictive samples. Defaults to ``False``.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.

    Returns
    -------
    cap_data : pandas.DataFrame
        A DataFrame with the ``create_cap_data`` and model predictions.

    Raises
    ------
    ValueError
        If ``pps`` is ``True`` and ``target`` is not ``"mean"``.
        If ``conditional`` is a list and the length is greater than 3.
        If ``prob`` is not > 0 and < 1.
    """
    if pps and target != "mean":
        raise ValueError("When passing 'pps=True', target must be 'mean'")

    if isinstance(conditional, list):
        if len(conditional) > 3:
            raise ValueError(
                f"Only 3 covariates can be passed to 'conditional'. {len(conditional)} "
                "were passed. If you would like to pass more than 3 covariates, use "
                "a dictionary."
            )

    conditional_info = ConditionalInfo(model, conditional)
    transforms = transforms if transforms is not None else {}

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]
    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    cap_data = create_predictions_data(conditional_info, conditional_info.user_passed)

    if target != "mean":
        component = model.components[target]
        if component.alias:
            # use only the aliased name (without appended target)
            response_name = get_aliased_name(component)
            target = None
        else:
            # use the default response "y" and append target
            response_name = get_aliased_name(model.response_component.response_term)
    else:
        response_name = get_aliased_name(model.response_component.response_term)

    response = ResponseInfo(response_name, target)
    response_transform = transforms.get(response_name, identity)

    if pps:
        idata = model.predict(
            idata, data=cap_data, sample_new_groups=sample_new_groups, inplace=False, kind="pps"
        )
        y_hat = response_transform(idata["posterior_predictive"][response.name])
        y_hat_mean = y_hat.mean(("chain", "draw"))
    else:
        idata = model.predict(
            idata, data=cap_data, sample_new_groups=sample_new_groups, inplace=False
        )
        y_hat = response_transform(idata["posterior"][response.name_target])
        y_hat_mean = y_hat.mean(("chain", "draw"))

    if use_hdi and pps:
        y_hat_bounds = az.hdi(y_hat, prob)[response.name].T
    elif use_hdi:
        y_hat_bounds = az.hdi(y_hat, prob)[response.name_target].T
    else:
        lower_bound = round((1 - prob) / 2, 4)
        upper_bound = 1 - lower_bound
        y_hat_bounds = y_hat.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound
    response.lower_bound, response.upper_bound = lower_bound, upper_bound

    cap_data = cap_data.copy()
    if y_hat_mean.ndim > 1:
        cap_data = merge(y_hat_mean, y_hat_bounds, cap_data)
        cap_data = cap_data.rename(
            columns={
                f"{response.name}_dim": "estimate_dim",
                f"{response.name_target}": "estimate",
                f"{response.name_target}_x": response.lower_bound_name,
                f"{response.name_target}_y": response.upper_bound_name,
            }
        )
    else:
        cap_data["estimate"] = y_hat_mean
        cap_data[response.lower_bound_name] = y_hat_bounds[0]
        cap_data[response.upper_bound_name] = y_hat_bounds[1]

    if average_by is not None:
        if average_by is True:
            average_by = "all"
        cap_data = average_over(cap_data, covariate=average_by)

    return cap_data


def comparisons(
    model: Model,
    idata: az.InferenceData,
    contrast: Union[str, dict],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, bool, None] = None,
    comparison_type: str = "diff",
    use_hdi: bool = True,
    prob: Union[float, None] = None,
    transforms: Union[dict, None] = None,
    sample_new_groups: bool = False,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    contrast : str, dict
        The predictor name whose contrast we would like to compare.
    conditional : str, list, dict, optional
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    comparison_type : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the comparison values, highest density interval, contrast name,
        contrast value, and conditional values.

    Raises
    ------
    ValueError
        If `wrt` is a dict and length of ``contrast`` is greater than 1.
        If `wrt` is a dict and length of ``contrast`` is greater than 2 and
        ``conditional`` is ``None``.
        If ``conditional`` is None and ``contrast`` is categorical with > 2 values.
        If ``conditional`` is a list and the length is greater than 3.
        If ``comparison_type`` is not 'diff' or 'ratio'.
        If ``prob`` is not > 0 and < 1.
    """
    contrast_name = contrast
    if isinstance(contrast, dict):
        if len(contrast) > 1:
            raise ValueError(
                f"Only one predictor can be passed to 'contrast'. {len(contrast)} were passed."
            )
        contrast_name, contrast_values = next(iter(contrast.items()))
        if len(contrast_values) > 2 and conditional is None:
            raise ValueError(
                "'conditional' must be specified when 'contrast' has more than 2 values."
                f"{contrast_name} was passed {len(contrast_values)} values."
            )

    if isinstance(conditional, list):
        if len(conditional) > 3:
            raise ValueError(
                f"Only 3 covariates can be passed to 'conditional'. {len(conditional)} "
                "were passed. If you would like to pass more than 3 covariates, "
                "use a dictionary."
            )

    if conditional is None:
        if is_categorical_dtype(model.data[contrast_name]) or is_string_dtype(
            model.data[contrast_name]
        ):
            num_levels = len(model.data[contrast_name].unique())
            if num_levels > 2:
                raise ValueError(
                    f"'conditional' must be specified when 'contrast' has more than 2 values. "
                    f"{contrast_name} has {num_levels} unique values."
                )

    if comparison_type not in ("diff", "ratio"):
        raise ValueError("'comparison_type' must be 'diff' or 'ratio'")

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]
    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    contrast_info = VariableInfo(model, contrast, "comparisons", eps=0.5)
    conditional_info = ConditionalInfo(model, conditional)

    transforms = transforms if transforms is not None else {}

    response_name = get_aliased_name(model.response_component.response_term)
    response = ResponseInfo(
        response_name, target="mean", lower_bound=lower_bound, upper_bound=upper_bound
    )
    response_transform = transforms.get(response_name, identity)

    # 'comparisons' not be limited to ("main", "group", "panel")
    comparisons_data = create_differences_data(
        conditional_info, contrast_info, conditional_info.user_passed, kind="comparisons"
    )
    idata = model.predict(
        idata, data=comparisons_data, sample_new_groups=sample_new_groups, inplace=False
    )

    # returns empty array if model predictions do not have multiple dimensions
    response_dim_key = response.name + "_dim"
    if response_dim_key in idata.posterior.coords:
        response_dim = idata.posterior.coords[response_dim_key].values
    else:
        response_dim = np.empty(0)

    predictive_difference = PredictiveDifferences(
        model,
        comparisons_data,
        contrast_info,
        conditional_info,
        response,
        use_hdi,
        kind="comparisons",
    )
    comparisons_summary = predictive_difference.get_estimate(
        idata, response_transform, comparison_type, prob=prob
    ).get_summary_df(response_dim)

    if average_by:
        comparisons_summary = predictive_difference.average_by(variable=average_by)

    return enforce_dtypes(comparisons_data, comparisons_summary)


def slopes(
    model: Model,
    idata: az.InferenceData,
    wrt: Union[str, dict],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, bool, None] = None,
    eps: float = 1e-4,
    slope: str = "dydx",
    use_hdi: bool = True,
    prob: Union[float, None] = None,
    transforms: Union[dict, None] = None,
    sample_new_groups: bool = False,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Slopes

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    wrt : str, dict
        The slope of the regression with respect to (wrt) this predictor will be computed.
    conditional : str, list, dict, optional
        The covariates we would like to condition on. If dict, keys are the covariate names and
        values are the values to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    eps : float, optional
        To compute the slope, 'wrt' is evaluated at wrt +/- 'eps'. The rate of change is then
        computed as the difference between the two values divided by 'eps'. Defaults to 1e-4.
    slope: str, optional
        The type of slope to compute. Defaults to 'dydx'.
        'dydx' represents a unit increase in 'wrt' is associated with an n-unit change in
        the response.
        'eyex' represents a percentage increase in 'wrt' is associated with an n-percent
        change in the response.
        'eydx' represents a unit increase in 'wrt' is associated with an n-percent
        change in the response.
        'dyex' represents a percent change in 'wrt' is associated with a unit increase
        in the response.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    sample_new_groups : bool, optional
        If the model contains group-level effects, and data is passed for unseen groups, whether
        to sample from the new groups. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the comparison values, highest density interval, ``wrt`` name,
        contrast value, and conditional values.

    Raises
    ------
    ValueError
        If length of ``wrt`` is greater than 1.
        If ``conditional`` is ``None`` and ``wrt`` is passed more than 2 values.
        If ``conditional`` is ``None`` and default ``wrt`` has more than 2 unique values.
        If ``conditional`` is a list and the length is greater than 3.
        If ``slope`` is not 'dydx', 'dyex', 'eyex', or 'eydx'.
        If ``prob`` is not > 0 and < 1.
    """
    wrt_name = wrt
    if isinstance(wrt, dict):
        if len(wrt) > 1:
            raise ValueError(f"Only one predictor can be passed to 'wrt'. {len(wrt)} were passed.")
        wrt_name, wrt_values = next(iter(wrt.items()))
        if not isinstance(wrt_values, (list, np.ndarray)):
            wrt_values = [wrt_values]
        if len(wrt_values) > 2 and conditional is None:
            raise ValueError(
                f"'conditional' must be specified when 'wrt' has more than 2 values. "
                f"{wrt_name} was passed {len(wrt_values)} values."
            )

    if isinstance(conditional, list):
        if len(conditional) > 3:
            raise ValueError(
                f"Only 3 covariates can be passed to 'conditional'. {len(conditional)} "
                " were passed. If you would like to pass more than 3 covariates, "
                "use a dictionary."
            )

    if not isinstance(wrt, dict) and conditional is None:
        if is_categorical_dtype(model.data[wrt_name]) or is_string_dtype(model.data[wrt_name]):
            num_levels = len(model.data[wrt_name].unique())
            if num_levels > 2:
                raise ValueError(
                    f"'conditional' must be specified when 'wrt' has more than 2 values. "
                    f"{wrt_name} has {num_levels} unique values."
                )

    if slope not in ("dydx", "dyex", "eyex", "eydx"):
        raise ValueError("'slope' must be one of ('dydx', 'dyex', 'eyex', 'eydx')")

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]
    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    # 'slopes' should not be limited to ("main", "group", "panel")
    conditional_info = ConditionalInfo(model, conditional)

    grid = bool(conditional_info.covariates)
    # if wrt is categorical or string dtype, call 'comparisons' to compute the
    # difference between group means as the slope
    effect_type = "slopes"
    if is_categorical_dtype(model.data[wrt_name]) or is_string_dtype(model.data[wrt_name]):
        effect_type = "comparisons"
        eps = None
    wrt_info = VariableInfo(model, wrt, effect_type, grid, eps)

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    transforms = transforms if transforms is not None else {}

    response_name = get_aliased_name(model.response_component.response_term)
    response = ResponseInfo(response_name, "mean", lower_bound, upper_bound)
    response_transform = transforms.get(response_name, identity)

    slopes_data = create_differences_data(
        conditional_info, wrt_info, conditional_info.user_passed, effect_type
    )
    idata = model.predict(
        idata, data=slopes_data, sample_new_groups=sample_new_groups, inplace=False
    )

    # returns empty array if model predictions do not have multiple dimensions
    response_dim_key = response.name + "_dim"
    if response_dim_key in idata.posterior.coords:
        response_dim = idata.posterior.coords[response_dim_key].values
    else:
        response_dim = np.empty(0)

    predictive_difference = PredictiveDifferences(
        model, slopes_data, wrt_info, conditional_info, response, use_hdi, effect_type
    )
    slopes_summary = predictive_difference.get_estimate(
        idata, response_transform, "diff", slope, eps
    ).get_summary_df(response_dim)

    if average_by:
        slopes_summary = predictive_difference.average_by(variable=average_by)

    return enforce_dtypes(slopes_data, slopes_summary)
