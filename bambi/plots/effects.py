from dataclasses import dataclass, field
import itertools
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype
import xarray as xr

from bambi.models import Model
from bambi.plots.create_data import create_cap_data, create_differences_data
from bambi.plots.utils import (
    average_over,
    ConditionalInfo,
    identity,
    VariableInfo,
)
from bambi.utils import get_aliased_name, listify


@dataclass
class ResponseInfo:
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
        Assigns commonly used f-strings for indexing and column names as attributes.
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
    mean: Dict[str, xr.DataArray]
    hdi: Dict[str, xr.Dataset]
    lower: xr.DataArray = field(init=False)
    higher: xr.DataArray = field(init=False)

    def __post_init__(self):
        """ """
        self.hdi_list = [self.hdi[key] for key in self.hdi]
        self.mean = np.array([value for value in self.mean.values()]).flatten()
        data_var = list(self.hdi_list[0].data_vars)[0]
        # TODO: add logic for if not az.InferenceData
        self.lower = np.array(
            [self.hdi[key][data_var].sel(hdi="lower") for key in self.hdi]
        ).flatten()
        self.higher = np.array(
            [self.hdi[key][data_var].sel(hdi="higher") for key in self.hdi]
        ).flatten()


SUPPORTED_SLOPES = ("dydx", "eyex")
SUPPORTED_COMPARISONS = {
    "diff": lambda x, y: x - y,
    "ratio": lambda x, y: x / y,
}


@dataclass
class PredictiveDifferences:
    model: Model
    idata: az.InferenceData
    preds_data: pd.DataFrame
    variable: VariableInfo
    conditional: ConditionalInfo
    response: ResponseInfo
    use_hdi: bool
    kind: str

    def get_estimate(self, comparison_type: str = "diff", slope: str = "dydx", eps: float = None):
        assert self.kind in ("slopes", "comparisons")
        assert comparison_type in SUPPORTED_COMPARISONS.keys()

        if self.kind == "slopes":
            self.estimate_name = slope
        else:
            self.estimate_name = comparison_type

        function = SUPPORTED_COMPARISONS[comparison_type]

        if self.variable.values.ndim == 1:
            self.variable.values = np.array(self.variable.values).reshape(-1, 1)

        draws = {}
        data = {}
        for idx, _ in enumerate(self.variable.values):
            mask = np.array(self.preds_data[self.variable.name].isin(self.variable.values[idx]))
            select_draw = self.idata.posterior[self.response.name_target].sel(
                {self.response.name_obs: mask}
            )
            select_draw = select_draw.assign_coords(
                {self.response.name_obs: np.arange(len(select_draw.coords[self.response.name_obs]))}
            )
            draws[f"draw_mask_{idx}"] = select_draw
            if slope == "eyex":
                data[f"draw_mask_{idx}"] = self.preds_data[
                    self.preds_data[self.variable.name].isin(self.variable.values[idx])
                ][self.variable.name]

        # TODO: could be a namedtuple so the code / indexing is more understanable???
        keys = np.array(list(draws.keys()))
        pairwise_variables = list(itertools.combinations(keys, 2))

        difference_mean = {}
        difference_bounds = {}
        for idx, pair in enumerate(pairwise_variables):
            predictive_difference = function(draws[pair[1]], draws[pair[0]])

            # TODO: move slopes estimate to different func???
            if self.kind == "slopes":
                predictive_difference = predictive_difference / eps

                if slope == "eyex":
                    x1 = xr.DataArray(
                        data[pair[1]],
                        coords={self.response.name_obs: np.arange(0, len(data[pair[1]]))},
                        dims=[self.response.name_obs],
                    )
                    y1 = draws[pair[1]]
                    predictive_difference = (predictive_difference * (x1 / y1)).rename(
                        self.response.name_target
                    )

            difference_mean[f"draw_{idx}"] = predictive_difference.mean(("chain", "draw"))

            if self.use_hdi:
                # TODO: add prob. arg.
                difference_bounds[f"draw_{idx}"] = az.hdi(predictive_difference, 0.94)
            else:
                difference_bounds[f"draw_{idx}"] = difference_mean.quantile(
                    q=(self.response.lower_bound, self.response.upper_bound), dim=("chain", "draw")
                )

        self.estimate = Estimate(difference_mean, difference_bounds)

        return self

    def get_summary_df(self) -> pd.DataFrame:
        if len(self.variable.values) > 2:
            self.summary_df = self.preds_data.drop(columns=self.variable.name).drop_duplicates()
            covariates_cols = self.summary_df.columns
            covariate_vals = np.tile(self.summary_df.T, len(self.variable.values))
            self.summary_df = pd.DataFrame(data=covariate_vals.T, columns=covariates_cols)
            self.contrast_values = list(itertools.combinations(self.variable.values.flatten(), 2))
            self.contrast_values = np.repeat(
                self.contrast_values, self.preds_data.shape[0] // len(self.contrast_values), axis=0
            )
            self.contrast_values = [tuple(elem) for elem in self.contrast_values]
        else:
            wrt = {}
            for idx, _ in enumerate(self.variable.values):
                mask = np.array(self.preds_data[self.variable.name].isin(self.variable.values[idx]))
                wrt[f"draw_mask_{idx}"] = self.preds_data[mask][self.variable.name].reset_index(
                    drop=True
                )
                # only need to get "a" dataframe since remaining N dataframes are identical
                if idx == 0:
                    self.summary_df = (
                        self.preds_data[mask]
                        .drop(columns=self.variable.name)
                        .reset_index(drop=True)
                    )
            self.contrast_values = pd.concat(wrt.values(), axis=1).apply(tuple, axis=1)

        # TODO: is it okay to assign self.summary_df = summary_df at the end
        # TODO: of this method to avoid the code being littered with "self."???
        self.summary_df.insert(0, "term", self.variable.name)
        self.summary_df.insert(1, "estimate_type", self.estimate_name)
        self.summary_df.insert(2, "value", self.contrast_values)
        self.summary_df.insert(len(self.summary_df.columns), "estimate", self.estimate.mean)
        self.summary_df.insert(
            len(self.summary_df.columns), self.response.lower_bound_name, self.estimate.lower
        )
        self.summary_df.insert(
            len(self.summary_df.columns), self.response.upper_bound_name, self.estimate.higher
        )

        return self.summary_df

    def average_by(self, by: Union[bool, str]) -> pd.DataFrame:
        if by is True:
            contrast_df_avg = average_over(self.summary_df, None)
            contrast_df_avg.insert(0, "term", self.variable.name)
            contrast_df_avg.insert(1, "estimate_type", self.estimate_name)
            if self.kind != "slopes":
                contrast_df_avg.insert(2, "value", self.contrast_values)
        else:
            contrast_df_avg = average_over(self.summary_df, by)
            contrast_df_avg.insert(0, "term", self.variable.name)
            contrast_df_avg.insert(1, "estimate_type", self.estimate_name)
            if self.kind != "slopes":
                contrast_df_avg.insert(2, "value", self.contrast_values)

        return contrast_df_avg.reset_index(drop=True)


def predictions(
    model: Model,
    idata: az.InferenceData,
    covariates: Union[str, dict, list],
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    prob=None,
    transforms=None,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    covariates : list or dict
        A sequence of between one and three names of variables or a dict of length between one
        and three.
        If a sequence, the first variable is taken as the main variable and is mapped to the
        horizontal axis. If present, the second name is a coloring/grouping variable,
        and the third is mapped to different plot panels.
        If a dictionary, keys must be taken from ("main", "group", "panel") and the values
        are the names of the variables.
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

    Returns
    -------
    cap_data : pandas.DataFrame
        A DataFrame with the ``create_cap_data`` and model predictions.

    Raises
    ------
    ValueError
        If ``pps`` is ``True`` and ``target`` is not ``"mean"``.
        If passed ``covariates`` is not in correct key, value format.
        If length of ``covariates`` is not between 1 and 3.
    """

    if pps and target != "mean":
        raise ValueError("When passing 'pps=True', target must be 'mean'")

    covariate_kinds = ("main", "group", "panel")
    if not isinstance(covariates, dict):
        covariates = listify(covariates)
        covariates = dict(zip(covariate_kinds, covariates))
    else:
        assert covariate_kinds[0] in covariates
        assert set(covariates).issubset(set(covariate_kinds))

    assert 1 <= len(covariates) <= 3

    if transforms is None:
        transforms = {}

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]

    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    cap_data = create_cap_data(model, covariates)

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
        idata = model.predict(idata, data=cap_data, inplace=False, kind="pps")
        y_hat = response_transform(idata.posterior_predictive[response.name])
        y_hat_mean = y_hat.mean(("chain", "draw"))
    else:
        idata = model.predict(idata, data=cap_data, inplace=False)
        y_hat = response_transform(idata.posterior[response.name_target])
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

    cap_data["estimate"] = y_hat_mean
    cap_data[response.lower_bound_name] = y_hat_bounds[0]
    cap_data[response.upper_bound_name] = y_hat_bounds[1]

    return cap_data


def comparisons(
    model: Model,
    idata: az.InferenceData,
    contrast: Union[str, dict, list],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, bool, None] = None,
    comparison_type: str = "diff",
    use_hdi: bool = True,
    prob=None,
    transforms=None,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Comparisons

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    contrast : str, dict, list
        The predictor name whose contrast we would like to compare.
    conditional : str, dict, list
        The covariates we would like to condition on.
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

    Returns
    -------
    pandas.DataFrame
        A dataframe with the comparison values, highest density interval, contrast name,
        contrast value, and conditional values.

    Raises
    ------
    ValueError
        If length of ``contrast`` is greater than 1.
        If ``contrast`` is not a string, dictionary, or list.
        If ``comparison_type`` is not 'diff' or 'ratio'.
        If ``prob`` is not > 0 and < 1.
    """

    if not isinstance(contrast, (dict, list, str)):
        raise ValueError("'contrast' must be a string, dictionary, or list.")
    if isinstance(contrast, (dict, list)):
        if len(contrast) > 1:
            raise ValueError(
                f"Only one contrast predictor can be passed. {len(contrast)} were passed."
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

    # TODO: this should be a input to 'PredictiveDifferences'
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response = ResponseInfo(
        response_name, target="mean", lower_bound=lower_bound, upper_bound=upper_bound
    )

    # TODO 'comparisons' not be limited to ("main", "group", "panel")
    comparisons_data = create_differences_data(
        conditional_info, contrast_info, conditional_info.user_passed, kind="comparisons"
    )
    idata = model.predict(idata, data=comparisons_data, inplace=False)

    predictive_difference = PredictiveDifferences(
        model,
        idata,
        comparisons_data,
        contrast_info,
        conditional_info,
        response,
        use_hdi,
        kind="comparisons",
    )
    comparisons_summary = predictive_difference.get_estimate(comparison_type).get_summary_df()

    if average_by:
        comparisons_summary = predictive_difference.average_by(average_by)

    return comparisons_summary


def slopes(
    model: Model,
    idata: az.InferenceData,
    wrt: Union[str, dict],
    conditional: Union[str, dict, list, None] = None,
    average_by: Union[str, list, bool, None] = None,
    eps: float = 1e-4,
    slope: str = "dydx",
    use_hdi: bool = True,
    prob=None,
    transforms=None,
) -> pd.DataFrame:
    """Compute Conditional Adjusted Effects

    Parameters
    ----------
    model : bambi.Model
        The model for which we want to plot the predictions.
    idata : arviz.InferenceData
        The InferenceData object that contains the samples from the posterior distribution of
        the model.
    wrt : str, dict
        The slope of the regression with respect to (wrt) this predictor will be computed.
    conditional : str, dict, list
        The covariates we would like to condition on.
    average_by: str, list, bool, optional
        The covariates we would like to average by. The passed covariate(s) will marginalize
        over the other covariates in the model. If True, it averages over all covariates
        in the model to obtain the average estimate. Defaults to ``None``.
    eps : float, optional
        To compute the slope, 'wrt' is evaluated at wrt +/- 'eps'. The rate of change is then
        computed as the difference between the two values divided by 'eps'. Defaults to 1e-4.
    slope: str, optional
        The type of slope to compute. Defaults to 'dydx'.
        'dydx' represents a unit increase in 'wrt' is associated with n-unit change in the response.
        'eyex' represents a percentage increase in 'wrt' is associated with n-percent change in
        the response.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the comparison values, highest density interval, ``wrt`` name,
        contrast value, and conditional values.

    Raises
    ------
    ValueError
        If ``conditional`` is ``None`` and ``wrt`` is a dictionary.
        If length of ``wrt`` is greater than 1.
        If ``prob`` is not > 0 and < 1.
    """
    if conditional is None and isinstance(wrt, dict):
        raise ValueError("If a value is passed with 'wrt', then 'conditional' cannot be 'None'.")

    wrt_name = wrt
    if isinstance(wrt, dict):
        if len(wrt) > 1:
            raise ValueError(f"Only one predictor can be passed to 'wrt'. {len(wrt)} were passed.")
        wrt_name = list(wrt.keys())[0]

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]
    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    # TODO 'slopes' not be limited to ("main", "group", "panel")
    conditional_info = ConditionalInfo(model, conditional)

    # TODO: this feels like a hack???
    grid = False
    if conditional_info.covariates:
        grid = True

    # if wrt is categorical or string dtype, call 'comparisons' to compute the
    # difference between group means as the slope
    effect_type = "slopes"
    if is_categorical_dtype(model.data[wrt_name]) or is_string_dtype(model.data[wrt_name]):
        effect_type = "comparisons"
        eps = None
    wrt_info = VariableInfo(model, wrt, effect_type, grid, eps)

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    # TODO: this should be a input to 'PredictiveDifferences'
    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response = ResponseInfo(response_name, "mean", lower_bound, upper_bound)

    slopes_data = create_differences_data(
        conditional_info, wrt_info, conditional_info.user_passed, effect_type
    )
    idata = model.predict(idata, data=slopes_data, inplace=False)

    predictive_difference = PredictiveDifferences(
        model, idata, slopes_data, wrt_info, conditional_info, response, use_hdi, effect_type
    )
    slopes_summary = predictive_difference.get_estimate("diff", slope, eps).get_summary_df()

    if average_by:
        slopes_summary = predictive_difference.average_by(average_by)

    return slopes_summary
