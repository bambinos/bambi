# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass, field
import itertools
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from bambi.models import Model
from bambi.plots.create_data import create_cap_data, create_comparisons_data
from bambi.plots.utils import average_over, ConditionalInfo, ContrastInfo, identity
from bambi.utils import get_aliased_name, listify


@dataclass
class ResponseInfo:
    name: str
    target: str = "mean"
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
        self.name_target = f"{self.name}_{self.target}"
        self.name_obs = f"{self.name}_obs"
        self.lower_bound_name = f"lower_{self.lower_bound * 100}%"
        self.upper_bound_name = f"upper_{self.upper_bound * 100}%"


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
        If passed ``covariates`` is not in correct key, value format.
        If length of ``covariates`` is not between 1 and 3.
    """

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


@dataclass
class ContrastEstimate:
    comparison: Dict[str, xr.DataArray]
    hdi: Dict[str, xr.Dataset]


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

    comparison_functions = {"diff": lambda x, y: x - y, "ratio": lambda x, y: x / y}
    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    contrast_info = ContrastInfo(model, contrast)
    conditional_info = ConditionalInfo(model, conditional)

    # 'comparisons' should not be restricted to ("main", "group", "panel")
    comparisons_df = create_comparisons_data(
        conditional_info, contrast_info, user_passed=conditional_info.user_passed
    )

    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)
    response = ResponseInfo(response_name, lower_bound=lower_bound, upper_bound=upper_bound)

    # perform predictions on new data
    idata = model.predict(idata, data=comparisons_df, inplace=False)

    def _compute_contrast_estimate(
        contrast: ContrastInfo,
        response: ResponseInfo,
        comparisons_df: pd.DataFrame,
        idata: az.InferenceData,
    ) -> ContrastEstimate:
        """
        Computes the contrast comparison estimate and highest density interval
        for a given contrast and response by first subsetting posterior draws
        using a contrast mask. Then, pairwise comparisons are computed for the
        contrast values. Finally, the mean comparison and lower/upper bounds
        are computed for each pairwise comparison.
        """
        function = comparison_functions[comparison_type]

        draws = {}
        for idx, val in enumerate(contrast.values):
            mask = np.array(comparisons_df[contrast.name] == contrast.values[idx])
            select_draw = idata.posterior[response.name_target].sel({response.name_obs: mask})
            select_draw = select_draw.assign_coords(
                {response.name_obs: np.arange(len(select_draw.coords[response.name_obs]))}
            )
            draws[val] = select_draw

        pairwise_contrasts = list(itertools.combinations(contrast.values, 2))

        comparison_mean = {}
        comparison_bounds = {}
        for idx, pair in enumerate(pairwise_contrasts):
            comparison_estimate = function(draws[pair[1]], draws[pair[0]])
            comparison_mean[pair] = comparison_estimate.mean(("chain", "draw"))
            if use_hdi:
                comparison_bounds[pair] = az.hdi(comparison_estimate, prob)
            else:
                comparison_bounds[pair] = comparison_estimate.quantile(
                    q=(response.lower_bound, response.upper_bound), dim=("chain", "draw")
                )

        return ContrastEstimate(comparison_mean, comparison_bounds)

    def _build_contrasts_df(
        contrast: ContrastInfo,
        condition: ConditionalInfo,
        response: ResponseInfo,
        comparisons_df: pd.DataFrame,
        idata: az.InferenceData,
        average_by,
    ) -> pd.DataFrame:
        """
        Builds a dataframe with the comparison values and lower / upper bounds from
        ``_compute_contrast_estimate`` along with the contrast name, contrast value,
        and conditional values.
        """
        contrast_estimate = _compute_contrast_estimate(contrast, response, comparisons_df, idata)
        # lower_bound, upper_bound = lower_bound * 100, upper_bound * 100

        # if two contrast values, then can drop duplicates to build contrast_df
        if len(contrast.values) < 3:
            if not any(condition.covariates.values()):
                contrast_df = model.data[comparisons_df.columns].drop(columns=contrast.name)
                num_rows = contrast_df.shape[0]
                contrast_df.insert(0, "term", contrast.name)
                contrast_df.insert(
                    1, "contrast", list(np.tile(contrast.values, num_rows).reshape(num_rows, 2))
                )
                contrast_df["estimate"] = contrast_estimate.comparison[
                    tuple(contrast.values)
                ].to_numpy()
            else:
                contrast_df = comparisons_df.drop_duplicates(
                    list(condition.covariates.values())
                ).reset_index(drop=True)
                contrast_df = contrast_df.drop(columns=contrast.name)
                num_rows = contrast_df.shape[0]
                contrast_df.insert(0, "term", contrast.name)
                contrast_df.insert(
                    1, "contrast", list(np.tile(contrast.values, num_rows).reshape(num_rows, 2))
                )
                contrast_df["estimate"] = contrast_estimate.comparison[
                    tuple(contrast.values)
                ].to_numpy()

            if use_hdi:
                contrast_df[response.lower_bound_name] = (
                    contrast_estimate.hdi[tuple(contrast.values)][response.name_target]
                    .sel(hdi="lower")
                    .values
                )
                contrast_df[response.upper_bound_name] = (
                    contrast_estimate.hdi[tuple(contrast.values)][response.name_target]
                    .sel(hdi="higher")
                    .values
                )
            else:
                contrast_df[response.lower_bound_name] = contrast_estimate.hdi[
                    tuple(contrast.values)
                ].sel(quantile=lower_bound)
                contrast_df[response.upper_bound_name] = contrast_estimate.hdi[
                    tuple(contrast.values)
                ].sel(quantile=upper_bound)

        # if > 2 contrast values, then need the full dataframe to build contrast_df
        elif len(contrast.values) >= 3:
            num_rows = comparisons_df.shape[0]
            contrast_df = comparisons_df.drop(columns=contrast.name)
            contrast_df.insert(0, "term", contrast.name)
            contrast_keys = [list(elem) for elem in list(contrast_estimate.comparison.keys())]
            contrast_df.insert(1, "contrast", contrast_keys * (num_rows // len(contrast.values)))

            estimates = []
            for val in contrast_estimate.comparison.values():
                estimates.append(val)
            contrast_df["estimate"] = np.array(estimates).flatten()

            lower = []
            upper = []
            for pair, val in zip(contrast_keys, contrast_estimate.hdi.values()):
                if use_hdi:
                    lower.append(
                        (
                            contrast_estimate.hdi[tuple(pair)][response.name_target]
                            .sel(hdi="lower")
                            .values
                        )
                    )
                    upper.append(
                        (
                            contrast_estimate.hdi[tuple(pair)][response.name_target]
                            .sel(hdi="higher")
                            .values
                        )
                    )
                else:
                    lower.append(contrast_estimate.hdi[tuple(pair)].sel(quantile=lower_bound))
                    upper.append(contrast_estimate.hdi[tuple(pair)].sel(quantile=upper_bound))

            contrast_df[response.lower_bound_name] = np.array(lower).flatten()
            contrast_df[response.upper_bound_name] = np.array(upper).flatten()

        contrast_df["contrast"] = contrast_df["contrast"].apply(tuple)

        if average_by:
            if average_by is True:
                contrast_df_avg = average_over(contrast_df, None)
                contrast_df_avg.insert(0, "term", contrast.name)
                contrast_df_avg.insert(
                    1,
                    "contrast",
                    np.tile(contrast_df["contrast"].drop_duplicates(), len(contrast_df_avg)),
                )
            else:
                contrast_df_avg = average_over(contrast_df, average_by)
                contrast_df_avg.insert(0, "term", contrast.name)
                contrast_df_avg.insert(
                    1,
                    "contrast",
                    np.tile(contrast_df["contrast"].drop_duplicates(), len(contrast_df_avg)),
                )
            return contrast_df_avg.reset_index(drop=True)
        else:
            return contrast_df.reset_index(drop=True)

    return _build_contrasts_df(
        contrast_info,
        conditional_info,
        response,
        comparisons_df,
        idata,
        average_by,
    )
