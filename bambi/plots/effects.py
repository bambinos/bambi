# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass
import itertools
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from bambi.models import Model
from bambi.plots.create_data import create_cap_data, create_comparisons_data
from bambi.plots.utils import average_by_group, Comparison, identity
from bambi.utils import get_aliased_name, listify


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
    response_transform = transforms.get(response_name, identity)

    if pps:
        idata = model.predict(idata, data=cap_data, inplace=False, kind="pps")
        y_hat = response_transform(idata.posterior_predictive[response_name])
        y_hat_mean = y_hat.mean(("chain", "draw"))
    else:
        idata = model.predict(idata, data=cap_data, inplace=False)
        y_hat = response_transform(idata.posterior[f"{response_name}_{target}"])
        y_hat_mean = y_hat.mean(("chain", "draw"))

    if use_hdi and pps:
        y_hat_bounds = az.hdi(y_hat, prob)[response_name].T
    elif use_hdi:
        y_hat_bounds = az.hdi(y_hat, prob)[f"{response_name}_{target}"].T
    else:
        lower_bound = round((1 - prob) / 2, 4)
        upper_bound = 1 - lower_bound
        y_hat_bounds = y_hat.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    cap_data["estimate"] = y_hat_mean
    cap_data[f"lower_{lower_bound}%"] = y_hat_bounds[0]
    cap_data[f"upper_{upper_bound}%"] = y_hat_bounds[1]

    return cap_data


@dataclass
class Contrast:
    term: Union[str, list]
    value: Union[list, np.ndarray, None]


@dataclass
class Response:
    name: str
    target: str = "mean"


@dataclass
class ContrastEstimate:
    comparison: Dict[str, xr.DataArray]
    hdi: Dict[str, xr.Dataset]


def comparisons(
    model: Model,
    idata: az.InferenceData,
    contrast: Union[str, dict, list],
    conditional: Union[str, dict, list],
    average_by: Union[str, list, None] = None,
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
    comparison_type : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
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
        When length of ``contrast`` is greater than 1.
        When ``contrast`` is not a string, dictionary, or list.
        When ``prob`` is not > 0 and < 1.
    """

    if isinstance(contrast, (dict, list)):
        if len(contrast) > 1:
            raise ValueError(
                f"Only one contrast predictor can be passed. {len(contrast)} were passed."
            )
        if isinstance(contrast, dict):
            contrast_name = next(iter(contrast.keys()))
        else:
            contrast_name = contrast[0]
    elif isinstance(contrast, str):
        contrast_name = contrast
    else:
        raise ValueError("'Contrast' must be a string, dictionary, or list.")

    if comparison_type not in ("diff", "ratio"):
        raise ValueError("'comparison_type' must be 'diff' or 'ratio'")

    comparison_functions = {"diff": lambda x, y: x - y, "ratio": lambda x, y: x / y}

    if prob is None:
        prob = az.rcParams["stats.hdi_prob"]

    if not 0 < prob < 1:
        raise ValueError(f"'prob' must be greater than 0 and smaller than 1. It is {prob}.")

    lower_bound = round((1 - prob) / 2, 4)
    upper_bound = 1 - lower_bound

    covariate_kinds = ("main", "group", "panel")
    # if not dict, then user did not pass values to condition on
    if not isinstance(conditional, dict):
        conditional = listify(conditional)
        conditional = dict(zip(covariate_kinds, conditional))
        comparisons_df = create_comparisons_data(
            model, Comparison(model, contrast, conditional), user_passed=False
        )
    # if dict, user passed values to condition on
    elif isinstance(conditional, dict):
        comparisons_df = create_comparisons_data(
            model, Comparison(model, contrast, conditional), user_passed=True
        )
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))

    if transforms is None:
        transforms = {}

    response_name = get_aliased_name(model.response_component.response_term)

    # perform predictions on new data
    idata = model.predict(idata, data=comparisons_df, inplace=False)

    def _compute_contrast_estimate(
        contrast: Contrast,
        response: Response,
        comparisons_df: pd.DataFrame,
        idata: az.InferenceData,
    ) -> ContrastEstimate:
        """
        Computes the contrast comparison estimate and highest density interval
        for a given contrast and response.
        """
        function = comparison_functions[comparison_type]

        # subset draw by observation using contrast mask
        draws = {}
        for idx, val in enumerate(contrast.value):
            mask = np.array(comparisons_df[contrast.term] == contrast.value[idx])
            select_draw = idata.posterior[f"{response.name}_{response.target}"].sel(
                {f"{response.name}_obs": mask}
            )
            select_draw = select_draw.assign_coords(
                {f"{response.name}_obs": np.arange(len(select_draw.coords[f"{response.name}_obs"]))}
            )
            draws[val] = select_draw

        # iterate over pairwise combinations of contrast values
        pairwise_contrasts = list(itertools.combinations(contrast.value, 2))

        # compute mean comparison and HDI for each pairwise comparison
        comparison_mean = {}
        comparison_bounds = {}
        for idx, pair in enumerate(pairwise_contrasts):
            comparison_estimate = function(draws[pair[1]], draws[pair[0]])
            comparison_mean[pair] = comparison_estimate.mean(("chain", "draw"))

            if use_hdi:
                comparison_bounds[pair] = az.hdi(comparison_estimate, prob)
            else:
                comparison_bounds[pair] = comparison_estimate.quantile(
                    q=(lower_bound, upper_bound), dim=("chain", "draw")
                )

        return ContrastEstimate(comparison_mean, comparison_bounds)

    def _build_contrasts_df(
        contrast: Contrast,
        response: Response,
        comparisons_df: pd.DataFrame,
        idata: az.InferenceData,
        average_by,
    ) -> pd.DataFrame:
        """
        Builds a dataframe with the comparison values and highest density interval
        from ``_compute_contrast_estimate`` along with the contrast name, contrast value,
        and conditional values.
        """
        contrast_estimate = _compute_contrast_estimate(contrast, response, comparisons_df, idata)
        # if two contrast values, then can drop duplicates to build contrast_df
        if len(contrast.value) < 3:
            contrast_df = comparisons_df.drop_duplicates(list(conditional.values())).reset_index(
                drop=True
            )
            contrast_df = contrast_df.drop(columns=contrast.term)
            num_rows = contrast_df.shape[0]
            contrast_df.insert(0, "term", contrast.term)
            contrast_df.insert(
                1, "contrast", list(np.tile(contrast.value, num_rows).reshape(num_rows, 2))
            )
            contrast_df["estimate"] = contrast_estimate.comparison[tuple(contrast.value)].to_numpy()

            if use_hdi:
                contrast_df[f"hdi_{lower_bound}%"] = (
                    contrast_estimate.hdi[tuple(contrast.value)][
                        f"{response.name}_{response.target}"
                    ]
                    .sel(hdi="lower")
                    .values
                )
                contrast_df[f"hdi_{upper_bound}%"] = (
                    contrast_estimate.hdi[tuple(contrast.value)][
                        f"{response.name}_{response.target}"
                    ]
                    .sel(hdi="higher")
                    .values
                )
            else:
                contrast_df[f"lower_{lower_bound}%"] = contrast_estimate.hdi[
                    tuple(contrast.value)
                ].sel(quantile=lower_bound)
                contrast_df[f"upper_{upper_bound}%"] = contrast_estimate.hdi[
                    tuple(contrast.value)
                ].sel(quantile=upper_bound)

        # if > 2 contrast values, then need the full dataframe to build contrast_df
        elif len(contrast.value) >= 3:
            num_rows = comparisons_df.shape[0]
            contrast_df = comparisons_df.drop(columns=contrast.term)
            contrast_df.insert(0, "term", contrast.term)
            contrast_keys = [list(elem) for elem in list(contrast_estimate.comparison.keys())]
            contrast_df.insert(1, "contrast", contrast_keys * (num_rows // len(contrast.value)))

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
                            contrast_estimate.hdi[tuple(pair)][f"{response.name}_{response.target}"]
                            .sel(hdi="lower")
                            .values
                        )
                    )
                    upper.append(
                        (
                            contrast_estimate.hdi[tuple(pair)][f"{response.name}_{response.target}"]
                            .sel(hdi="higher")
                            .values
                        )
                    )
                else:
                    lower.append(contrast_estimate.hdi[tuple(pair)].sel(quantile=lower_bound))
                    upper.append(contrast_estimate.hdi[tuple(pair)].sel(quantile=upper_bound))

            contrast_df[f"lower_{lower_bound}%"] = np.array(lower).flatten()
            contrast_df[f"upper_{upper_bound}%"] = np.array(upper).flatten()

        contrast_df["contrast"] = contrast_df["contrast"].apply(tuple).apply(lambda x: str(x))

        if average_by:
            if len(conditional) <= 1:
                raise ValueError(
                    "Must have more than one conditional covariate to average by comparisons"
                )

            contrast_df_avg = average_by_group(contrast_df, average_by)
            contrast_df_avg.insert(0, "term", contrast.term)
            contrast_df_avg.insert(
                1,
                "contrast",
                np.tile(contrast_df["contrast"].drop_duplicates(), len(contrast_df_avg)),
            )
            return contrast_df_avg
        else:
            return contrast_df

    contrast_vals = np.sort(np.unique(comparisons_df[contrast_name]))
    contrast_df = _build_contrasts_df(
        Contrast(contrast_name, contrast_vals),
        Response(response_name),
        comparisons_df,
        idata,
        average_by,
    )

    return contrast_df
