# pylint: disable = protected-access
# pylint: disable = too-many-function-args
# pylint: disable = too-many-nested-blocks
from dataclasses import dataclass
import itertools
from typing import Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import bambi as bmb
from bambi.utils import listify, get_aliased_name
from bambi.plots.create_data import create_cap_data, create_comparisons_data
from bambi.plots.utils import identity 



def predictions(
    model: bmb.Model,
    idata: az.InferenceData,
    covariates: Union[str, dict, list],
    target: str = "mean",
    pps: bool = False,
    use_hdi: bool = True,
    hdi_prob=None,
    transforms=None
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
        If a sequence, the first variable is taken as the main variable,
        mapped to the horizontal axis. If present, the second name is a coloring/grouping variable,
        and the third is mapped to different plot panels.
        If a dictionary, keys must be taken from ("horizontal", "color", "panel") and the values
        are the names of the variables.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    pps: bool, optional
        Whether to plot the posterior predictive samples. Defaults to ``False``.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    hdi_prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.

    Returns
    -------


    Raises
    ------

    """
    
    if hdi_prob is None:
        hdi_prob = az.rcParams["stats.hdi_prob"]

    if not 0 < hdi_prob < 1:
        raise ValueError(f"'hdi_prob' must be greater than 0 and smaller than 1. It is {hdi_prob}.")

    cap_data = create_cap_data(model, covariates)

    if transforms is None:
        transforms = {}

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
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[response_name].T
    elif use_hdi:
        y_hat_bounds = az.hdi(y_hat, hdi_prob)[f"{response_name}_{target}"].T
    else:
        lower_bound = round((1 - hdi_prob) / 2, 4)
        upper_bound = 1 - lower_bound
        y_hat_bounds = y_hat.quantile(q=(lower_bound, upper_bound), dim=("chain", "draw"))
    
    cap_data["estimate"] = y_hat_mean
    cap_data["hdi_3%"] = y_hat_bounds[0]
    cap_data["hdi_97%"] = y_hat_bounds[1]

    return cap_data


@dataclass
class Comparison:
    model: bmb.Model
    contrast_predictor: Union[str, dict, list]
    conditional: Union[str, dict, list]


@dataclass
class Contrast:
    term: Union[str, list]
    value: Union[list, np.ndarray, None]


@dataclass
class Response:
    name: str
    target: str


@dataclass
class ContrastEstimate:
    comparison: np.ndarray
    hdi: xr.Dataset


def comparisons(
        model: bmb.Model,
        idata: az.InferenceData,
        contrast_predictor: Union[str, dict, list],
        conditional: Union[str, dict, list],
        comparison_type: str = "diff",
        target: str = "mean",
        use_hdi: bool = True,
        hdi_prob=None,
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
    contrast_predictor : str, dict, list
        The predictor name whose contrast we would like to compare.
    conditional : str, dict, list
        The covariates we would like to condition on.
    comparison_type : str, optional
        The type of comparison to plot. Defaults to 'diff'.
    target : str
        Which model parameter to plot. Defaults to 'mean'. Passing a parameter into target only
        works when pps is False as the target may not be available in the posterior predictive
        distribution.
    use_hdi : bool, optional
        Whether to compute the highest density interval (defaults to True) or the quantiles.
    hdi_prob : float, optional
        The probability for the credibility intervals. Must be between 0 and 1. Defaults to 0.94.
        Changing the global variable ``az.rcParam["stats.hdi_prob"]`` affects this default.
    legend : bool, optional
        Whether to automatically include a legend in the plot. Defaults to ``True``.
    transforms : dict, optional
        Transformations that are applied to each of the variables being plotted. The keys are the
        name of the variables, and the values are functions to be applied. Defaults to ``None``.
    """

    covariate_kinds = ("horizontal", "color", "panel")
    # if not dict, then user did not pass values to condition on
    if not isinstance(conditional, dict):
        conditional = listify(conditional)
        conditional = dict(zip(covariate_kinds, conditional))
        comparisons_df = create_comparisons_data(
            Comparison(
                model, 
                contrast_predictor,
                conditional
            ),
            user_passed=False
        )
    # if dict, user passed values to condition on
    elif isinstance(conditional, dict):
        comparisons_df = create_comparisons_data(
            Comparison(
                model, 
                contrast_predictor,
                conditional
            ),
            user_passed=True
        )
        conditional = {k: listify(v) for k, v in conditional.items()}
        conditional = dict(zip(covariate_kinds, conditional))
    
    if isinstance(contrast_predictor, dict):
        contrast_name, contrast = next(iter(contrast_predictor.items()))
        if len(contrast) > 2:
            raise UserWarning(
                f"Length of contrast values must be 1. It is {len(contrast)}."
            )
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

    # perform predictions on new data
    idata = model.predict(idata, data=comparisons_df, inplace=False)

        
    def _compute_contrast_estimate(
            contrast: Contrast,
            response: Response,
            comparisons_df: pd.DataFrame,
            idata: az.InferenceData,
            comparison_type: str = comparison_type
    ) -> ContrastEstimate:
        """
        """
        assert comparison_type in ("diff", "ratio"), \
            "comparison_type must be 'diff' or 'ratio'"
        
        # subset draw by observation using contrast mask
        draws = {}
        for idx, val in enumerate(contrast.value):
            mask = np.array(comparisons_df[contrast.term] == contrast.value[idx])
            select_draw = (idata
                        .posterior[f"{response.name}_{response.target}"]
                        .sel({f"{response.name}_obs": mask})
            )
            select_draw = select_draw.assign_coords(
                {f"{response.name}_obs": np.arange(len(select_draw.coords[f"{response.name}_obs"]))}
            )
            draws[val] = select_draw
        
        # iterate over pairwise combinations of contrast values
        pairwise_contrasts = list(itertools.combinations(contrast.value, 2))

        # choose which function to use for comparison
        functions = {
            "diff": lambda x, y: x - y,
            "ratio": lambda x, y: x / y
        }
        function = functions[comparison_type]

        # compute mean comparison and HDI for each pairwise comparison
        mean_comparison = {}
        for idx, pair in enumerate(pairwise_contrasts):
            mean_comparison[pair] = function(draws[pair[0]], draws[pair[1]]).mean(("chain", "draw")) * -1
            hdi = az.hdi(function(draws[pair[0]], draws[pair[1]]), hdi_prob) * -1
        mean_comparison = np.array(list(mean_comparison.values())).flatten()

        return ContrastEstimate(mean_comparison, hdi)


    def _build_contrasts_df(
        contrast: Contrast,
        response: Response,
        comparisons_df: pd.DataFrame,
        idata: az.InferenceData
    ) -> pd.DataFrame:
        """
        """
        contrast_estimate = _compute_contrast_estimate(
            contrast,
            response,
            comparisons_df,
            idata,
            comparison_type
        )
        
        # build contrast dataframe
        contrast_df = (comparisons_df
                    .drop_duplicates(list(conditional.values()))
                    .reset_index(drop=True)
        )

        contrast_df = contrast_df.drop(columns=contrast.term)
        N = contrast_df.shape[0]
        contrast_df["term"] = contrast.term
        contrast_df["contrast"] = list(np.tile(contrast.value, N).reshape(N, 2))
        contrast_df["estimate"] = contrast_estimate.comparison
        contrast_df["hdi_3%"] = (contrast_estimate.hdi[f"{response_name}_{target}"]
                                 .sel(hdi="lower").values)
        contrast_df["hdi_97%"] = (contrast_estimate.hdi[f"{response_name}_{target}"]
                                  .sel(hdi="higher").values)

        return contrast_df
    

    contrast_vals = np.sort(np.unique(comparisons_df[contrast_name]))
    contrast_df = _build_contrasts_df(
        Contrast(contrast_name, contrast_vals),
        Response(response_name, target),
        comparisons_df,
        idata
    )

    return contrast_df
