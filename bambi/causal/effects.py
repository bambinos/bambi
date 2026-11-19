"""Posterior g-computation over a user-specified empirical population."""

import numpy as np
import pandas as pd
import xarray as xr

from bambi.causal.types import CausalResult
from bambi.causal.validate import (
    interventions,
    reference_data,
    reference_weights,
    strata,
    validate_design,
    validate_family,
)
from bambi.interpret.effects import get_summary_stats
from bambi.utils import as_dataset, get_aliased_name


def _finite(array, name):
    if not np.isfinite(array.values).all():
        raise ValueError(f"Nonfinite posterior outcome parameter or estimand: {name}.")
    return array


def _parameter(model, posterior, name, template=None):
    component = model.parameters[name]
    key = get_aliased_name(component)
    if key in posterior:
        value = posterior[key]
    elif isinstance(getattr(component, "prior", None), (int, float, np.number)):
        if template is None:
            raise ValueError(f"Missing predicted parent parameter: {name}.")
        value = xr.full_like(template, float(component.prior))
    else:
        raise ValueError(f"Missing posterior likelihood parameter: {name}.")
    if template is not None:
        value, _ = xr.broadcast(value, template)
    if set(value.dims) != {"chain", "draw", "__obs__"}:
        raise ValueError(f"Unsupported likelihood parameter dimensions: {name}.")
    return _finite(value.transpose("chain", "draw", "__obs__"), name)


def _success_event(model, outcome, event):
    success = model.response_term.success
    training = model.data[outcome]
    if isinstance(training.dtype, pd.CategoricalDtype):
        levels = list(training.cat.categories)
    elif pd.api.types.is_numeric_dtype(training) and training.isin([0, 1]).all():
        levels = [0, 1]
    else:
        levels = list(pd.unique(training))
    if len(levels) != 2 or success not in levels:
        raise ValueError("Bernoulli response must have two known event labels.")
    chosen = success if event is None else event
    if chosen not in levels:
        raise ValueError("event must be one of the actual Bernoulli response labels.")
    return chosen, success


def _outcome_mean(model, posterior, family, outcome, event):
    parent = model.family.likelihood.parent
    mean = _parameter(model, posterior, parent)
    # Check every auxiliary parameter, including ones not needed in the mean formula.
    parameters = {parent: mean}
    for name in model.family.likelihood.params:
        if name != parent:
            value = _parameter(model, posterior, name, mean)
            if (value <= 0).any():
                raise ValueError(f"Likelihood parameter {name} must be positive.")
            parameters[name] = value
    chosen = None
    if family == "bernoulli":
        if ((mean < 0) | (mean > 1)).any():
            raise ValueError("Bernoulli outcome probabilities must lie in [0, 1].")
        chosen, success = _success_event(model, outcome, event)
        if chosen != success:
            mean = 1 - mean
    elif event is not None:
        raise ValueError("event is only supported for Bernoulli outcomes.")
    elif family == "lognormal":
        mean = np.exp(mean + parameters["sigma"] ** 2 / 2)
    elif family == "exgaussian":
        mean = mean + parameters["nu"]
    if family in {"gamma", "weibull", "exponential"} and (mean <= 0).any():
        raise ValueError("Positive outcome means must be greater than zero.")
    if family == "beta" and ((mean <= 0) | (mean >= 1)).any():
        raise ValueError("Beta outcome means must lie strictly between zero and one.")
    return _finite(mean, "outcome mean"), chosen


def _predict_means(
    model, idata, data, treatment, values, family, outcome, event, include_group_specific
):
    blocks = []
    for value in values:
        block = data.copy(deep=True)
        if isinstance(data[treatment].dtype, pd.CategoricalDtype):
            block[treatment] = pd.Categorical([value] * len(data), dtype=data[treatment].dtype)
        else:
            block[treatment] = value
        blocks.append(block)
    stacked = pd.concat(blocks, ignore_index=True)
    predicted = model.predict(
        idata=idata,
        data=stacked,
        kind="response_params",
        inplace=False,
        include_group_specific=include_group_specific,
    )
    posterior = as_dataset(predicted["posterior"])
    original = as_dataset(idata["posterior"])
    for dim in ("chain", "draw"):
        if dim not in posterior.coords or not posterior[dim].equals(original[dim]):
            raise ValueError(f"Prediction changed posterior {dim} coordinates.")
    mean, chosen = _outcome_mean(model, posterior, family, outcome, event)
    if mean.sizes["__obs__"] != len(stacked):
        raise ValueError("Prediction row count does not match positional intervention blocks.")
    return mean, chosen


def _standardize(mean, values, size, groups, weights):
    by_value = []
    for position, _ in enumerate(values):
        block = mean.isel(__obs__=slice(position * size, (position + 1) * size))
        block = block.assign_coords(__obs__=np.arange(size))
        by_group = []
        for indices, _ in groups:
            normalized = weights[indices] / weights[indices].sum()
            weighted = block.isel(__obs__=indices) * xr.DataArray(
                normalized,
                dims="__obs__",
                coords={"__obs__": indices},
            )
            by_group.append(weighted.sum("__obs__"))
        by_value.append(xr.concat(by_group, dim=pd.Index(range(len(groups)), name="stratum")))
    return xr.concat(by_value, dim=pd.Index(values, name="intervention")).transpose(
        "chain", "draw", "intervention", "stratum"
    )


def _contrast(contrast, reference, comparison):
    if comparison == "difference":
        result = contrast - reference
    elif comparison == "risk_ratio":
        if (reference <= 0).any():
            raise ValueError("risk_ratio is undefined for zero standardized reference risk.")
        result = contrast / reference
    else:
        if ((reference <= 0) | (reference >= 1)).any() or ((contrast <= 0) | (contrast >= 1)).any():
            raise ValueError(
                "odds_ratio requires standardized risks strictly between zero and one."
            )
        result = (contrast / (1 - contrast)) / (reference / (1 - reference))
    return _finite(result, comparison)


def _effect_rows(standardized, values, estimand, comparison, reference_value, groups, weights):
    raw_mean = estimand == "standardized_mean" or (
        estimand == "dose_response" and reference_value is None
    )
    reference = None
    if not raw_mean:
        reference = reference_value if estimand == "dose_response" else values[0]
    output_values = values if estimand in {"standardized_mean", "dose_response"} else values[1:]
    arrays, rows = [], []
    for value in output_values:
        estimate = standardized.sel(intervention=value, drop=True)
        if not raw_mean:
            baseline = standardized.sel(intervention=reference, drop=True)
            estimate = _contrast(estimate, baseline, comparison)
        for position, (indices, labels) in enumerate(groups):
            arrays.append(estimate.isel(stratum=position, drop=True))
            rows.append(
                {
                    "estimand": estimand,
                    "comparison": "mean" if raw_mean else comparison,
                    "reference": reference,
                    "value": value,
                    **labels,
                    "n_reference": len(indices),
                    "weight_sum": float(weights[indices].sum()),
                }
            )
    effects = xr.concat(arrays, dim=pd.Index(range(len(arrays)), name="__obs__"))
    return effects.transpose("chain", "draw", "__obs__"), rows


def g_computation(
    model,
    idata,
    *,
    treatment,
    outcome,
    adjustment_set,
    temporal_order,
    estimand,
    reference_population,
    values,
    comparison="difference",
    conditional=None,
    average_by=None,
    weights=None,
    membership=None,
    event=None,
    include_group_specific=True,
    prob=None,
    use_hdi=True,
    reference_value=None,
):
    """Standardize posterior outcome expectations over an explicit reference population.

    Parameters
    ----------
    model : bambi.Model
        Fitted native Bernoulli, Gaussian, Beta, Gamma, Weibull, Exponential,
        LogNormal, or ExGaussian model with an untransformed response.
    idata : xarray.DataTree
        Posterior samples from the fitted model.
    treatment, outcome : str
        Model treatment covariate and untransformed outcome name.
    adjustment_set : sequence of str
        All other model covariates, including grouping factors. This declaration
        is checked for consistency, not for causal identification.
    temporal_order : sequence of str
        Adjustment variables, treatment, and outcome in that order. Every variable
        must occur once, with outcome last.
    estimand : str
        ``standardized_mean``, ``dose_response``, ``ate``, ``att``, ``atu``,
        ``cate``, or ``gate``.
    reference_population : pandas.DataFrame
        Explicit empirical target population. Its rows and weights are held fixed
        when computing posterior uncertainty. Include all model covariates.
    values : sequence
        Ordered interventions; population effects require exactly two values,
        with the reference first. Dose-response requires numeric treatment.
    comparison : str, optional
        ``difference``, ``risk_ratio``, or ``odds_ratio``. Ratios compare
        standardized Bernoulli risks or odds, not average individual ratios.
    conditional : dict, optional
        CATE strata specified by observed original reference values. Rows are
        conditioned on these values before intervention; other covariates are
        not replaced by unconditional means or modes. The outcome and reserved
        summary column names cannot be conditioning or grouping variables.
    average_by : str or sequence of str, optional
        GATE groups defined by original reference-population columns.
    weights : array-like, optional
        Finite nonnegative positional reference weights, normalized within each
        selected stratum.
    membership : array-like of bool, optional
        Explicit ATT/ATU target membership. For binary reference treatment it
        may subset only the original treated or untreated cohort, respectively.
        For nonbinary reference treatment, membership must explicitly define the
        target population. Membership is determined before applying interventions.
    event : object, optional
        Bernoulli event label. Defaults to the model's actual success event.
    include_group_specific : bool, optional
        Include fitted group effects. False sets them to zero; it does not
        marginalize over their population distribution. New groups are not sampled.
    prob : float or sequence of float, optional
        Credible interval probabilities, defaulting to 0.94.
    use_hdi : bool, optional
        Use highest-density rather than equal-tailed intervals.
    reference_value : float, optional
        Dose-response reference for paired contrasts. Without it, return means.

    Returns
    -------
    CausalResult
        Summaries, actual paired posterior estimand draws, and provenance.

    Notes
    -----
    Causal interpretation requires user-justified exchangeability, positivity,
    consistency, temporal ordering, and an appropriate outcome model. This
    function cannot establish those assumptions or account for uncertainty in
    the empirical reference population. CATE uses original conditional reference
    populations; GATE uses original group membership. No predictive noise,
    posterior resampling, or new-group sampling is added.
    """
    if not isinstance(include_group_specific, (bool, np.bool_)):
        raise ValueError("include_group_specific must be boolean.")
    family = validate_family(model, outcome)
    covariates = validate_design(model, treatment, outcome, adjustment_set, temporal_order)
    data = reference_data(model, reference_population, covariates)
    values = interventions(
        model, data, treatment, values, estimand, comparison, family, reference_value
    )
    weights = reference_weights(weights, len(data))
    groups = strata(
        data, treatment, values, estimand, conditional, average_by, membership, weights, outcome
    )
    all_values = values.copy()
    if reference_value is not None and reference_value not in all_values:
        all_values.append(reference_value)
    mean, chosen = _predict_means(
        model, idata, data, treatment, all_values, family, outcome, event, include_group_specific
    )
    standardized = _finite(
        _standardize(mean, all_values, len(data), groups, weights), "standardized outcome mean"
    )
    effects, rows = _effect_rows(
        standardized, values, estimand, comparison, reference_value, groups, weights
    )
    probability = 0.94 if prob is None else prob
    probabilities = [probability] if np.isscalar(probability) else list(probability)
    if not probabilities or any(not 0 < item < 1 for item in probabilities):
        raise ValueError("prob must contain credible interval probabilities between zero and one.")
    stats = get_summary_stats(effects, probabilities, use_hdi)
    summary = pd.concat([pd.DataFrame(rows), stats.reset_index(drop=True)], axis=1)
    summary.insert(0, "term", treatment)
    draws = xr.DataTree.from_dict(
        {
            "effects": xr.Dataset({"estimate": effects}),
            "standardized": xr.Dataset({"mean": standardized}),
        }
    )
    metadata = {
        "treatment": treatment,
        "outcome": outcome,
        "adjustment_set": list(adjustment_set),
        "temporal_order": list(temporal_order),
        "family": family,
        "event": chosen,
        "success_event": model.response_term.success if family == "bernoulli" else None,
        "reference_index": list(reference_population.index),
        "group_effects": "included" if include_group_specific else "set_to_zero",
        "uncertainty": "conditional_on_fixed_reference_population",
        "values": list(values),
        "reference_value": reference_value,
        "comparison": comparison,
        "estimand": estimand,
        "conditional": {
            key: ([value] if np.isscalar(value) else list(value))
            for key, value in (conditional or {}).items()
        },
        "average_by": ([average_by] if isinstance(average_by, str) else list(average_by or [])),
        "reference_weights": weights.tolist(),
        "membership_supplied": membership is not None,
        "membership": None if membership is None else np.asarray(membership).tolist(),
        "strata": [
            {
                "labels": dict(labels),
                "positions": indices.tolist(),
                "original_index": [reference_population.index[int(index)] for index in indices],
                "normalized_weights": (weights[indices] / weights[indices].sum()).tolist(),
            }
            for indices, labels in groups
        ],
        "assumptions": ["exchangeability", "positivity", "consistency", "correct outcome model"],
    }
    return CausalResult(summary, draws, metadata)
