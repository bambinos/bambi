"""Validation of explicit causal inputs and empirical reference populations."""

# Exact native classes are intentional: subclasses can change expectation semantics.
# pylint: disable=unidiomatic-typecheck

from itertools import product

import numpy as np
import pandas as pd

from bambi.defaults.families import BUILTIN_FAMILIES
from bambi.interpret.utils import get_model_covariates, get_model_terms

SUPPORTED_FAMILIES = (
    "bernoulli",
    "gaussian",
    "beta",
    "gamma",
    "weibull",
    "exponential",
    "lognormal",
    "exgaussian",
)
ESTIMANDS = {"standardized_mean", "dose_response", "ate", "att", "atu", "cate", "gate"}
COMPARISONS = {"difference", "risk_ratio", "odds_ratio"}


def validate_family(model, outcome):
    """Return the supported native family, rejecting altered likelihoods and responses."""
    family = model.family
    name = next(
        (key for key in SUPPORTED_FAMILIES if type(family) is BUILTIN_FAMILIES[key]["family"]),
        None,
    )
    if name is None:
        raise ValueError(
            f"Unsupported family {type(family).__name__}: expected a supported native family."
        )
    spec = BUILTIN_FAMILIES[name]["likelihood"]
    likelihood = family.likelihood
    if (
        likelihood.dist is not None
        or likelihood.name != spec["name"]
        or list(likelihood.params) != spec["params"]
        or likelihood.parent != spec["parent"]
    ):
        raise ValueError("Unsupported custom or altered native likelihood.")
    response = model.response_term
    wrappers = ("is_censored", "is_constrained", "is_truncated", "is_weighted")
    components = response.term.components
    if (
        response.name != outcome
        or any(getattr(response, key) for key in wrappers)
        or len(components) != 1
        or type(components[0]).__name__ != "Variable"
        or components[0].name != outcome
    ):
        raise ValueError("outcome must name a single untransformed, unwrapped model response.")
    return name


def validate_design(model, treatment, outcome, adjustment_set, temporal_order):
    """Check declared variables and ordering without claiming causal identification."""
    covariates = list(get_model_covariates(model))
    if treatment not in covariates:
        raise ValueError("treatment must be a model covariate.")
    adjustment = list(adjustment_set)
    if len(adjustment) != len(set(adjustment)) or treatment in adjustment:
        raise ValueError("adjustment_set must contain unique covariates other than treatment.")
    expected = set(covariates) - {treatment}
    if set(adjustment) != expected:
        missing, extra = sorted(expected - set(adjustment)), sorted(set(adjustment) - expected)
        raise ValueError(f"adjustment_set mismatch: missing={missing}, extra={extra}.")
    order = list(temporal_order)
    required = expected | {treatment, outcome}
    if len(order) != len(required) or set(order) != required or order[-1] != outcome:
        raise ValueError(
            "temporal_order must list every adjustment, treatment and outcome once, "
            "with outcome last."
        )
    post = [key for key in adjustment if order.index(key) > order.index(treatment)]
    if post:
        raise ValueError(f"Adjustment variables occur after treatment: {post}.")
    return covariates


def _categorical_levels(model):
    """Identify categorical raw covariates from training data and fitted design terms."""
    names = {
        name
        for name in get_model_covariates(model)
        if (
            isinstance(model.data[name].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(model.data[name])
            or pd.api.types.is_string_dtype(model.data[name])
            or pd.api.types.is_bool_dtype(model.data[name])
        )
    }
    pending = list(get_model_terms(model).values())
    seen = set()
    while pending:
        term = pending.pop()
        if id(term) in seen:
            continue
        seen.add(id(term))
        pending.extend(getattr(term, "components", []))
        if hasattr(term, "factor"):
            pending.append(term.factor)
        if hasattr(term, "expr"):
            pending.append(term.expr)
        if getattr(term, "kind", None) != "categoric":
            continue
        name = getattr(term, "name", None)
        if name in model.data:
            names.add(name)
        call = getattr(term, "call", None)
        if call is not None:
            arguments = list(call.args) + list(call.kwargs.values())
            for argument in arguments:
                name = getattr(argument, "name", None)
                if name in model.data:
                    names.add(name)
    return {
        name: (
            list(model.data[name].cat.categories)
            if isinstance(model.data[name].dtype, pd.CategoricalDtype)
            else list(pd.unique(model.data[name]))
        )
        for name in names
    }


def reference_data(model, reference_population, covariates):
    """Copy reference rows positionally and retain the training categorical levels."""
    if not isinstance(reference_population, pd.DataFrame) or reference_population.empty:
        raise ValueError("reference_population must be a nonempty pandas DataFrame.")
    if not reference_population.columns.is_unique:
        raise ValueError("reference_population must have unique column names.")
    missing = set(covariates) - set(reference_population.columns)
    if missing:
        raise ValueError(f"reference_population is missing model covariates: {sorted(missing)}.")
    data = reference_population.copy(deep=True).reset_index(drop=True)
    if data[covariates].isna().any().any():
        raise ValueError("reference_population model covariates must not contain missing values.")
    levels = _categorical_levels(model)
    for key in covariates:
        if key in levels and not data[key].isin(levels[key]).all():
            raise ValueError(f"Unknown categorical levels in reference_population: {key}.")
        dtype = model.data[key].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            if not data[key].isin(dtype.categories).all():
                raise ValueError(f"Unknown categorical levels in reference_population: {key}.")
            data[key] = pd.Categorical(data[key], dtype=dtype)
    return data


def interventions(model, data, treatment, values, estimand, comparison, family, reference_value):
    """Validate ordered intervention values and their outcome-scale contrasts."""
    if estimand not in ESTIMANDS or comparison not in COMPARISONS:
        raise ValueError("Unsupported estimand or comparison.")
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be an ordered sequence of unique intervention values.")
    values = list(values)
    if not values or pd.Series(values).isna().any() or len(pd.unique(values)) != len(values):
        raise ValueError("values must contain unique, nonmissing intervention values.")
    if estimand not in {"standardized_mean", "dose_response"} and len(values) != 2:
        raise ValueError("Population effects require exactly two ordered values.")
    levels = _categorical_levels(model)
    if estimand == "dose_response":
        if treatment in levels:
            raise ValueError("dose_response requires a numeric noncategorical treatment.")
        if (
            not pd.api.types.is_numeric_dtype(data[treatment])
            or len(values) < 2
            or not np.isfinite(np.asarray(values, dtype=float)).all()
        ):
            raise ValueError("dose_response requires numeric treatment and two finite values.")
        if reference_value is not None and not np.isfinite(float(reference_value)):
            raise ValueError("reference_value must be finite.")
    elif reference_value is not None:
        raise ValueError("reference_value is only supported for dose_response.")
    if comparison != "difference" and family != "bernoulli":
        raise ValueError("risk_ratio and odds_ratio require a native Bernoulli outcome.")
    if estimand == "standardized_mean" or (estimand == "dose_response" and reference_value is None):
        if comparison != "difference":
            raise ValueError("Raw standardized means do not have a risk or odds comparison.")
    dtype = data[treatment].dtype
    all_values = values + ([] if reference_value is None else [reference_value])
    if treatment in levels:
        if not pd.Series(all_values).isin(levels[treatment]).all():
            raise ValueError("Intervention contains an unknown treatment categorical level.")
    elif pd.api.types.is_numeric_dtype(dtype):
        if not np.isfinite(np.asarray(all_values, dtype=float)).all():
            raise ValueError("Numeric intervention values must be finite.")
    return values


def reference_weights(weights, size):
    """Return finite nonnegative positional reference weights."""
    result = np.ones(size) if weights is None else np.asarray(weights, dtype=float).copy()
    if result.shape != (size,) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("weights must be finite, nonnegative and match reference rows.")
    return result


def strata(
    data, treatment, values, estimand, conditional, average_by, membership, weights, outcome
):
    """Select original reference rows before any intervention changes treatment."""
    selected = np.ones(len(data), dtype=bool)
    if estimand in {"att", "atu"}:
        if membership is None:
            observed = list(pd.unique(data[treatment]))
            if len(observed) != 2 or set(observed) != set(values):
                raise ValueError("ATT/ATU require explicit membership for a nonbinary reference.")
            selected = np.asarray(data[treatment] == values[1 if estimand == "att" else 0])
        else:
            selected = np.asarray(membership)
            if selected.dtype != np.dtype(bool) or selected.shape != (len(data),):
                raise ValueError("membership must be a positional boolean array matching rows.")
            observed = list(pd.unique(data[treatment]))
            if len(observed) == 2 and set(observed) == set(values):
                cohort = np.asarray(data[treatment] == values[1 if estimand == "att" else 0])
                if (selected & ~cohort).any():
                    raise ValueError("membership must select only the original ATT/ATU cohort.")
    elif membership is not None:
        raise ValueError("membership is only supported for ATT or ATU.")
    if estimand != "cate" and conditional is not None:
        raise ValueError("conditional is only supported for CATE.")
    if estimand != "gate" and average_by is not None:
        raise ValueError("average_by is only supported for GATE.")
    reserved = {
        "term",
        "estimand",
        "comparison",
        "reference",
        "value",
        "n_reference",
        "weight_sum",
        "estimate",
    }
    selectors = list(conditional) if isinstance(conditional, dict) else []
    if average_by is not None:
        selectors += [average_by] if isinstance(average_by, str) else list(average_by)
    if any(not isinstance(key, str) for key in selectors):
        raise ValueError("Conditioning/grouping columns must be named by strings.")
    if outcome in selectors:
        raise ValueError("The outcome cannot define CATE conditioning or GATE groups.")
    if any(key in reserved or key.startswith(("lower_", "upper_")) for key in selectors):
        raise ValueError("Conditioning/grouping column collides with a reserved summary column.")
    groups = []
    if estimand == "cate":
        if not isinstance(conditional, dict) or not conditional:
            raise ValueError("CATE requires conditional observed reference strata.")
        keys = list(conditional)
        if treatment in keys or any(key not in data for key in keys):
            raise ValueError("conditional must name original non-treatment reference columns.")
        levels = []
        for key, requested in conditional.items():
            requested = [requested] if np.isscalar(requested) else list(requested)
            if not requested or pd.Series(requested).isna().any():
                raise ValueError("conditional values must be nonempty and nonmissing.")
            levels.append(requested)
        for combination in product(*levels):
            mask = selected.copy()
            for key, value in zip(keys, combination):
                mask &= np.asarray(data[key] == value)
            groups.append((np.flatnonzero(mask), dict(zip(keys, combination))))
    elif estimand == "gate":
        keys = [average_by] if isinstance(average_by, str) else list(average_by or [])
        if (
            not keys
            or treatment in keys
            or len(set(keys)) != len(keys)
            or any(key not in data for key in keys)
            or data[keys].isna().any().any()
        ):
            raise ValueError("GATE requires nonmissing original non-treatment grouping columns.")
        positions = {}
        for position, labels in enumerate(data[keys].itertuples(index=False, name=None)):
            positions.setdefault(labels, []).append(position)
        for labels, indices in positions.items():
            groups.append((np.asarray(indices), dict(zip(keys, labels))))
    else:
        groups = [(np.flatnonzero(selected), {})]
    for indices, _ in groups:
        total = weights[indices].sum()
        if not len(indices) or not np.isfinite(total) or total <= 0:
            raise ValueError("Reference stratum is empty or has zero total weight.")
    return groups
