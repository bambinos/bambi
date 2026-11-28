"""Tests for posterior g-computation in bambi.causal."""

from copy import deepcopy
import random
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import bambi as bmb
from bambi.causal import CausalResult, g_computation, plot_effects
from bambi.causal.validate import interventions

matplotlib.use("Agg")


@pytest.fixture
def reference():
    return pd.DataFrame({"t": [0, 0, 1, 1], "z": [0, 0, 1, 1], "y": [1, 2, 3, 4]})


def posterior(theta=None):
    values = np.ones((2, 3)) if theta is None else np.asarray(theta).reshape(2, 3)
    ds = xr.Dataset(
        {"theta": (("chain", "draw"), values)}, coords={"chain": [11, 23], "draw": [2, 5, 9]}
    )
    return xr.DataTree.from_dict({"posterior": ds})


def parameter_name(model, name):
    components = {**model.conditional_parameters, **model.marginal_parameters}
    component = components.get(name)
    return component.alias if component is not None and component.alias else name


def attach_prediction(monkeypatch, model, mean_function, *, parameters=None, corrupt=None, omit=()):
    """Replace `model.predict` with known response parameters for each row."""
    calls = []

    def predict(idata, *, data, kind, inplace, include_group_specific):
        assert kind == "response_params" and inplace is False
        calls.append((data.copy(deep=True), include_group_specific))
        source = idata["posterior"].ds
        theta = source["theta"].values[..., None]
        values = np.broadcast_to(mean_function(data, theta), (2, 3, len(data))).copy()
        ds = source.copy(deep=True)
        parent = model.family.likelihood.parent
        arrays = {name: 1.0 for name in model.family.likelihood.params if name != parent}
        arrays[parent] = values
        arrays.update(parameters or {})
        for name, value in arrays.items():
            if name in omit:
                continue
            array = np.broadcast_to(value, (2, 3, len(data))).copy()
            ds[parameter_name(model, name)] = (("chain", "draw", "__obs__"), array)
        result = xr.DataTree.from_dict({"posterior": ds})
        return corrupt(result) if corrupt is not None else result

    monkeypatch.setattr(model, "predict", predict)
    return calls


def linear_mean(data, theta):
    z = data["z"].to_numpy()
    t = data["t"].to_numpy()
    return 1 + z + theta * (1 + 3 * z) * t


def compute(model, idata, reference, **kwargs):
    arguments = dict(
        treatment="t",
        outcome="y",
        adjustment_set=["z"],
        temporal_order=["z", "t", "y"],
        estimand="ate",
        reference_population=reference,
        values=[0, 1],
    )
    arguments.update(kwargs)
    return g_computation(model, idata, **arguments)


def effects(result):
    assert isinstance(result, CausalResult)
    draws = result.draws["effects"].ds["estimate"]
    assert draws.dims == ("chain", "draw", "__obs__")
    assert draws.sizes["__obs__"] == len(result.summary)
    return draws.values


def assert_draws(result, expected):
    actual = effects(result)
    target = np.broadcast_to(np.asarray(expected), actual.shape)
    np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.summary["estimate"], actual.mean(axis=(0, 1)))


def linear_case(monkeypatch, reference, theta=None):
    model = bmb.Model("y ~ t * z", reference)
    idata = posterior(theta)
    calls = attach_prediction(monkeypatch, model, linear_mean)
    return model, idata, calls


@pytest.mark.parametrize("estimand,multiplier", [("ate", 2.5), ("att", 4), ("atu", 1)])
def test_population_effects_are_paired_draws(monkeypatch, reference, estimand, multiplier):
    theta = np.array([[0.8, 1, 1.2], [1.4, 1.6, 1.8]])
    model, idata, calls = linear_case(monkeypatch, reference, theta)
    result = compute(model, idata, reference, estimand=estimand)
    assert_draws(result, theta[..., None] * multiplier)
    assert len(calls) == 1
    np.testing.assert_array_equal(result.draws["effects"].coords["chain"], [11, 23])
    np.testing.assert_array_equal(result.draws["effects"].coords["draw"], [2, 5, 9])
    assert result.metadata["uncertainty"] == "conditional_on_fixed_reference_population"


def test_weighted_reference_and_membership_weights_renormalize(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)
    weights = [3, 3, 1, 1]
    weighted = compute(model, idata, reference, weights=weights)
    assert_draws(weighted, 1.75)
    assert weighted.metadata["reference_weights"] == weights
    assert weighted.metadata["strata"][0]["positions"] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        weighted.metadata["strata"][0]["normalized_weights"], [3 / 8, 3 / 8, 1 / 8, 1 / 8]
    )
    treated = [False, False, True, True]
    untreated = [True, True, False, False]
    assert_draws(
        compute(model, idata, reference, estimand="att", membership=treated, weights=weights), 4
    )
    assert_draws(
        compute(model, idata, reference, estimand="atu", membership=untreated, weights=weights), 1
    )


def test_standardized_mean_and_ordered_dose_response(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)
    assert_draws(compute(model, idata, reference, estimand="standardized_mean", values=[1]), 4)
    result = compute(model, idata, reference, estimand="dose_response", values=[0, 1, 3])
    assert_draws(result, np.array([1.5, 4, 9]))
    assert result.summary["value"].tolist() == [0, 1, 3]
    assert set(result.summary["comparison"]) == {"mean"}
    standardized = result.draws["standardized"].ds["mean"]
    assert standardized.dims == ("chain", "draw", "intervention", "stratum")
    np.testing.assert_allclose(standardized.values[..., 0], np.broadcast_to([1.5, 4, 9], (2, 3, 3)))


def test_dose_reference_need_not_be_in_grid(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)
    result = compute(
        model, idata, reference, estimand="dose_response", values=[3, 0], reference_value=1
    )
    assert_draws(result, np.array([5, -2.5]))
    assert result.summary["reference"].tolist() == [1, 1]
    assert result.summary["value"].tolist() == [3, 0]


def test_cate_uses_observed_strata_not_unconditional_covariates(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)
    result = compute(model, idata, reference, estimand="cate", conditional={"z": [1, 0]})
    observed = dict(zip(result.summary["z"], effects(result).mean(axis=(0, 1))))
    assert observed == {1: 4, 0: 1}
    assert result.summary["n_reference"].tolist() == [2, 2]


def test_cate_preserves_associated_adjustment_distribution(monkeypatch, reference):
    frame = reference.assign(w=[1, 3, 8, 12])
    model = bmb.Model("y ~ t * w + z", frame)
    attach_prediction(
        monkeypatch, model, lambda data, theta: theta * data.t.to_numpy() * data.w.to_numpy()
    )
    result = compute(
        model,
        posterior(),
        frame,
        adjustment_set=["z", "w"],
        temporal_order=["z", "w", "t", "y"],
        estimand="cate",
        conditional={"z": [0, 1]},
    )
    assert dict(zip(result.summary.z, effects(result).mean(axis=(0, 1)))) == {0: 2, 1: 10}


def test_gate_aggregates_original_groups_and_retains_order(monkeypatch, reference):
    frame = reference.iloc[[2, 0, 3, 1]].copy()
    model, idata, _ = linear_case(monkeypatch, frame)
    result = compute(model, idata, frame, estimand="gate", average_by="z")
    assert result.summary.z.tolist() == [1, 0]
    assert_draws(result, np.array([4, 1]))
    pooled = np.average(effects(result), axis=-1, weights=result.summary.n_reference)
    np.testing.assert_allclose(pooled, 2.5)


def binary_case(monkeypatch, reference, categorical=False, reverse=False):
    frame = reference.assign(y=[0, 1, 0, 1])
    if categorical:
        categories = ["yes", "no"] if reverse else ["no", "yes"]
        frame["y"] = pd.Categorical(["no", "yes", "no", "yes"], categories=categories)
    model = bmb.Model("y ~ t + z", frame, family="bernoulli")
    positive_event = "yes" if categorical else 1

    def risk(data, theta):
        p0 = np.where(data.z.to_numpy() == 0, 0.1, 0.8)
        p1 = 2 * p0 / (1 + p0)
        positive_risk = np.where(data.t.to_numpy() == 0, p0, p1) + theta * 0
        return positive_risk if model.response_term.success == positive_event else 1 - positive_risk

    attach_prediction(monkeypatch, model, risk)
    return model, posterior(), frame


@pytest.mark.parametrize(
    "comparison,expected",
    [
        ("difference", 53 / 99 - 9 / 20),
        ("risk_ratio", 1060 / 891),
        ("odds_ratio", 583 / 414),
    ],
)
def test_binary_marginal_scales_are_not_unit_ratio_averages(
    monkeypatch, reference, comparison, expected
):
    model, idata, frame = binary_case(monkeypatch, reference)
    result = compute(model, idata, frame, comparison=comparison, event=1)
    assert_draws(result, expected)
    if comparison == "risk_ratio":
        assert not np.isclose(expected, np.mean([20 / 11, 10 / 9]))
    if comparison == "odds_ratio":
        assert not np.isclose(expected, 2)


@pytest.mark.parametrize(
    "categorical,reverse,event,opposite",
    [
        (False, False, 1, 0),
        (True, False, "yes", "no"),
        (True, True, "yes", "no"),
    ],
)
def test_bernoulli_success_and_complement_events(
    monkeypatch, reference, categorical, reverse, event, opposite
):
    model, idata, frame = binary_case(monkeypatch, reference, categorical, reverse)
    success = compute(model, idata, frame, event=event)
    complement = compute(model, idata, frame, event=opposite)
    np.testing.assert_allclose(effects(success), -effects(complement))
    assert_draws(
        compute(model, idata, frame, estimand="standardized_mean", values=[0], event=opposite),
        11 / 20,
    )
    default = compute(model, idata, frame)
    explicit_success = compute(model, idata, frame, event=model.response_term.success)
    np.testing.assert_allclose(effects(default), effects(explicit_success))
    with pytest.raises(ValueError):
        compute(model, idata, frame, event="absent")


@pytest.mark.parametrize("estimand,comparison", [("att", "risk_ratio"), ("atu", "odds_ratio")])
def test_population_and_scale_are_independent(monkeypatch, reference, estimand, comparison):
    model, idata, frame = binary_case(monkeypatch, reference)
    z = 1 if estimand == "att" else 0
    p0 = [0.1, 0.8][z]
    p1 = 2 * p0 / (1 + p0)
    expected = p1 / p0 if comparison == "risk_ratio" else 2
    assert_draws(compute(model, idata, frame, estimand=estimand, comparison=comparison), expected)


def test_intervals_summarize_estimand_draws_after_averaging(monkeypatch, reference):
    theta = np.array([[0, 0.1, 0.2], [1, 3, 8]])
    model, idata, _ = linear_case(monkeypatch, reference, theta)
    result = compute(model, idata, reference, prob=0.8, use_hdi=False)
    sample = (2.5 * theta).ravel()
    lower = next(c for c in result.summary if c.startswith("lower_"))
    upper = next(c for c in result.summary if c.startswith("upper_"))
    np.testing.assert_allclose(
        result.summary[[lower, upper]].iloc[0], np.quantile(sample, [0.1, 0.9])
    )


@pytest.mark.parametrize("estimand", ["att", "atu"])
def test_nonbinary_membership_must_be_explicit(monkeypatch, reference, estimand):
    frame = reference.assign(t=[0.0, 0.5, 1.0, 2.0])
    model, idata, _ = linear_case(monkeypatch, frame)
    with pytest.raises(ValueError, match="membership"):
        compute(model, idata, frame, estimand=estimand)
    selected = [False, False, True, True] if estimand == "att" else [True, True, False, False]
    result = compute(model, idata, frame, estimand=estimand, membership=selected)
    assert_draws(result, 4 if estimand == "att" else 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "other"},
        {"treatment": "other"},
        {"adjustment_set": []},
        {"adjustment_set": ["z", "other"]},
        {"adjustment_set": ["z", "t"]},
        {"temporal_order": ["t", "z", "y"]},
        {"temporal_order": ["z", "t", "t", "y"]},
        {"temporal_order": ["z", "y", "t"]},
        {"conditional": {"z": [0]}},
        {"average_by": "z"},
        {"estimand": "cate"},
        {"estimand": "gate"},
        {"estimand": "ate", "values": [0]},
        {"estimand": "ate", "values": [0, 1, 2]},
        {"estimand": "dose_response", "values": [0, 0]},
        {"estimand": "dose_response", "values": [0, np.inf]},
        {"estimand": "unknown"},
        {"comparison": "unit_ratio"},
        {"comparison": "risk_ratio"},
    ],
)
def test_invalid_declarations_raise(monkeypatch, reference, changes):
    model, idata, _ = linear_case(monkeypatch, reference)
    with pytest.raises(ValueError):
        compute(model, idata, reference, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"weights": [-1, 1, 1, 1]},
        {"weights": [0, 0, 0, 0]},
        {"weights": [1, np.nan, 1, 1]},
        {"weights": [1, np.inf, 1, 1]},
        {"weights": [1e308] * 4},
        {"weights": [1, 2]},
        {"membership": [0, 1, 0, 1], "estimand": "att"},
        {"membership": [True, False], "estimand": "att"},
        {"membership": [False] * 4, "estimand": "att"},
        {"conditional": {"z": [99]}, "estimand": "cate"},
    ],
)
def test_invalid_weights_and_membership_raise(monkeypatch, reference, changes):
    model, idata, _ = linear_case(monkeypatch, reference)
    with pytest.raises(ValueError):
        compute(model, idata, reference, **changes)


def test_reference_positions_survive_duplicate_reordered_indexes(monkeypatch, reference):
    frame = reference.iloc[[3, 0, 2, 1]].copy()
    frame.index = [7, 7, -3, 12]
    model, idata, _ = linear_case(monkeypatch, frame)
    result = compute(model, idata, frame, estimand="att")
    assert_draws(result, 4)
    assert result.metadata["reference_index"] == [7, 7, -3, 12]
    assert result.metadata["strata"][0]["positions"] == [0, 2]
    assert result.metadata["strata"][0]["original_index"] == [7, -3]
    np.testing.assert_allclose(result.metadata["strata"][0]["normalized_weights"], [0.5, 0.5])


@pytest.mark.parametrize("problem", ["missing_column", "nan", "unknown_category", "empty"])
def test_invalid_reference_population_raises(monkeypatch, reference, problem):
    frame = reference.copy()
    frame["z"] = pd.Categorical(frame.z, categories=[0, 1])
    model, idata, _ = linear_case(monkeypatch, frame)
    modified = frame.copy()
    if problem == "missing_column":
        modified = modified.drop(columns="z")
    elif problem == "nan":
        modified.loc[0, "z"] = np.nan
    elif problem == "unknown_category":
        modified["z"] = [0, 1, 0, 2]
    else:
        modified = modified.iloc[:0]
    with pytest.raises(ValueError):
        compute(model, idata, modified)


@pytest.mark.parametrize("corruption", ["rows", "chain", "draw"])
def test_misaligned_predictions_raise(monkeypatch, reference, corruption):
    model = bmb.Model("y ~ t * z", reference)

    def corrupt(idata):
        ds = idata["posterior"].ds
        if corruption == "rows":
            ds = ds.isel(__obs__=slice(1, None))
        else:
            ds = ds.isel({corruption: slice(None, None, -1)})
        return xr.DataTree.from_dict({"posterior": ds})

    attach_prediction(monkeypatch, model, linear_mean, corrupt=corrupt)
    with pytest.raises(ValueError):
        compute(model, posterior(), reference)


@pytest.mark.parametrize(
    "family,parent,aux,expected",
    [
        ("gaussian", 0.4, {"sigma": 0.7}, 0.4),
        ("beta", 0.4, {"kappa": 3.0}, 0.4),
        ("gamma", 1.4, {"alpha": 2.3}, 1.4),
        ("weibull", 1.4, {"alpha": 2.3}, 1.4),
        ("exponential", 1.4, {}, 1.4),
        ("lognormal", 0.4, {"sigma": 0.7}, np.exp(0.4 + 0.7**2 / 2)),
        ("exgaussian", 0.4, {"sigma": 0.7, "nu": 0.6}, 1.0),
        ("bernoulli", 0.25, {}, 0.25),
    ],
)
def test_each_native_family_guard_and_outcome_expectation(
    monkeypatch, reference, family, parent, aux, expected
):
    frame = reference.assign(
        y=(
            [0, 1, 0, 1]
            if family == "bernoulli"
            else [0.1, 0.3, 0.6, 0.8] if family == "beta" else [1, 2, 3, 4]
        )
    )
    model = bmb.Model("y ~ t + z", frame, family=family)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + parent, parameters=aux)
    result = compute(model, posterior(), frame, estimand="standardized_mean", values=[0])
    assert_draws(result, expected)


@pytest.mark.parametrize(
    "family,parameters,expected",
    [
        ("lognormal", {"sigma": 2.0}, np.exp(2.5)),
        ("exgaussian", {"nu": 3.0, "sigma": 2.0}, 3.5),
    ],
)
def test_distributional_auxiliary_aliases_are_response_scale_parameters(
    monkeypatch, reference, family, parameters, expected
):
    formula = bmb.Formula("y ~ t + z", *(f"{name} ~ t + z" for name in parameters))
    model = bmb.Model(formula, reference, family=family)
    aliases = {name: {name: f"causal_{name}"} for name in ["mu", *parameters]}
    model.set_alias(aliases)
    attach_prediction(
        monkeypatch, model, lambda data, theta: theta * 0 + 0.5, parameters=parameters
    )
    result = compute(model, posterior(), reference, estimand="standardized_mean", values=[1])
    assert_draws(result, expected)


def test_custom_family_with_native_name_raises(monkeypatch, reference):
    native = bmb.Model("y ~ t + z", reference, family="gamma").family
    base = bmb.Family("gamma", deepcopy(native.likelihood), deepcopy(native.link))
    base.set_default_priors(deepcopy(native.default_priors))
    model = bmb.Model("y ~ t + z", reference, family=base)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + 1)
    with pytest.raises(ValueError):
        compute(model, posterior(), reference)
    custom = bmb.Model("y ~ t + z", reference)
    custom.family.likelihood.dist = pm.Normal
    attach_prediction(monkeypatch, custom, lambda data, theta: theta * 0 + 1)
    with pytest.raises(ValueError):
        compute(custom, posterior(), reference)


@pytest.mark.parametrize(
    "attribute,value",
    [("_params", ("mu", "sigma", "extra")), ("_parent", "sigma"), ("name", "StudentT")],
)
def test_altered_likelihood_raises(monkeypatch, reference, attribute, value):
    model, idata, _ = linear_case(monkeypatch, reference)
    setattr(model.family.likelihood, attribute, value)
    with pytest.raises(ValueError):
        compute(model, idata, reference)


@pytest.mark.parametrize(
    "formula",
    [
        "truncated(y, -5, 5) ~ t + z",
        "constrained(y, -5, 5) ~ t + z",
        "censored(y, status) ~ t + z",
        "weighted(y, w) ~ t + z",
        "np.log(y) ~ t + z",
    ],
)
def test_wrapped_or_transformed_response_raises(monkeypatch, reference, formula):
    frame = reference.assign(status=["none", "right", "none", "none"], w=[1, 2, 1, 2])
    model = bmb.Model(formula, frame)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + 1)
    with pytest.raises(ValueError):
        compute(model, posterior(), frame)


@pytest.mark.parametrize(
    "family",
    [
        "t",
        "wald",
        "laplace",
        "asymmetriclaplace",
        "vonmises",
        "poisson",
        "negativebinomial",
        "zero_inflated_poisson",
        "hurdle_poisson",
        "categorical",
        "cumulative",
    ],
)
def test_unsupported_family_raises(monkeypatch, reference, family):
    frame = reference.assign(y=[0, 1, 0, 1])
    if family in ["wald", "hurdle_poisson"]:
        frame["y"] = [1, 2, 3, 4]
    if family in ["categorical", "cumulative"]:
        frame["y"] = pd.Categorical(["a", "b", "a", "c"], ordered=True)
    model = bmb.Model("y ~ t + z", frame, family=family)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + 1)
    with pytest.raises(ValueError):
        compute(model, posterior(), frame)


@pytest.mark.parametrize(
    "family,parent,comparison",
    [
        ("bernoulli", -0.01, "difference"),
        ("bernoulli", 1.01, "difference"),
        ("bernoulli", 0, "risk_ratio"),
        ("bernoulli", 0, "odds_ratio"),
        ("bernoulli", 1, "odds_ratio"),
        ("gamma", 0, "difference"),
        ("weibull", -1, "difference"),
        ("exponential", 0, "difference"),
        ("beta", 0, "difference"),
        ("beta", 1, "difference"),
        ("gaussian", np.nan, "difference"),
        ("gaussian", np.inf, "difference"),
    ],
)
def test_response_mean_domains_are_checked_before_summaries(
    monkeypatch, reference, family, parent, comparison
):
    frame = reference.assign(
        y=(
            [0, 1, 0, 1]
            if family == "bernoulli"
            else [0.1, 0.3, 0.6, 0.8] if family == "beta" else [1, 2, 3, 4]
        )
    )
    model = bmb.Model("y ~ t + z", frame, family=family)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + parent)
    with pytest.raises(ValueError):
        compute(model, posterior(), frame, comparison=comparison)


def test_constant_numeric_auxiliary_broadcast_and_parent_alias(monkeypatch, reference):
    model = bmb.Model("y ~ t * z", reference, priors={"sigma": 2.0})
    model.set_alias({"mu": "location", "sigma": "scale", "y": "observed"})
    attach_prediction(monkeypatch, model, linear_mean, omit=("sigma",))
    assert_draws(compute(model, posterior(), reference), 2.5)


def test_family_subclass_raises(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)

    class DifferentGaussian(type(model.family)):
        pass

    model.family.__class__ = DifferentGaussian
    with pytest.raises(ValueError):
        compute(model, idata, reference)


@pytest.mark.parametrize(
    "family,formula",
    [
        ("binomial", "p(y, n) ~ t + z"),
        ("beta_binomial", "p(y, n) ~ t + z"),
        ("multinomial", "counts(y, n) ~ t + z"),
        ("dirichlet_multinomial", "counts(y, n) ~ t + z"),
    ],
)
def test_count_and_multivariate_responses_are_rejected(monkeypatch, reference, family, formula):
    frame = reference.assign(y=[0, 1, 0, 1], n=[2, 3, 2, 3])
    model = bmb.Model(formula, frame, family=family)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + 1)
    with pytest.raises(ValueError):
        compute(model, posterior(), frame)


def test_finite_weight_rescaling_preserves_every_draw(monkeypatch, reference):
    model, idata, _ = linear_case(monkeypatch, reference)
    weights = np.array([1, 2, 3, 4], dtype=float)
    first = compute(model, idata, reference, weights=weights)
    second = compute(model, idata, reference, weights=weights * 1e200)
    np.testing.assert_allclose(effects(first), effects(second), rtol=1e-12)
    assert_draws(first, 3.1)


@pytest.mark.parametrize("estimand,selector", [("cate", "conditional"), ("gate", "average_by")])
@pytest.mark.parametrize("column", ["estimate", "value", "n_reference", "lower_bad"])
def test_selector_names_cannot_overwrite_result_columns(
    monkeypatch, reference, estimand, selector, column
):
    frame = reference.rename(columns={"z": column})
    model = bmb.Model(f"y ~ t * {column}", frame)
    attach_prediction(monkeypatch, model, lambda data, theta: theta * 0 + 1)
    choice = {column: [0]} if selector == "conditional" else column
    with pytest.raises(ValueError, match="reserved"):
        compute(
            model,
            posterior(),
            frame,
            adjustment_set=[column],
            temporal_order=[column, "t", "y"],
            estimand=estimand,
            **{selector: choice},
        )


@pytest.mark.parametrize("estimand,selector", [("cate", "conditional"), ("gate", "average_by")])
def test_outcome_cannot_define_a_causal_stratum(monkeypatch, reference, estimand, selector):
    model, idata, _ = linear_case(monkeypatch, reference)
    choice = {"y": [1]} if selector == "conditional" else "y"
    with pytest.raises(ValueError, match="outcome"):
        compute(model, idata, reference, estimand=estimand, **{selector: choice})


@pytest.mark.parametrize("prob", [0, 1, -0.1, 1.1, np.nan, []])
def test_invalid_prob_raises(monkeypatch, reference, prob):
    model, idata, _ = linear_case(monkeypatch, reference)
    with pytest.raises(ValueError):
        compute(model, idata, reference, prob=prob)


@pytest.mark.parametrize("estimand", ["att", "atu"])
def test_explicit_binary_membership_preserves_original_cohort(monkeypatch, reference, estimand):
    model, idata, _ = linear_case(monkeypatch, reference)
    selected = [False, False, True, True] if estimand == "att" else [True, True, False, False]
    expected = 4 if estimand == "att" else 1
    assert_draws(compute(model, idata, reference, estimand=estimand, membership=selected), expected)
    subset = [False, False, True, False] if estimand == "att" else [True, False, False, False]
    assert_draws(compute(model, idata, reference, estimand=estimand, membership=subset), expected)


@pytest.mark.parametrize(
    "estimand,mask",
    [
        ("att", [True, False, False, False]),
        ("atu", [False, False, True, False]),
        ("att", [True, True, True, True]),
        ("atu", [True, True, True, True]),
        ("att", [True, False, True, False]),
        ("atu", [True, False, True, False]),
    ],
)
def test_explicit_binary_membership_cannot_include_the_opposite_cohort(
    monkeypatch, reference, estimand, mask
):
    model, idata, _ = linear_case(monkeypatch, reference)
    with pytest.raises(ValueError):
        compute(model, idata, reference, estimand=estimand, membership=mask)


@pytest.mark.parametrize(
    "family,parameters",
    [
        ("gaussian", {"sigma": 0}),
        ("lognormal", {"sigma": np.nan}),
        ("gamma", {"alpha": -1}),
        ("weibull", {"alpha": 0}),
        ("exgaussian", {"nu": -1}),
        ("beta", {"kappa": np.inf}),
    ],
)
def test_invalid_auxiliary_parameter_raises(monkeypatch, reference, family, parameters):
    model = bmb.Model("y ~ t + z", reference.assign(y=[0.1, 0.3, 0.6, 0.8]), family=family)
    attach_prediction(
        monkeypatch, model, lambda data, theta: theta * 0 + 0.4, parameters=parameters
    )
    with pytest.raises(ValueError):
        compute(model, posterior(), reference.assign(y=[0.1, 0.3, 0.6, 0.8]))


def deterministic_native_posterior(model, coefficients):
    """Fixed posterior draws for the common terms of the parent parameter."""
    arrays = {"Intercept": (("chain", "draw"), np.full((2, 3), 0.5))}
    for name, component in model.conditional_parameters.items():
        if name != model.family.likelihood.parent:
            continue
        for term_name, term in component.common_terms.items():
            width = 1 if np.asarray(term.data).ndim == 1 else np.asarray(term.data).shape[1]
            value = np.asarray(coefficients.get(term_name, np.ones(width)))
            if width == 1:
                arrays[term_name] = (("chain", "draw"), np.full((2, 3), value.item()))
            else:
                arrays[term_name] = (
                    ("chain", "draw", term_name + "_test_dim"),
                    np.broadcast_to(value, (2, 3, width)).copy(),
                )
    return xr.DataTree.from_dict(
        {"posterior": xr.Dataset(arrays, coords={"chain": [11, 23], "draw": [2, 5, 9]})}
    )


@pytest.mark.parametrize("expression", ["scale(t)", "center(t)", "bs(t, knots=knots, degree=1)"])
def test_native_interventions_reuse_training_transform_state(expression):
    frame = pd.DataFrame(
        {
            "t": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0],
            "z": [0, 1] * 4,
            "y": np.arange(8, dtype=float),
        }
    )
    model = bmb.Model(
        f"y ~ {expression} + z",
        frame,
        priors={"sigma": 1},
        extra_namespace={"knots": np.array([1.0, 2.0])},
    )
    transformed_name = next(
        name for name in model.conditional_parameters["mu"].common_terms if name != "z"
    )
    if expression.startswith("bs"):
        coefficients = {transformed_name: np.array([0.2, 0.4, 0.9]), "z": 0.3}
        expected = 0.9
        expected_at_constant = 0.5 + 0.3 * frame.z.mean() + 0.2 * 0.25
    else:
        coefficients = {transformed_name: 2.0, "z": 0.3}
        # Read the affine map from the training design instead of assuming a ddof.
        original = np.asarray(
            model.conditional_parameters["mu"].common_terms[transformed_name].data
        ).reshape(-1)
        origin = original[frame.t.to_numpy() == 0][0]
        slope = (original[frame.t.to_numpy() == 3][0] - origin) / 3
        np.testing.assert_allclose(original, origin + slope * frame.t.to_numpy())
        expected = 6 * slope
        expected_at_constant = 0.5 + 0.3 * frame.z.mean() + 2 * (origin + slope * 0.25)
    idata = deterministic_native_posterior(model, coefficients)
    # A constant intervention must reuse the training centering, scaling and knots.
    standardized = compute(model, idata, frame, estimand="standardized_mean", values=[0.25])
    assert_draws(standardized, expected_at_constant)
    result = compute(model, idata, frame, values=[0.0, 3.0])
    assert np.isfinite(effects(result)).all()
    assert_draws(result, expected)
    # Repeated reference rows must not change the transform either.
    repeated = pd.concat([frame.iloc[[3]], frame.iloc[[0]]] * 3)
    assert_draws(compute(model, idata, repeated, values=[0.0, 3.0]), expected)


def test_native_categorical_intervention_keeps_training_contrasts():
    frame = pd.DataFrame(
        {"t": pd.Categorical(["a", "b", "a", "b"]), "z": [0, 0, 1, 1], "y": [1, 2, 3, 4]}
    )
    model = bmb.Model("y ~ t + z", frame, priors={"sigma": 1})
    idata = deterministic_native_posterior(model, {"t": 2.0, "z": 0.3})
    assert_draws(compute(model, idata, frame, values=["a", "b"]), 2)
    with pytest.raises(ValueError):
        compute(model, idata, frame, values=["a", "unseen"])


@pytest.mark.parametrize("kind", ["object", "numeric_categorical"])
def test_categorical_levels_from_object_and_c(kind):
    frame = pd.DataFrame(
        {
            "t": ["a", "b", "a", "b"] if kind == "object" else [0, 1, 2, 0],
            "z": [0, 0, 1, 1],
            "y": [1, 2, 3, 4],
        }
    )
    formula = "y ~ t + z" if kind == "object" else "y ~ C(t) + z"
    values = ["a", "b"] if kind == "object" else [0, 2]
    model = bmb.Model(formula, frame, priors={"sigma": 1})
    name = next(name for name in model.conditional_parameters["mu"].common_terms if name != "z")
    coefficient = 2.0 if kind == "object" else np.array([1.5, 3.0])
    idata = deterministic_native_posterior(model, {name: coefficient, "z": 0.3})
    assert_draws(compute(model, idata, frame, values=values), 2 if kind == "object" else 3)
    unknown = "unseen" if kind == "object" else 3
    modified = frame.copy()
    modified.loc[0, "t"] = unknown
    with pytest.raises(ValueError):
        compute(model, idata, modified, values=values)
    with pytest.raises(ValueError):
        compute(model, idata, frame, values=[values[0], unknown])
    with pytest.raises(ValueError):
        compute(model, idata, frame, estimand="dose_response", values=values)


def test_inputs_and_global_rng_are_unchanged(monkeypatch, reference):
    model, idata, calls = linear_case(monkeypatch, reference)
    before_model = model.data.copy(deep=True)
    before_reference = reference.copy(deep=True)
    before_posterior = idata["posterior"].ds.copy(deep=True)
    numpy_state = deepcopy(np.random.get_state())
    python_state = random.getstate()
    first = compute(model, idata, reference, include_group_specific=False)
    second = compute(model, idata, reference, include_group_specific=False)
    for name in ["effects", "standardized"]:
        xr.testing.assert_identical(first.draws[name].ds, second.draws[name].ds)
    pd.testing.assert_frame_equal(model.data, before_model)
    pd.testing.assert_frame_equal(reference, before_reference)
    xr.testing.assert_identical(idata["posterior"].ds.copy(deep=True), before_posterior)
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    assert all(not include for _, include in calls)
    assert first.metadata["group_effects"] == "set_to_zero"


def test_plot_ratio_and_difference_reference_lines_and_facets(monkeypatch, reference):
    import matplotlib.pyplot as plt  # after matplotlib.use("Agg") at module level

    model, idata, frame = binary_case(monkeypatch, reference)
    for comparison, baseline in [("difference", 0), ("risk_ratio", 1), ("odds_ratio", 1)]:
        result = compute(
            model, idata, frame, estimand="gate", average_by="z", comparison=comparison
        )
        figure = plot_effects(result, main="value", group="z", panel="z").plot()._figure
        assert len(figure.axes) == 2
        for axis in figure.axes:
            horizontals = [np.asarray(line.get_ydata()) for line in axis.lines]
            assert any(values.size >= 1 and np.allclose(values, baseline) for values in horizontals)
        plt.close(figure)


@pytest.fixture(scope="module")
def fitted_gaussian():
    """Small NUTS fit of a Gaussian model with a group-specific intercept."""
    rng = np.random.default_rng(105501)
    n = 48
    group = np.repeat(np.arange(6).astype(str), 8)
    t = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    y = 0.7 + 0.9 * t + 0.4 * z + 0.3 * t * z + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "t": t, "z": z, "group": group})
    model = bmb.Model("y ~ t * z + (1 | group)", frame)
    idata = model.fit(
        tune=60,
        draws=60,
        chains=2,
        cores=1,
        random_seed=105501,
        target_accept=0.9,
        progressbar=False,
    )
    return model, idata, frame


@pytest.fixture(scope="module")
def fitted_bernoulli():
    """Small NUTS fit of a Bernoulli model with a logit link."""
    rng = np.random.default_rng(105502)
    n = 64
    t = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-0.5 + 0.9 * t + 0.4 * z)))
    frame = pd.DataFrame({"y": rng.binomial(1, p), "t": t, "z": z})
    model = bmb.Model("y ~ t + z", frame, family="bernoulli")
    idata = model.fit(
        tune=60,
        draws=60,
        chains=2,
        cores=1,
        random_seed=105502,
        target_accept=0.9,
        progressbar=False,
    )
    return model, idata, frame


@pytest.mark.parametrize("fixture,parent", [("fitted_gaussian", "mu"), ("fitted_bernoulli", "p")])
def test_fitted_models_match_predict(request, fixture, parent):
    model, idata, frame = request.getfixturevalue(fixture)
    before = idata["posterior"].ds.copy(deep=True)
    means = []
    for value in [0, 1]:
        predicted = model.predict(
            idata,
            data=frame.assign(t=value),
            kind="response_params",
            inplace=False,
            include_group_specific=True,
        )
        means.append(predicted["posterior"].ds[parent].transpose("chain", "draw", "__obs__"))
    kwargs = {}
    if parent == "mu":
        kwargs = dict(adjustment_set=["z", "group"], temporal_order=["z", "group", "t", "y"])
    result = compute(model, idata, frame, **kwargs)
    expected = (means[1] - means[0]).mean("__obs__").values[..., None]
    np.testing.assert_allclose(effects(result), expected)
    assert result.metadata["group_effects"] == "included"
    xr.testing.assert_identical(idata["posterior"].ds.copy(deep=True), before)
    if parent == "mu":
        with pytest.raises(ValueError, match="group|level"):
            compute(model, idata, frame.assign(group="unseen"), **kwargs)


def test_list_intervention_values_do_not_warn(reference):
    model = bmb.Model("y ~ t * z", reference)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = interventions(model, reference, "t", [0, 1], "ate", "difference", "gaussian", None)
    assert values == [0, 1]
