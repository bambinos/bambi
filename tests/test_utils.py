import pytest

import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace
from pymc.distributions.distribution import support_point
from pymc.distributions.shape_utils import change_dist_size
from scipy import stats

from bambi.utils import listify
from bambi.backend.pymc.links import cloglog, probit
from bambi.backend.pymc.data import shape_common_data
from bambi.backend.pymc.utils import (
    CureWeibull,
    LogLogistic,
    make_competing_risks_distribution,
    make_cure_distribution,
    make_weighted_distribution,
)
from bambi.transformations import CompetingRisks, censored, constrained, counts, truncated, weighted


def test_listify():
    assert listify(None) == []
    assert listify([1, 2, 3]) == [1, 2, 3]
    assert listify("giraffe") == ["giraffe"]


def test_shape_common_data_no_coords_single_column():
    data = np.arange(5)[:, np.newaxis]

    result = shape_common_data(data, {})

    assert result.shape == (5,)
    assert np.array_equal(result, np.arange(5))
    assert result.dtype == float


def test_shape_common_data_no_coords_multi_column():
    data = np.arange(10).reshape(5, 2)

    with pytest.raises(ValueError, match="without coordinates"):
        shape_common_data(data, {})


def test_probit():
    x = probit(np.random.normal(scale=10000, size=100)).eval()
    assert (x > 0).all() and (x < 1).all()


def test_cloglog():
    x = cloglog(np.random.normal(scale=10000, size=100)).eval()
    assert (x > 0).all() and (x < 1).all()


def test_loglogistic_distribution():
    mu, alpha = 0.7, 1.3
    value = np.array([0.1, 1.0, 10.0])
    dist = LogLogistic.dist(mu, alpha)

    actual_logp = pm.logp(dist, value).eval()
    actual_logcdf = pm.logcdf(dist, value).eval()
    reference = stats.fisk(c=1 / alpha, scale=np.exp(mu))

    np.testing.assert_allclose(actual_logp, reference.logpdf(value))
    np.testing.assert_allclose(actual_logcdf, reference.logcdf(value))
    assert np.isneginf(pm.logp(dist, 0).eval())
    assert np.isneginf(pm.logcdf(dist, 0).eval())

    draws = pm.draw(LogLogistic.dist(np.array([mu, mu]), alpha), draws=10, random_seed=42)
    assert draws.shape == (10, 2)
    assert (draws > 0).all()


@pytest.mark.parametrize("cure", [0.0, 0.3, 1.0])
def test_cure_weibull_distribution(cure):
    alpha, beta = 1.7, 2.4
    time = np.array([0.1, 1.0, 10.0])
    dist = CureWeibull.dist(alpha, beta, cure=cure)
    reference = stats.weibull_min(c=alpha, scale=beta)
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(
            pm.logp(dist, time).eval(), np.log1p(-cure) + reference.logpdf(time)
        )
        np.testing.assert_allclose(
            pm.logcdf(dist, time).eval(), np.log1p(-cure) + reference.logcdf(time)
        )
        np.testing.assert_allclose(pm.logp(dist, np.inf).eval(), np.log(cure))
    assert np.isneginf(pm.logp(dist, -1).eval())
    assert np.isneginf(pm.logcdf(dist, -1).eval())
    assert np.isneginf(pm.logcdf(dist, 0).eval())
    assert pm.logcdf(dist, np.inf).eval() == 0


@pytest.mark.parametrize("cure", [0.0, 0.3, 1.0])
def test_cure_weibull_random(cure):
    draws = pm.draw(CureWeibull.dist(1.7, 2.4, cure=cure, size=10000), random_seed=42)
    assert (draws > 0).all()
    assert np.isinf(draws).mean() == pytest.approx(cure, abs=0.02)
    if cure < 1:
        finite = draws[np.isfinite(draws)]
        assert finite.mean() == pytest.approx(stats.weibull_min(c=1.7, scale=2.4).mean(), rel=0.03)
    vector = pm.draw(CureWeibull.dist([1.0, 2.0], [2.0, 3.0], cure=[0.0, 1.0]), random_seed=42)
    assert vector.shape == (2,)
    assert np.isfinite(vector[0])
    assert np.isinf(vector[1])


def test_cure_weibull_random_broadcasts_all_parameters():
    dist = CureWeibull.dist(1.0, [1.0, 2.0], cure=0.0)
    draws = pm.draw(dist, draws=1000, random_seed=42)
    assert draws.shape == (1000, 2)
    assert abs(np.corrcoef(draws.T)[0, 1]) < 0.1
    dist = CureWeibull.dist(1.0, 2.0, cure=[0.0, 1.0])
    draws = pm.draw(dist, draws=10, random_seed=42)
    assert draws.shape == (10, 2)
    assert np.isfinite(draws[:, 0]).all()
    assert np.isinf(draws[:, 1]).all()


def test_cure_weibull_truncated_normalization():
    alpha, beta, cure = 1.7, 2.4, 0.3
    lower = np.array([1.0, -np.inf])
    upper = np.array([np.inf, 2.0])
    time = np.array([1.5, 1.5])
    dist = pm.Truncated.dist(CureWeibull.dist(alpha, beta, cure=cure), lower=lower, upper=upper)
    reference = stats.weibull_min(c=alpha, scale=beta)
    norm = [cure + (1 - cure) * reference.sf(1.0), (1 - cure) * reference.cdf(2.0)]
    expected = np.log1p(-cure) + reference.logpdf(time) - np.log(norm)
    np.testing.assert_allclose(pm.logp(dist, time).eval(), expected)


@pytest.mark.filterwarnings("error:Numba will use object mode:UserWarning")
def test_cure_weibull_numba_random_without_object_mode():
    dist = CureWeibull.dist(1.7, [1.0, 2.0], cure=[0.0, 1.0])
    draws = pm.draw(dist, draws=10, random_seed=42, mode="NUMBA")
    assert draws.shape == (10, 2)
    assert np.isfinite(draws[:, 0]).all()
    assert np.isinf(draws[:, 1]).all()
    assert np.unique(draws[:, 0]).size == 10


@pytest.mark.parametrize(
    "distribution, parameters, reference",
    [
        (pm.Weibull, {"alpha": 1.7, "beta": 2.4}, stats.weibull_min(c=1.7, scale=2.4)),
        (pm.Exponential, {"lam": 0.8}, stats.expon(scale=1 / 0.8)),
        (pm.LogNormal, {"mu": 0.5, "sigma": 0.7}, stats.lognorm(s=0.7, scale=np.exp(0.5))),
        (LogLogistic, {"mu": 0.5, "alpha": 0.7}, stats.fisk(c=1 / 0.7, scale=np.exp(0.5))),
    ],
)
def test_cure_distribution(distribution, parameters, reference):
    cured_distribution = make_cure_distribution(distribution)
    dist = cured_distribution.dist(**parameters, cure=0.3)
    time = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(pm.logp(dist, time).eval(), np.log(0.7) + reference.logpdf(time))
    np.testing.assert_allclose(pm.logcdf(dist, time).eval(), np.log(0.7) + reference.logcdf(time))
    assert pm.logp(dist, np.inf).eval() == pytest.approx(np.log(0.3))
    assert pm.logcdf(dist, np.inf).eval() == 0
    assert np.isneginf(pm.logp(dist, -np.inf).eval())
    assert np.isneginf(pm.logcdf(dist, -np.inf).eval())
    assert np.isfinite(support_point(dist).eval())

    with pm.Model() as model:
        cured_distribution("time", **parameters, cure=0.3, observed=time)
    assert np.isfinite(model.compile_logp()(model.initial_point()))

    draws = pm.draw(dist, draws=10000, random_seed=42)
    assert np.isinf(draws).mean() == pytest.approx(0.3, abs=0.02)
    np.testing.assert_allclose(
        np.quantile(draws[np.isfinite(draws)], [0.25, 0.5, 0.75]),
        reference.ppf([0.25, 0.5, 0.75]),
        rtol=0.05,
    )


def test_cure_distribution_resize():
    cure = pt.vector("cure")
    beta = pt.vector("beta")
    dist = CureWeibull.dist(alpha=1.7, beta=beta, cure=cure)
    updated_cure = pt.vector("updated_cure")
    updated_beta = pt.vector("updated_beta")
    cloned = clone_replace(dist, {cure: updated_cure, beta: updated_beta})
    resized = change_dist_size(cloned, new_size=(3,))
    draws = pm.draw(
        resized,
        draws=10,
        random_seed=42,
        givens={updated_cure: np.array([0.0, 1.0, 0.0]), updated_beta: np.ones(3)},
    )
    assert draws.shape == (10, 3)
    assert np.isinf(draws[:, 1]).all()
    assert np.isfinite(draws[:, [0, 2]]).all()

    expanded = change_dist_size(CureWeibull.dist(1.7, 2.4, cure=[0.0, 1.0]), (4,), expand=True)
    assert pm.draw(expanded, random_seed=42).shape == (4, 2)


def test_cure_weibull_gradients_at_infinity():
    alpha, beta, cure = pt.scalars("alpha", "beta", "cure")
    dist = CureWeibull.dist(alpha=alpha, beta=beta, cure=cure)
    gradients = pt.grad(pm.logp(dist, np.inf), [alpha, beta, cure])
    evaluate = pytensor.function([alpha, beta, cure], gradients)
    np.testing.assert_allclose(evaluate(1.7, 2.4, 0.3), [0, 0, 1 / 0.3])

    gradients = pt.grad(pm.logcdf(dist, np.inf), [alpha, beta, cure])
    evaluate = pytensor.function([alpha, beta, cure], gradients)
    np.testing.assert_allclose(evaluate(1.7, 2.4, 0.3), [0, 0, 0])


def test_censored():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "status": ["none", "right", "none", "left", "none"],
        }
    )

    df_bad = pd.DataFrame({"x": [1, 2], "status": ["foo", "bar"]})

    x = censored(df["x"], df["status"])
    assert x.shape == (5, 2)
    assert (x[:, -1] == np.array([0, 1, 0, -1, 0])).all()

    # Statuses are not the expected
    with pytest.raises(AssertionError, match="Statuses must be in"):
        censored(df_bad["x"], df_bad["status"])

    # Bad number of arguments
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        censored(df["x"])

    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        censored(df["x"], df["x"], df["status"])

    # Bad length
    with pytest.raises(AssertionError):
        censored(df["x"], df_bad["status"])

    # Interval censoring is not supported
    with pytest.raises(AssertionError, match="Statuses must be in"):
        censored(df["x"], ["none", "right", "interval", "left", "none"])


def test_competing_risks():
    transform = CompetingRisks()
    result = transform(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array(["right", "event", "event", "right"]),
        np.array(["none", "cause_b", "cause_a", "none"]),
    )

    assert result.shape == (4, 3)
    assert np.array_equal(result[:, 1], [1, 0, 0, 1])
    assert np.array_equal(result[:, 2], [0, 2, 1, 0])
    # The stateful transform preserves training codes when a cause is absent in new data.
    result = transform(np.array([6.0]), np.array(["event"]), np.array(["cause_b"]))
    assert np.array_equal(result[:, 2], [2])

    with pytest.raises(ValueError, match="Unknown competing-risks cause"):
        transform(np.array([7.0]), np.array(["event"]), np.array(["cause_c"]))

    with pytest.raises(ValueError, match="Left censoring is not supported"):
        CompetingRisks()(np.array([1.0]), np.array(["left"]), np.array(["cause_a"]))

    with pytest.raises(ValueError, match="must contain only"):
        CompetingRisks()(np.array([1.0]), np.array(["interval"]), np.array(["none"]))

    with pytest.raises(ValueError, match="must not be 'none' when status is 'event'"):
        CompetingRisks()(np.array([1.0]), np.array(["event"]), np.array(["none"]))

    with pytest.raises(ValueError, match="must be 'none' when status is 'right'"):
        CompetingRisks()(np.array([1.0]), np.array(["right"]), np.array(["cause_a"]))

    with pytest.raises(ValueError, match="cannot contain missing values"):
        CompetingRisks()(np.array([1.0]), np.array(["right"]), np.array([np.nan]))

    with pytest.raises(ValueError, match="requires at least one observed cause"):
        CompetingRisks()(np.array([1.0]), np.array(["right"]), np.array(["none"]))


@pytest.mark.parametrize(
    ("distribution", "reference_distribution", "parameter_names"),
    [
        (pm.Exponential, pz.Exponential, ("lam",)),
        (pm.Weibull, pz.Weibull, ("alpha", "beta")),
    ],
)
def test_competing_risks_distribution(distribution, reference_distribution, parameter_names):
    dist = make_competing_risks_distribution(distribution)
    value = np.array([1.0, 2.0, 3.0])
    status = np.array([0, 1, 0])
    cause = np.array([1, 0, 2])
    parameter_grid = 1 + np.arange(value.size * 2).reshape(value.size, 2)

    with pm.Model() as model:
        parameters = {
            name: pm.Normal(name, mu=parameter_grid + index / 2, sigma=0.1)
            for index, name in enumerate(parameter_names)
        }
        status_data = pm.Data("status", status)
        cause_data = pm.Data("cause", cause)
        dist("y", status=status_data, cause=cause_data, observed=value, **parameters)

    point = model.initial_point()
    logp = model.compile_logp(vars=[model["y"]])(point)
    parameters = {name: point[name] for name in parameter_names}

    n_causes = next(iter(parameters.values())).shape[-1]
    reference_dists = [
        [
            reference_distribution(
                **{name: values[row, cause] for name, values in parameters.items()}
            )
            for cause in range(n_causes)
        ]
        for row in range(value.size)
    ]

    log_density = np.array(
        [
            [reference_dist.logpdf(time) for reference_dist in row]
            for time, row in zip(value, reference_dists)
        ]
    )
    log_survival = np.array(
        [
            [reference_dist.logsf(time) for reference_dist in row]
            for time, row in zip(value, reference_dists)
        ]
    )

    cause_index = np.maximum(cause - 1, 0)
    rows = np.arange(value.size)
    total_log_survival = log_survival.sum(axis=-1)
    event_logp = (
        log_density[rows, cause_index] + total_log_survival - log_survival[rows, cause_index]
    )
    expected = np.where(status == 0, event_logp, total_log_survival).sum()
    assert np.isclose(logp, expected)


def test_counts():
    y1 = np.array([1, 2, 3])
    y2 = np.array([3, 4, 3])
    totals = np.array([4, 6, 6])

    result = counts(y1, y2)
    assert np.array_equal(result, np.column_stack([y1, y2]))

    assert np.array_equal(counts(y1, y2, n=totals), result)
    assert np.array_equal(
        counts(np.array([1, 2]), np.array([3, 2]), n=4), np.array([[1, 3], [2, 2]])
    )

    with pytest.raises(ValueError, match="must sum to 'n'"):
        counts(y1, y2, n=5)

    with pytest.raises(ValueError, match="length of 'n'"):
        counts(y1, y2, n=np.array([4, 6]))


def test_truncated():
    x = np.array([-3, -2, -1, 0, 0, 0, 1, 1, 2, 3])
    lower = -5
    upper = 4.5
    lower_arr = np.array([-5] * 6 + [-4] * 4)
    upper_arr = np.array([5] * 6 + [5.35] * 4)

    # Arguments and expected outcomes
    iterable = {
        "lower": (lower, None, lower, lower_arr, None, lower_arr),
        "upper": (None, upper, upper, None, upper_arr, upper_arr),
        "elower": (lower, -np.inf, lower, lower_arr, -np.inf, lower_arr),
        "eupper": (np.inf, upper, upper, np.inf, upper_arr, upper_arr),
    }

    for l, u, el, eu in zip(*iterable.values()):
        result = truncated(x, lb=l, ub=u)
        assert result.shape == (10, 3)
        assert (result[:, 0] == x).all()
        assert (result[:, 1] == el).all()
        assert (result[:, 2] == eu).all()

    with pytest.raises(ValueError, match="'lb' and 'ub' cannot both be None"):
        truncated(x)

    with pytest.raises(ValueError, match="'truncated' only works with 1-dimensional arrays"):
        truncated(np.column_stack([x, x]))

    with pytest.raises(AssertionError, match="The length of 'lb' must be equal to the one of 'x'"):
        truncated(x, np.array([-5, -6]))

    with pytest.raises(AssertionError, match="The length of 'ub' must be equal to the one of 'x'"):
        truncated(x, ub=np.array([5, 6]))

    with pytest.raises(ValueError, match="'lb' must be 0 or 1 dimensional."):
        truncated(x, np.column_stack([lower_arr, lower_arr]))

    with pytest.raises(ValueError, match="'ub' must be 0 or 1 dimensional."):
        truncated(x, ub=np.column_stack([upper_arr, upper_arr]))


def test_constrained():
    x = np.array([-3, -2, -1, 0, 0, 0, 1, 1, 2, 3])
    lower = -5
    upper = 4.5

    # Arguments and expected outcomes
    iterable = {
        "lower": (lower, None, lower),
        "upper": (None, upper, upper),
        "elower": (lower, -np.inf, lower),
        "eupper": (np.inf, upper, upper),
    }

    for l, u, el, eu in zip(*iterable.values()):
        result = constrained(x, lb=l, ub=u)
        assert result.shape == (10, 3)
        assert (result[:, 0] == x).all()
        assert (result[:, 1] == el).all()
        assert (result[:, 2] == eu).all()

    with pytest.raises(ValueError, match="'lb' must be None or scalar."):
        constrained(x, np.array([lower, lower]))

    with pytest.raises(ValueError, match="'ub' must be None or scalar."):
        constrained(x, ub=np.array([upper, upper]))


def test_weighted():
    rng = np.random.default_rng(1234)
    weights = 1 + rng.poisson(lam=3, size=100)
    weights_wrong = rng.normal(size=100)
    y = rng.exponential(scale=3, size=100)

    out = weighted(y, weights)
    assert out.shape == (100, 2)
    assert (out[:, 0] == y).all()
    assert (out[:, 1] == weights).all()

    with pytest.raises(ValueError, match="Weights must be positive"):
        weighted(y, weights_wrong)

    # Draw function works and matches the non-weighted version
    WeightedNormal = make_weighted_distribution(pm.Normal)
    draws1 = pm.draw(WeightedNormal.dist(mu=0, sigma=1), draws=10, random_seed=1234)
    draws2 = pm.draw(pm.Normal.dist(mu=0, sigma=1), draws=10, random_seed=1234)
    assert np.allclose(draws1, draws2)

    WeightedExponential = make_weighted_distribution(pm.Exponential)
    draws1 = pm.draw(WeightedExponential.dist(lam=2.0), draws=10, random_seed=11)
    draws2 = pm.draw(pm.Exponential.dist(lam=2.0), draws=10, random_seed=11)
    assert np.allclose(draws1, draws2)

    # Logp works and is propertly weighted
    weights = np.array([0.5, 1.0, 3.2, 4.5, 1.0])
    values = np.array([-2, -1, 0, 1.0, 2.0])
    logp1 = pm.logp(WeightedNormal.dist(mu=0.5, sigma=0.3, weights=weights), value=values).eval()
    logp2 = pm.logp(pm.Normal.dist(mu=0.5, sigma=0.3), value=values).eval()
    assert np.allclose(logp1 / logp2, weights)

    weights = np.array([1, 2.5, 2.5])
    values = np.array([1, 1, 4.0])
    logp1 = pm.logp(WeightedExponential.dist(lam=2, weights=weights), value=values).eval()
    logp2 = pm.logp(pm.Exponential.dist(2), value=values).eval()

    assert np.allclose(logp1 / logp2, weights)
