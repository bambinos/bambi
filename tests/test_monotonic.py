"""Tests for monotonic effects ``mo()`` (ported from brms)."""

import numpy as np
import pandas as pd
import pytest

import bambi as bmb


LEVELS = ["below_20", "20_to_40", "40_to_100", "greater_100"]
MEANS = {"below_20": 30.0, "20_to_40": 60.0, "40_to_100": 70.0, "greater_100": 75.0}


@pytest.fixture(scope="module")
def data_ordered():
    rng = np.random.default_rng(2026)
    income = pd.Categorical(rng.choice(LEVELS, 300), categories=LEVELS, ordered=True)
    y = np.array([MEANS[i] for i in income]) + rng.normal(0, 7, 300)
    return pd.DataFrame({"y": y, "income": income})


@pytest.fixture(scope="module")
def data_integer():
    rng = np.random.default_rng(2026)
    dose = rng.integers(1, 6, 200)  # 1..5 inclusive
    y = 2.0 * (dose - 1) + rng.normal(0, 1, 200)
    return pd.DataFrame({"y": y, "dose": dose})


def test_build_with_ordered_factor(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.build()
    assert "mo(income)" in model.components["mu"].monotonic_terms
    term = model.components["mu"].monotonic_terms["mo(income)"]
    assert term.K == 4
    assert term.D == 3
    assert term.kind == "ordered"
    assert list(term.levels) == LEVELS


def test_build_with_integer(data_integer):
    model = bmb.Model("y ~ mo(dose)", data_integer)
    model.build()
    term = model.components["mu"].monotonic_terms["mo(dose)"]
    assert term.K == 5
    assert term.D == 4
    assert term.kind == "integer"


def test_default_simplex_prior_is_uniform_dirichlet(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.build()
    term = model.components["mu"].monotonic_terms["mo(income)"]
    simplex_prior = term.prior["simplex"]
    assert simplex_prior.name == "Dirichlet"
    np.testing.assert_array_equal(simplex_prior.args["a"], np.ones(3))


def test_default_slope_is_autoscaled(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.build()
    term = model.components["mu"].monotonic_terms["mo(income)"]
    slope_prior = term.prior["slope"]
    assert slope_prior.name == "Normal"
    # Auto-scaled to STD * response_std / D
    response_std = float(np.std(data_ordered["y"].to_numpy()))
    expected_sigma = 2.5 * response_std / 3
    assert slope_prior.args["sigma"] == pytest.approx(expected_sigma, rel=1e-5)


def test_custom_simplex_prior_overrides_default(data_ordered):
    custom = bmb.Prior("Dirichlet", a=np.array([2.0, 1.0, 1.0]))
    model = bmb.Model(
        "y ~ mo(income)",
        data_ordered,
        priors={"mo(income)": {"simplex": custom}},
    )
    model.build()
    term = model.components["mu"].monotonic_terms["mo(income)"]
    np.testing.assert_array_equal(term.prior["simplex"].args["a"], [2.0, 1.0, 1.0])
    # Slope default is preserved when user only overrides the simplex
    assert term.prior["slope"].name == "Normal"


def test_custom_slope_prior_overrides_default(data_ordered):
    custom = bmb.Prior("Normal", mu=0.0, sigma=50.0)
    model = bmb.Model(
        "y ~ mo(income)",
        data_ordered,
        priors={"mo(income)": {"slope": custom}},
    )
    model.build()
    term = model.components["mu"].monotonic_terms["mo(income)"]
    # User-supplied slope is not auto-rescaled
    assert term.prior["slope"].args["sigma"] == pytest.approx(50.0)


def test_rejects_unordered_categorical():
    df = pd.DataFrame(
        {
            "y": np.arange(20, dtype=float),
            "x": pd.Categorical(["a", "b"] * 10, ordered=False),
        }
    )
    with pytest.raises(ValueError, match="ordered categorical"):
        bmb.Model("y ~ mo(x)", df)


def test_rejects_continuous():
    df = pd.DataFrame({"y": np.arange(20, dtype=float), "x": np.linspace(0, 1, 20)})
    with pytest.raises(ValueError, match="integer or ordered categorical"):
        bmb.Model("y ~ mo(x)", df)


def test_rejects_single_level():
    df = pd.DataFrame({"y": np.arange(10, dtype=float), "x": [1] * 10})
    with pytest.raises(ValueError, match="at least 2 distinct values"):
        bmb.Model("y ~ mo(x)", df)


def test_fit_recovers_category_means(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    idata = model.fit(tune=600, draws=600, chains=2, random_seed=20260522, progressbar=False)

    posterior = idata.posterior
    assert "mo(income)_simplex" in posterior.data_vars
    assert "mo(income)_b" in posterior.data_vars

    simplex = posterior["mo(income)_simplex"]
    assert simplex.dims == ("chain", "draw", "mo(income)_simplex_dim")
    assert simplex.shape[-1] == 3
    # Simplex sums to 1 within numerical tolerance
    s = simplex.sum("mo(income)_simplex_dim").to_numpy()
    np.testing.assert_allclose(s, 1.0, atol=1e-6)

    # Reconstruct category means and check we recover the truth
    intercept = posterior["Intercept"].mean().item()
    slope = posterior["mo(income)_b"].mean().item()
    s_mean = simplex.mean(("chain", "draw")).to_numpy()
    cumsum = np.concatenate([[0.0], np.cumsum(s_mean)])
    fitted = intercept + slope * 3 * cumsum
    truth = np.array([MEANS[l] for l in LEVELS])
    # 3-sigma noise band on means of ~75 obs/category is generous
    np.testing.assert_allclose(fitted, truth, atol=4.0)


def test_predict_in_sample_and_new_data(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    idata = model.fit(tune=400, draws=400, chains=2, random_seed=42, progressbar=False)

    # In-sample
    idata_in = model.predict(idata, kind="response_params", inplace=False)
    mu = idata_in.posterior["mu"].mean(("chain", "draw")).to_numpy()
    assert mu.shape == (len(data_ordered),)

    # Out-of-sample: one row per level
    new_df = pd.DataFrame(
        {"income": pd.Categorical(LEVELS, categories=LEVELS, ordered=True)}
    )
    idata_new = model.predict(idata, kind="response_params", data=new_df, inplace=False)
    mu_new = idata_new.posterior["mu"].mean(("chain", "draw")).to_numpy()
    assert mu_new.shape == (4,)
    # Monotonic increasing
    assert np.all(np.diff(mu_new) > 0)
    # Roughly matches truth
    truth = np.array([MEANS[l] for l in LEVELS])
    np.testing.assert_allclose(mu_new, truth, atol=4.0)


def test_predict_rejects_unseen_category(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    idata = model.fit(tune=300, draws=300, chains=2, random_seed=42, progressbar=False)
    bad_levels = ["unseen"]
    bad_df = pd.DataFrame(
        {"income": pd.Categorical(bad_levels * 3, categories=bad_levels, ordered=True)}
    )
    with pytest.raises(ValueError, match="unseen categories"):
        model.predict(idata, kind="response_params", data=bad_df, inplace=False)


def test_predict_posterior_predictive(data_ordered):
    model = bmb.Model("y ~ mo(income)", data_ordered)
    idata = model.fit(tune=300, draws=300, chains=2, random_seed=42, progressbar=False)
    idata = model.predict(idata, kind="response", inplace=False)
    assert "y" in idata.posterior_predictive.data_vars
    assert idata.posterior_predictive["y"].shape[-1] == len(data_ordered)


def test_mo_combined_with_common_term():
    rng = np.random.default_rng(2026)
    income = pd.Categorical(rng.choice(LEVELS, 300), categories=LEVELS, ordered=True)
    x = rng.normal(size=300)
    y = (
        np.array([MEANS[i] for i in income])
        + 4.0 * x
        + rng.normal(0, 3, 300)
    )
    df = pd.DataFrame({"y": y, "income": income, "x": x})

    model = bmb.Model("y ~ x + mo(income)", df)
    idata = model.fit(tune=500, draws=500, chains=2, random_seed=42, progressbar=False)

    posterior = idata.posterior
    # The continuous slope on x is recovered
    assert posterior["x"].mean().item() == pytest.approx(4.0, abs=0.5)
    # And the monotonic structure is still there
    assert "mo(income)_simplex" in posterior.data_vars


def test_distributional_use_in_auxiliary_dpar(data_ordered):
    formula = bmb.Formula("y ~ 1", "sigma ~ mo(income)")
    model = bmb.Model(formula, data_ordered, family="gaussian")
    model.build()
    sigma_component = model.components["sigma"]
    assert "mo(income)" in sigma_component.monotonic_terms
    term = sigma_component.monotonic_terms["mo(income)"]
    assert term.prefix == "sigma"
    assert term.name == "sigma_mo(income)"
