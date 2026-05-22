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
    assert "mo(income)_slope" in posterior.data_vars

    simplex = posterior["mo(income)_simplex"]
    assert simplex.dims == ("chain", "draw", "mo(income)_simplex_dim")
    assert simplex.shape[-1] == 3
    # Simplex sums to 1 within numerical tolerance
    s = simplex.sum("mo(income)_simplex_dim").to_numpy()
    np.testing.assert_allclose(s, 1.0, atol=1e-6)

    # Reconstruct category means and check we recover the truth
    intercept = posterior["Intercept"].mean().item()
    slope = posterior["mo(income)_slope"].mean().item()
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


def test_model_repr_shows_monotonic_section(data_ordered):
    """``print(model)`` must include a 'Monotonic effects' section with the slope
    and simplex priors."""
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.build()
    text = str(model)
    assert "Monotonic effects" in text
    assert "mo(income)" in text
    assert "simplex ~ Dirichlet" in text
    assert "slope ~ Normal" in text


def test_distributional_use_in_auxiliary_dpar(data_ordered):
    formula = bmb.Formula("y ~ 1", "sigma ~ mo(income)")
    model = bmb.Model(formula, data_ordered, family="gaussian")
    model.build()
    sigma_component = model.components["sigma"]
    assert "mo(income)" in sigma_component.monotonic_terms
    term = sigma_component.monotonic_terms["mo(income)"]
    assert term.prefix == "sigma"
    assert term.name == "sigma_mo(income)"


# ---------------------------------------------------------------------------
# id= shared-simplex tests (conditional monotonicity)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_two_ordered():
    """Two ordered predictors that share the same monotonic 'shape' so a shared
    simplex should fit well."""
    rng = np.random.default_rng(2026)
    n = 500
    income1 = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS, ordered=True)
    income2 = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS, ordered=True)
    shape = np.array([0.0, 30.0, 40.0, 45.0])  # steps 30, 10, 5
    mean = shape[np.asarray(income1.codes)] + shape[np.asarray(income2.codes)]
    y = mean + rng.normal(0, 5, n)
    return pd.DataFrame({"y": y, "income1": income1, "income2": income2})


def test_shared_id_creates_single_simplex(data_two_ordered):
    model = bmb.Model(
        "y ~ mo(income1, id='shape') + mo(income2, id='shape')", data_two_ordered
    )
    model.build()
    named = list(model.backend.model.named_vars)
    # Exactly one shared Dirichlet
    assert "simplex_shape" in named
    # Per-term simplices should NOT exist when id is shared
    assert "mo(income1, id='shape')_simplex" not in named
    assert "mo(income2, id='shape')_simplex" not in named
    # But each term has its own slope
    assert "mo(income1, id='shape')_slope" in named
    assert "mo(income2, id='shape')_slope" in named


def test_shared_id_inconsistent_K_raises():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "a": pd.Categorical(
                rng.choice(["x", "y", "z"], n),
                categories=["x", "y", "z"],
                ordered=True,
            ),
            "b": pd.Categorical(
                rng.choice(LEVELS, n), categories=LEVELS, ordered=True
            ),
        }
    )
    with pytest.raises(ValueError, match="inconsistent K"):
        bmb.Model("y ~ mo(a, id='same') + mo(b, id='same')", df)


def test_shared_id_unified_custom_prior(data_two_ordered):
    custom = bmb.Prior("Dirichlet", a=np.array([1.0, 2.0, 1.0]))
    model = bmb.Model(
        "y ~ mo(income1, id='shape') + mo(income2, id='shape')",
        data_two_ordered,
        priors={"mo(income1, id='shape')": {"simplex": custom}},
    )
    model.build()
    # Both terms now reference the same Dirichlet args
    t1 = model.components["mu"].monotonic_terms["mo(income1, id='shape')"]
    t2 = model.components["mu"].monotonic_terms["mo(income2, id='shape')"]
    np.testing.assert_array_equal(t1.prior["simplex"].args["a"], [1, 2, 1])
    np.testing.assert_array_equal(t2.prior["simplex"].args["a"], [1, 2, 1])


def test_shared_id_conflicting_priors_raise(data_two_ordered):
    p1 = bmb.Prior("Dirichlet", a=np.array([1.0, 2.0, 1.0]))
    p2 = bmb.Prior("Dirichlet", a=np.array([1.0, 1.0, 5.0]))
    with pytest.raises(ValueError, match="conflicting simplex priors"):
        bmb.Model(
            "y ~ mo(income1, id='shape') + mo(income2, id='shape')",
            data_two_ordered,
            priors={
                "mo(income1, id='shape')": {"simplex": p1},
                "mo(income2, id='shape')": {"simplex": p2},
            },
        )


def test_shared_id_recovers_truth(data_two_ordered):
    model = bmb.Model(
        "y ~ mo(income1, id='shape') + mo(income2, id='shape')", data_two_ordered
    )
    idata = model.fit(
        tune=600, draws=600, chains=2, random_seed=42, progressbar=False
    )
    post = idata.posterior
    assert "simplex_shape" in post.data_vars
    # Posterior mean of simplex should be near [30,10,5]/45
    s = post["simplex_shape"].mean(("chain", "draw")).to_numpy()
    np.testing.assert_allclose(s, [30.0 / 45, 10.0 / 45, 5.0 / 45], atol=0.05)
    # Both slopes should land near 15
    for term_name in (
        "mo(income1, id='shape')_slope",
        "mo(income2, id='shape')_slope",
    ):
        assert post[term_name].mean().item() == pytest.approx(15.0, abs=1.5)


def test_id_kwarg_validation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=50),
            "x": pd.Categorical(
                rng.choice(LEVELS, 50), categories=LEVELS, ordered=True
            ),
        }
    )
    with pytest.raises(ValueError, match="'id'.*must be a string"):
        bmb.Model("y ~ mo(x, id=1)", df)


def test_set_alias_renames_simplex_and_slope(data_ordered):
    """``set_alias`` must propagate to the simplex and slope variable names."""
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.set_alias({"mo(income)": "income_effect"})
    model.build()
    named = list(model.backend.model.named_vars)
    # Aliased names appear
    assert "income_effect_simplex" in named
    assert "income_effect_slope" in named
    assert "income_effect" in named  # the Deterministic for the contribution
    # The original (un-aliased) names should NOT be present
    assert "mo(income)_simplex" not in named
    assert "mo(income)_slope" not in named
    # And the coord dim is aliased too
    assert "income_effect_simplex_dim" in model.backend.model.coords


def test_set_alias_fit_then_predict(data_ordered):
    """End-to-end: aliasing survives ``fit`` + ``predict`` on new data."""
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.set_alias({"mo(income)": "income_effect"})
    idata = model.fit(
        tune=400, draws=400, chains=2, random_seed=42, progressbar=False
    )
    post = idata.posterior
    assert "income_effect_simplex" in post.data_vars
    assert "income_effect_slope" in post.data_vars

    new_df = pd.DataFrame(
        {"income": pd.Categorical(LEVELS, categories=LEVELS, ordered=True)}
    )
    idata_new = model.predict(
        idata, kind="response_params", data=new_df, inplace=False
    )
    mu_new = idata_new.posterior["mu"].mean(("chain", "draw")).to_numpy()
    assert mu_new.shape == (4,)
    assert np.all(np.diff(mu_new) > 0)  # monotonic increasing


def test_set_alias_on_interaction(data_ordered):
    """``set_alias`` on a monotonic interaction term must rename slope, simplex,
    and the contribution Deterministic."""
    rng = np.random.default_rng(0)
    df = data_ordered.assign(x=rng.normal(size=len(data_ordered)))
    model = bmb.Model("y ~ mo(income) * x", df)
    model.set_alias({"mo(income)": "inc", "mo(income):x": "inc_x"})
    model.build()
    named = list(model.backend.model.named_vars)
    assert "inc_slope" in named
    assert "inc_simplex" in named
    assert "inc_x_slope" in named
    assert "inc_x_simplex_0" in named  # independent simplex for the interaction
    assert "inc" in named  # main-effect deterministic
    assert "inc_x" in named  # interaction deterministic
    # Old names are gone
    assert "mo(income)_slope" not in named
    assert "mo(income):x_slope" not in named
    assert "mo(income)_simplex" not in named


def test_set_alias_on_group_specific(data_ordered):
    """``set_alias`` on a (mo(x) | g) term must rename the simplex variable,
    the simplex coord, AND the factor coord."""
    rng = np.random.default_rng(0)
    df = data_ordered.assign(
        g=pd.Categorical(rng.choice(["g1", "g2", "g3"], len(data_ordered)))
    )
    model = bmb.Model("y ~ (mo(income) | g)", df)
    model.set_alias({"mo(income)|g": "income_by_group"})
    model.build()
    named = list(model.backend.model.named_vars)
    coords = list(model.backend.model.coords)
    # Aliased
    assert "income_by_group" in named  # r_g vector
    assert "income_by_group_sigma" in named
    assert "income_by_group_offset" in named
    assert "income_by_group_simplex" in named
    assert "income_by_group_simplex_dim" in coords
    assert "income_by_group__factor_dim" in coords
    # Un-aliased names should be gone
    assert "mo(income)|g_simplex" not in named
    assert "mo(income)|g_simplex_dim" not in coords


def test_unshared_id_is_independent_per_term(data_ordered):
    """No id= means each term gets its own simplex (existing behavior)."""
    model = bmb.Model("y ~ mo(income)", data_ordered)
    model.build()
    named = list(model.backend.model.named_vars)
    assert "mo(income)_simplex" in named
    # No accidental shared registry
    assert not any(v.startswith("simplex_") for v in named)


# ---------------------------------------------------------------------------
# Interaction tests: mo() * continuous, mo() * categorical, mo() : mo()
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_mo_x():
    """mo(income) * x with known coefficients."""
    rng = np.random.default_rng(2026)
    n = 600
    income = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS, ordered=True)
    x = rng.normal(size=n)
    shape = np.array([0.0, 30.0, 40.0, 45.0])
    codes = np.asarray(income.codes)
    mu = 5.0 + shape[codes] + 4.0 * x + 0.2 * shape[codes] * x
    y = mu + rng.normal(0, 3, n)
    return pd.DataFrame({"y": y, "income": income, "x": x})


@pytest.fixture(scope="module")
def data_mo_g():
    """mo(income, id='inc') + mo(income, id='inc'):g with known multipliers."""
    rng = np.random.default_rng(2026)
    n = 800
    income = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS, ordered=True)
    g = pd.Categorical(rng.choice(["g1", "g2", "g3"], n))
    shape = np.array([0.0, 30.0, 40.0, 45.0])
    codes = np.asarray(income.codes)
    g_mult = np.where(g == "g1", 1.0, np.where(g == "g2", 1.5, 0.5))
    mu = 5.0 + shape[codes] * g_mult
    y = mu + rng.normal(0, 3, n)
    return pd.DataFrame({"y": y, "income": income, "g": g})


def test_mo_continuous_interaction_build(data_mo_x):
    model = bmb.Model("y ~ mo(income) * x", data_mo_x)
    model.build()
    interaction_terms = model.components["mu"].monotonic_interaction_terms
    assert "mo(income):x" in interaction_terms
    named = list(model.backend.model.named_vars)
    # Slope for the interaction
    assert "mo(income):x_slope" in named
    # Independent simplices for the main and interaction (no id=)
    assert "mo(income)_simplex" in named
    assert "mo(income):x_simplex_0" in named


def test_mo_continuous_interaction_recovers_truth(data_mo_x):
    model = bmb.Model("y ~ mo(income) * x", data_mo_x)
    idata = model.fit(
        tune=600, draws=600, chains=2, random_seed=42, progressbar=False
    )
    post = idata.posterior
    assert post["Intercept"].mean().item() == pytest.approx(5.0, abs=1.0)
    assert post["x"].mean().item() == pytest.approx(4.0, abs=0.5)
    assert post["mo(income)_slope"].mean().item() == pytest.approx(15.0, abs=1.5)
    # Interaction slope is on the main-effect scale: 0.2 * 45 / D = 3
    assert post["mo(income):x_slope"].mean().item() == pytest.approx(3.0, abs=0.6)


def test_mo_categorical_interaction_with_shared_id(data_mo_g):
    formula = "y ~ mo(income, id='inc') + mo(income, id='inc'):g"
    model = bmb.Model(formula, data_mo_g)
    model.build()
    named = list(model.backend.model.named_vars)
    # ONE shared simplex used by both the main and the interaction
    assert "simplex_inc" in named
    assert "mo(income, id='inc'):g_simplex_0" not in named
    # Vector slope for the 2 dummies of g
    assert "mo(income, id='inc'):g_slope" in named


def test_mo_categorical_interaction_recovers_truth(data_mo_g):
    formula = "y ~ mo(income, id='inc') + mo(income, id='inc'):g"
    model = bmb.Model(formula, data_mo_g)
    idata = model.fit(
        tune=600, draws=600, chains=2, random_seed=42, progressbar=False
    )
    post = idata.posterior
    assert post["Intercept"].mean().item() == pytest.approx(5.0, abs=1.0)
    simplex = post["simplex_inc"].mean(("chain", "draw")).to_numpy()
    np.testing.assert_allclose(simplex, [30.0 / 45, 10.0 / 45, 5.0 / 45], atol=0.05)
    # Main slope is the g1 effect (= 15)
    assert post["mo(income, id='inc')_slope"].mean().item() == pytest.approx(15.0, abs=1.5)
    # Interaction has dims (g_slope_dim,) of length 2: g2 and g3 excess slopes
    interaction = post["mo(income, id='inc'):g_slope"].mean(("chain", "draw")).to_numpy()
    # The order matches the dummy columns (treatment coding: g2 dummy first, g3 second)
    # g2 excess: 0.5 * 15 = 7.5; g3 excess: -0.5 * 15 = -7.5
    np.testing.assert_allclose(interaction, [7.5, -7.5], atol=1.0)


# ---------------------------------------------------------------------------
# Group-specific monotonic tests: (mo(x) | g)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_mo_gs():
    """Five groups, each with a per-group monotonic slope; same simplex shape."""
    rng = np.random.default_rng(2026)
    n = 800
    g_levels = [f"g{i}" for i in range(1, 6)]
    group = pd.Categorical(rng.choice(g_levels, n), categories=g_levels)
    income = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS, ordered=True)
    codes = np.asarray(income.codes)
    shape = np.array([0.0, 30.0, 40.0, 45.0])
    slope_per_group = {"g1": 1.0, "g2": 1.3, "g3": 0.7, "g4": 1.5, "g5": 0.5}
    g_mult = np.array([slope_per_group[g] for g in group])
    mu = 5.0 + shape[codes] * g_mult
    y = mu + rng.normal(0, 3, n)
    return pd.DataFrame({"y": y, "income": income, "g": group})


def test_mo_group_specific_build(data_mo_gs):
    model = bmb.Model("y ~ mo(income) + (mo(income) | g)", data_mo_gs)
    model.build()
    gs_terms = model.components["mu"].monotonic_group_specific_terms
    assert "mo(income)|g" in gs_terms
    term = gs_terms["mo(income)|g"]
    assert term.K == 4
    assert term.D == 3
    assert list(term.groups) == ["g1", "g2", "g3", "g4", "g5"]

    named = list(model.backend.model.named_vars)
    assert "mo(income)|g" in named  # per-group slope vector
    assert "mo(income)|g_sigma" in named  # hyperprior
    assert "mo(income)|g_simplex" in named  # this term's own simplex
    assert "mo(income)_simplex" in named  # main effect simplex (independent)


def test_mo_group_specific_recovers_truth(data_mo_gs):
    model = bmb.Model("y ~ mo(income) + (mo(income) | g)", data_mo_gs)
    idata = model.fit(
        tune=800,
        draws=800,
        chains=2,
        random_seed=42,
        progressbar=False,
        target_accept=0.95,
    )
    post = idata.posterior
    main_slope = post["mo(income)_slope"].mean().item()
    r_g = post["mo(income)|g"].mean(("chain", "draw")).to_numpy()
    totals = main_slope + r_g
    truth = np.array([15.0, 19.5, 10.5, 22.5, 7.5])
    np.testing.assert_allclose(totals, truth, atol=2.0)


def test_mo_group_specific_only(data_mo_gs):
    """(mo(x) | g) without a main mo(x) effect."""
    model = bmb.Model("y ~ (mo(income) | g)", data_mo_gs)
    model.build()
    assert "mo(income)|g" in model.components["mu"].monotonic_group_specific_terms
    # Implicit (1|g) was added by formulae
    assert "1|g" in model.components["mu"].group_specific_terms


def test_mo_group_specific_predict_new_data(data_mo_gs):
    model = bmb.Model("y ~ (mo(income) | g)", data_mo_gs)
    idata = model.fit(
        tune=400,
        draws=400,
        chains=2,
        random_seed=42,
        progressbar=False,
        target_accept=0.95,
    )
    g_levels = ["g1", "g2", "g3", "g4", "g5"]
    new_df = pd.DataFrame(
        {
            "income": pd.Categorical(LEVELS * 5, categories=LEVELS, ordered=True),
            "g": pd.Categorical(np.repeat(g_levels, 4), categories=g_levels),
        }
    )
    idata_new = model.predict(
        idata, kind="response_params", data=new_df, inplace=False
    )
    mu_new = idata_new.posterior["mu"].mean(("chain", "draw")).to_numpy()
    assert mu_new.shape == (20,)
    # Monotonic within each group
    for i in range(5):
        block = mu_new[i * 4 : (i + 1) * 4]
        assert np.all(np.diff(block) >= 0), f"non-monotonic block at group {i}"


def test_mo_group_specific_unseen_group_raises(data_mo_gs):
    model = bmb.Model("y ~ (mo(income) | g)", data_mo_gs)
    idata = model.fit(
        tune=200, draws=200, chains=2, random_seed=42, progressbar=False
    )
    bad_df = pd.DataFrame(
        {
            "income": pd.Categorical(
                [LEVELS[0]] * 3, categories=LEVELS, ordered=True
            ),
            "g": pd.Categorical(["unseen_g"] * 3),
        }
    )
    with pytest.raises(ValueError, match="unseen groups"):
        model.predict(idata, kind="response_params", data=bad_df, inplace=False)


def test_mo_group_specific_shared_id_with_main(data_mo_gs):
    """When (mo(x, id='s') | g) and mo(x, id='s') share id, only ONE simplex is built."""
    model = bmb.Model(
        "y ~ mo(income, id='s') + (mo(income, id='s') | g)", data_mo_gs
    )
    model.build()
    named = list(model.backend.model.named_vars)
    # ONE shared simplex
    assert "simplex_s" in named
    # No per-term simplex
    assert "mo(income, id='s')_simplex" not in named
    assert "mo(income, id='s')|g_simplex" not in named


def test_mo_interaction_predict_new_data(data_mo_x):
    model = bmb.Model("y ~ mo(income) * x", data_mo_x)
    idata = model.fit(
        tune=400, draws=400, chains=2, random_seed=42, progressbar=False
    )
    new_df = pd.DataFrame(
        {
            "income": pd.Categorical(LEVELS, categories=LEVELS, ordered=True),
            "x": [0.0, 1.0, -1.0, 0.5],
        }
    )
    idata_new = model.predict(
        idata, kind="response_params", data=new_df, inplace=False
    )
    mu = idata_new.posterior["mu"].mean(("chain", "draw")).to_numpy()
    assert mu.shape == (4,)
    # Sanity: at x=0 with income=below_20 (code=0), mu = intercept ~ 5
    assert abs(mu[0] - 5.0) < 2.0
