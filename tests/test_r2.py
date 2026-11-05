import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from arviz_stats import bayesian_r2, residual_r2

import bambi as bmb
from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import get_distribution_from_likelihood
from bambi.defaults import get_builtin_family


N_OBS = 20


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1061)
    x = rng.normal(size=N_OBS)
    return pd.DataFrame(
        {
            "x": x,
            "y": 1 + 2 * x + rng.normal(size=N_OBS),
            "count": rng.poisson(np.exp(0.5 + 0.5 * x)),
            "binary": rng.binomial(1, 1 / (1 + np.exp(-1.5 * x))),
            "trials": rng.integers(5, 15, size=N_OBS),
            "category": rng.choice(list("abc"), size=N_OBS),
            "status": rng.choice(["none", "right"], size=N_OBS),
        }
    ).assign(successes=lambda df: rng.binomial(df["trials"], 0.4))


def make_idata(posterior_vars, observed=None):
    """Build an InferenceData-like DataTree from fixed draws.

    Values with 3 dimensions are (chain, draw, __obs__) and with 2 dimensions (chain, draw).
    """
    data_vars = {}
    for name, values in posterior_vars.items():
        dims = ("chain", "draw", "__obs__")[: np.ndim(values)]
        data_vars[name] = (dims, np.asarray(values, dtype=float))
    groups = {"posterior": xr.Dataset(data_vars)}
    if observed is not None:
        groups["observed_data"] = xr.Dataset(
            {name: ("__obs__", np.asarray(values)) for name, values in observed.items()}
        )
    return xr.DataTree.from_dict(groups)


def manual_bayesian_r2(mean, variance):
    """R^2 for each draw from arrays of shape (chain, draw, obs)"""
    variance = np.broadcast_to(variance, mean.shape).reshape(-1, mean.shape[-1])
    mean = mean.reshape(-1, mean.shape[-1])
    var_mean = np.var(mean, axis=1, ddof=1)
    return var_mean / (var_mean + variance.mean(axis=1))


# Parameter values (response scale) for checking the moments against PyMC draws.
# The values make the higher moments finite, so the Monte Carlo error is small.
MOMENTS_CASES = {
    "asymmetriclaplace": {"mu": 1.0, "b": 1.5, "kappa": 2.0},
    "bernoulli": {"p": 0.3},
    "beta": {"mu": 0.3, "kappa": 5.0},
    "beta_binomial": {"mu": 0.3, "kappa": 5.0, "n": 10.0},
    "binomial": {"p": 0.3, "n": 10.0},
    "exgaussian": {"mu": 1.0, "sigma": 0.5, "nu": 2.0},
    "exponential": {"mu": 2.0},
    "gamma": {"mu": 2.0, "alpha": 3.0},
    "gaussian": {"mu": 1.0, "sigma": 2.0},
    "hurdle_gamma": {"mu": 2.0, "alpha": 3.0, "psi": 0.7},
    "hurdle_lognormal": {"mu": 0.5, "sigma": 0.4, "psi": 0.7},
    "hurdle_negativebinomial": {"mu": 2.0, "alpha": 3.0, "psi": 0.7},
    "hurdle_poisson": {"mu": 2.0, "psi": 0.7},
    "laplace": {"mu": 1.0, "b": 1.5},
    "loglogistic": {"mu": 0.5, "alpha": 0.2},
    "lognormal": {"mu": 0.5, "sigma": 0.4},
    "negativebinomial": {"mu": 2.0, "alpha": 3.0},
    "ordered_beta": {"mu": 0.6, "kappa": 5.0, "cut0": -1.0, "logdelta": 0.5},
    "poisson": {"mu": 2.0},
    "t": {"mu": 1.0, "sigma": 2.0, "nu": 10.0},
    "wald": {"mu": 2.0, "lam": 4.0},
    "weibull": {"mu": 2.0, "alpha": 3.0},
    "weibull_ph": {"lam": 0.5, "alpha": 2.0},
    "zero_inflated_binomial": {"p": 0.3, "psi": 0.7, "n": 10.0},
    "zero_inflated_negativebinomial": {"mu": 2.0, "alpha": 3.0, "psi": 0.7},
    "zero_inflated_poisson": {"mu": 2.0, "psi": 0.7},
    "zero_one_inflated_beta": {"mu": 0.4, "kappa": 5.0, "zoi": 0.2, "coi": 0.3},
}


@pytest.mark.parametrize("family_name", MOMENTS_CASES)
def test_moments_match_response_distribution(family_name):
    """The moments match the ones of draws from the PyMC distribution used in the likelihood"""
    values = MOMENTS_CASES[family_name]
    family = get_builtin_family(family_name)

    mean, variance = family.moments({k: xr.DataArray(v) for k, v in values.items()})

    kwargs = {k: np.asarray(v) for k, v in values.items() if k != "n"}
    transform = transforms_registry.get_parameter_transform(family)
    if transform is not None:
        kwargs = transform(kwargs)
    if "n" in values:
        kwargs["n"] = values["n"]
    dist = get_distribution_from_likelihood(family.likelihood).dist(**kwargs, size=200_000)
    draws = pm.draw(dist, random_seed=1061)

    assert np.isclose(float(mean), draws.mean(), rtol=0.02, atol=0.01)
    assert np.isclose(float(variance), draws.var(), rtol=0.05)


@pytest.mark.parametrize(
    "family_name",
    [
        "acat",
        "categorical",
        "cratio",
        "cumulative",
        "osm",
        "sratio",
        "multinomial",
        "dirichlet_multinomial",
        "cure_exponential",
        "cure_gamma",
        "cure_lognormal",
        "cure_loglogistic",
        "cure_weibull",
        "cure_weibull_ph",
    ],
)
def test_moments_not_defined(family_name):
    family = get_builtin_family(family_name)
    with pytest.raises(NotImplementedError, match="not"):
        family.moments({})


def test_vonmises_circular_variance():
    family = get_builtin_family("vonmises")
    kappa = xr.DataArray([0.5, 2.0, 50.0, 1e4])
    mean, variance = family.moments({"mu": xr.DataArray(0.3), "kappa": kappa})
    assert float(mean) == 0.3
    # Circular variance of the von Mises distribution, 1 - I1(kappa) / I0(kappa)
    from scipy.special import i0, i1

    np.testing.assert_allclose(variance[:3], 1 - i1(kappa[:3]) / i0(kappa[:3]))
    # Stays finite for very concentrated distributions
    assert np.all(np.isfinite(variance)) and 0 < float(variance[-1]) < 1e-3

    # Monte Carlo check against the circular variance of PyMC draws
    draws = pm.draw(pm.VonMises.dist(mu=0.3, kappa=2.0, size=200_000), random_seed=1061)
    mc_variance = 1 - np.abs(np.mean(np.exp(1j * draws)))
    assert np.isclose(float(variance[1]), mc_variance, rtol=0.02)


def test_r2_vonmises_constructed(data):
    angles = np.arctan2(np.sin(data["y"]), np.cos(data["y"]))
    model = bmb.Model("angle ~ x", data.assign(angle=angles), family="vonmises")
    rng = np.random.default_rng(5)
    mu = rng.uniform(-np.pi, np.pi, size=(2, 5, N_OBS))
    kappa = rng.uniform(0.5, 5, size=(2, 5))
    idata = make_idata({"mu": mu, "kappa": kappa})

    from scipy.special import i0, i1

    circ_var = 1 - i1(kappa) / i0(kappa)
    expected = bayesian_r2(
        make_idata({"mu": mu, "variance": circ_var}),
        pred_mean="mu",
        scale="variance",
        scale_kind="var",
        circular=True,
        summary=False,
    )
    np.testing.assert_allclose(model.r2_score(idata, summary=False), expected)


def test_r2_gaussian_constructed(data):
    model = bmb.Model("y ~ x", data)
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(2, 5, N_OBS))
    sigma = rng.uniform(0.5, 2, size=(2, 5))
    idata = make_idata({"mu": mu, "sigma": sigma}, observed={"y": data["y"]})

    r2_draws = model.r2_score(idata, summary=False)
    expected = manual_bayesian_r2(mu, sigma[..., np.newaxis] ** 2)
    np.testing.assert_allclose(r2_draws, expected)
    # Same as ArviZ's Bayesian R^2 with the Gaussian sigma
    np.testing.assert_allclose(
        r2_draws, bayesian_r2(idata, pred_mean="mu", scale="sigma", summary=False)
    )
    assert model.r2_score(idata) == bayesian_r2(idata, pred_mean="mu", scale="sigma")


def test_r2_distributional_and_constant_parameters(data):
    # sigma varies by observation
    model = bmb.Model(bmb.Formula("y ~ x", "sigma ~ x"), data)
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(2, 5, N_OBS))
    sigma = rng.uniform(0.5, 2, size=(2, 5, N_OBS))
    idata = make_idata({"mu": mu, "sigma": sigma})
    np.testing.assert_allclose(
        model.r2_score(idata, summary=False), manual_bayesian_r2(mu, sigma**2)
    )

    # sigma is fixed to a constant through the prior, so it's not in the posterior
    model = bmb.Model("y ~ x", data, priors={"sigma": 2})
    idata = make_idata({"mu": mu})
    np.testing.assert_allclose(
        model.r2_score(idata, summary=False), manual_bayesian_r2(mu, np.full_like(mu, 4))
    )


def test_r2_bernoulli_constructed(data):
    model = bmb.Model("binary ~ x", data, family="bernoulli")
    p = np.random.default_rng(3).uniform(0.05, 0.95, size=(2, 5, N_OBS))
    idata = make_idata({"p": p})

    r2_draws = model.r2_score(idata, summary=False)
    np.testing.assert_allclose(r2_draws, manual_bayesian_r2(p, p * (1 - p)))
    # ArviZ uses the same Bernoulli variance when 'scale=None'
    np.testing.assert_allclose(
        r2_draws, bayesian_r2(idata, pred_mean="p", scale=None, summary=False)
    )


def test_r2_binomial_uses_trials(data):
    model = bmb.Model("p(successes, trials) ~ x", data, family="binomial")
    p = np.random.default_rng(4).uniform(0.05, 0.95, size=(2, 5, N_OBS))
    idata = make_idata({"p": p})

    n = data["trials"].to_numpy(dtype=float)
    expected = manual_bayesian_r2(n * p, n * p * (1 - p))
    np.testing.assert_allclose(model.r2_score(idata, summary=False), expected)


def test_r2_negativebinomial_constructed(data):
    model = bmb.Model("count ~ x", data, family="negativebinomial")
    rng = np.random.default_rng(5)
    mu = rng.uniform(0.5, 5, size=(2, 5, N_OBS))
    alpha = rng.uniform(1, 5, size=(2, 5))
    idata = make_idata({"mu": mu, "alpha": alpha})

    expected = manual_bayesian_r2(mu, mu + mu**2 / alpha[..., np.newaxis])
    np.testing.assert_allclose(model.r2_score(idata, summary=False), expected)


def test_r2_t_infinite_variance_falls_back_to_residual(data):
    model = bmb.Model("y ~ x", data, family="t")
    mu = np.random.default_rng(6).normal(size=(1, 3, N_OBS))
    nu = np.array([[1.5, 3.0, 10.0]])
    idata = make_idata({"mu": mu, "sigma": np.ones((1, 3)), "nu": nu}, observed={"y": data["y"]})

    with pytest.warns(UserWarning, match="residual R"):
        r2_draws = model.r2_score(idata, summary=False)
    expected = residual_r2(idata, pred_mean="mu", obs_name="y", summary=False)
    np.testing.assert_allclose(r2_draws, expected)


def test_r2_t_finite_variance_is_bayesian(data):
    model = bmb.Model("y ~ x", data, family="t")
    mu = np.random.default_rng(7).normal(size=(1, 3, N_OBS))
    nu = np.array([[3.0, 5.0, 10.0]])
    idata = make_idata({"mu": mu, "sigma": np.ones((1, 3)), "nu": nu})

    variance = np.broadcast_to((nu / (nu - 2))[..., np.newaxis], mu.shape)
    np.testing.assert_allclose(
        model.r2_score(idata, summary=False), manual_bayesian_r2(mu, variance)
    )


def test_loglogistic_infinite_variance():
    family = get_builtin_family("loglogistic")
    params = {"mu": xr.DataArray(0.0), "alpha": xr.DataArray([0.25, 0.75, 1.5], dims="draw")}
    with pytest.warns(UserWarning, match="alpha"):
        mean, variance = family.moments(params)
    assert np.isfinite(variance[0]) and np.isinf(variance[1:]).all()
    assert np.isfinite(mean[:2]).all() and np.isinf(mean[2])


def test_r2_unsupported(data):
    model = bmb.Model("category ~ x", data, family="categorical")
    p = np.full((1, 2, N_OBS, 3), 1 / 3)
    posterior = xr.Dataset({"p": (("chain", "draw", "__obs__", "category_dim"), p)})
    idata = xr.DataTree.from_dict({"posterior": posterior})
    with pytest.raises(NotImplementedError, match="'categorical' family"):
        model.r2_score(idata)

    # A custom family that does not implement the moments
    likelihood = bmb.Likelihood("Normal", params=["mu", "sigma"], parent="mu")
    family = bmb.Family("my_gaussian", likelihood, "identity")
    model = bmb.Model("y ~ x", data, family=family, priors={"sigma": bmb.Prior("HalfNormal")})
    idata = make_idata({"mu": np.zeros((1, 2, N_OBS)), "sigma": np.ones((1, 2))})
    with pytest.raises(NotImplementedError, match="'my_gaussian' family"):
        model.r2_score(idata)


def test_r2_censored_not_supported(data):
    model = bmb.Model("censored(y, status) ~ x", data)
    idata = make_idata({"mu": np.zeros((1, 2, N_OBS)), "sigma": np.ones((1, 2))})
    with pytest.raises(NotImplementedError, match="censored"):
        model.r2_score(idata)


@pytest.mark.parametrize(
    "formula, family, parent",
    [("binary ~ x", "bernoulli", "p"), ("count ~ x", "poisson", "mu")],
)
def test_r2_fitted_model(data, formula, family, parent):
    model = bmb.Model(formula, data, family=family)
    idata = model.fit(tune=100, draws=100, chains=1, random_seed=1061, progressbar=False)
    assert parent not in idata["posterior"]

    summary = model.r2_score(idata)
    r2_draws = model.r2_score(idata, summary=False)

    # The response parameters are computed and added to the posterior when they are missing
    assert parent in idata["posterior"]
    mean = idata["posterior"][parent].to_numpy()
    variance = mean * (1 - mean) if family == "bernoulli" else mean
    np.testing.assert_allclose(r2_draws, manual_bayesian_r2(mean, variance))
    assert r2_draws.shape == (100,)
    assert np.all((r2_draws >= 0) & (r2_draws <= 1))
    assert 0 < summary[0] < 1
