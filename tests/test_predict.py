from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

import bambi as bmb


@pytest.fixture(scope="module")
def data_numeric_xy():
    rng = np.random.default_rng(121195)
    x = rng.uniform(size=100)
    y = x + rng.normal(scale=0.5, size=100)
    data = pd.DataFrame({"y": y, "x": x})
    return data


@pytest.fixture(scope="module")
def data_bernoulli():
    # Taken from https://juanitorduz.github.io/glm_pymc3/
    rng = np.random.default_rng(121195)
    n = 250
    x1 = rng.normal(loc=0.0, scale=2.0, size=n)
    x2 = rng.normal(loc=0.0, scale=2.0, size=n)
    intercept = -0.5
    beta_x1 = 1
    beta_x2 = -1
    beta_interaction = 2
    z = intercept + beta_x1 * x1 + beta_x2 * x2 + beta_interaction * x1 * x2
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(n=1, p=p, size=n)
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
    return df


@pytest.fixture(scope="module")
def data_beta():
    return pd.read_csv(join(dirname(__file__), "data", "gasoline.csv"))


@pytest.fixture(scope="module")
def data_gamma():
    rng = np.random.default_rng(121195)
    N = 200
    a, b, shape = 0.5, 1.1, 10
    x = rng.uniform(-1, 1, N)
    y = rng.gamma(shape, np.exp(a + b * x) / shape, N)
    data = pd.DataFrame({"x": x, "y": y})
    return data


@pytest.fixture(scope="module")
def data_count():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame({"y": rng.poisson(list(range(10)) * 10), "x": rng.uniform(size=100)})
    return data


@pytest.fixture(scope="module")
def inhaler():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "inhaler.csv"))
    data["rating"] = pd.Categorical(data["rating"], categories=[1, 2, 3, 4])
    return data


def test_predict_bernoulli(data_bernoulli):
    data = data_bernoulli
    model = bmb.Model("y ~ x1*x2", data, family="bernoulli")
    idata = model.fit(tune=100, draws=100, target_accept=0.9)

    # In sample prediction
    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
    assert (idata.posterior_predictive["y"].isin([0, 1])).all()

    # Out of sample prediction
    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
    assert (idata.posterior_predictive["y"].isin([0, 1])).all()


def test_predict_beta(data_beta):
    data = data_beta
    data["batch"] = pd.Categorical(data["batch"], [10, 1, 2, 3, 4, 5, 6, 7, 8, 9], ordered=True)
    model = bmb.Model("yield ~ temp + batch", data, family="beta")
    idata = model.fit(tune=100, draws=100, target_accept=0.90)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
    assert (0 < idata.posterior_predictive["yield"]).all() & (
        idata.posterior_predictive["yield"] < 1
    ).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
    assert (0 < idata.posterior_predictive["yield"]).all() & (
        idata.posterior_predictive["yield"] < 1
    ).all()


def test_predict_gamma(data_gamma):
    data = data_gamma

    model = bmb.Model("y ~ x", data, family="gamma", link="log")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()


def test_predict_gaussian(data_numeric_xy):
    data = data_numeric_xy
    model = bmb.Model("y ~ x", data, family="gaussian")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_negativebinomial(data_count):
    data = data_count

    model = bmb.Model("y ~ x", data, family="negativebinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all()
    assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all()
    assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()


def test_predict_poisson(data_count):
    data = data_count

    model = bmb.Model("y ~ x", data, family="negativebinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all()
    assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all()
    assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()


def test_predict_t(data_numeric_xy):
    data = data_numeric_xy
    model = bmb.Model("y ~ x", data, family="t")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    # A case where the prior for one of the parameters is constant
    model = bmb.Model("y ~ x", data, family="t", priors={"nu": 4})
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_wald(data_gamma):
    data = data_gamma

    model = bmb.Model("y ~ x", data, family="wald", link="log")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()


def test_predict_categorical(inhaler):
    model = bmb.Model("rating ~ period + carry + treat", inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model.predict(idata, data=inhaler.iloc[:20, :])
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model = bmb.Model("rating ~ period + carry + treat + (1|subject)", inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model.predict(idata, data=inhaler.iloc[:20, :])
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)


def test_posterior_predictive_categorical(inhaler):
    model = bmb.Model("rating ~ period", data=inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)
    model.predict(idata, kind="pps")
    pps = idata.posterior_predictive["rating"].to_numpy()

    assert pps.shape[-1] == inhaler.shape[0]
    assert (np.unique(pps) == [0, 1, 2, 3]).all()


def test_predict_categorical_group_specific():
    # see https://github.com/bambinos/bambi/issues/447
    rng = np.random.default_rng(1234)
    size = 100

    data = pd.DataFrame(
        {
            "y": rng.choice([0, 1], size=size),
            "x1": rng.choice(list("abcd"), size=size),
            "x2": rng.choice(list("XY"), size=size),
            "x3": rng.normal(size=size),
        }
    )

    model = bmb.Model("y ~ x1 + (0 + x2|x1) + (0 + x3|x1 + x2)", data, family="bernoulli")

    idata = model.fit(tune=100, draws=100, chains=2)

    model.predict(idata, data=data)

    assert idata.posterior.y_mean.values.shape == (2, 100, 100)
    assert (idata.posterior.y_mean.values > 0).all() and (idata.posterior.y_mean.values < 1).all()


def test_predict_multinomial(inhaler):
    df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
    df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    model.predict(idata, data=df.iloc[:3, :])

    # Numerical predictors
    model = bmb.Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    model.predict(idata, data=df.iloc[:3, :])

    # Categorical predictors
    df["treat"] = df["treat"].replace({-0.5: "A", 0.5: "B"})
    df["carry"] = df["carry"].replace({-1: "a", 0: "b", 1: "c"})

    model = bmb.Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    model.predict(idata, data=df.iloc[:3, :])

    data = pd.DataFrame(
        {
            "state": ["A", "B", "C"],
            "y1": [35298, 1885, 5775],
            "y2": [167328, 20731, 21564],
            "y3": [212682, 37716, 20222],
            "y4": [37966, 5196, 3277],
        }
    )

    # Contains group-specific effect
    model = bmb.Model(
        "c(y1, y2, y3, y4) ~ 1 + (1 | state)", data, family="multinomial", noncentered=False
    )
    idata = model.fit(tune=100, draws=100, random_seed=0)

    model.predict(idata)
    model.predict(idata, kind="pps")


def test_posterior_predictive_multinomial(inhaler):
    df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
    df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)


def test_predict_include_group_specific():
    rng = np.random.default_rng(1234)
    size = 100

    data = pd.DataFrame(
        {
            "y": rng.choice([0, 1], size=size),
            "x1": rng.choice(list("abcd"), size=size),
        }
    )

    model = bmb.Model("y ~ 1 + (1|x1)", data, family="bernoulli")
    idata = model.fit(tune=100, draws=100, chains=2, random_seed=1234)
    idata_1 = model.predict(idata, data=data, inplace=False, include_group_specific=True)
    idata_2 = model.predict(idata, data=data, inplace=False, include_group_specific=False)

    assert not np.isclose(
        idata_1.posterior["y_mean"].values,
        idata_2.posterior["y_mean"].values,
    ).all()

    # Since it's an intercept-only model, predictions are the same for all observations if
    # we drop group-specific terms.
    assert (idata_2.posterior["y_mean"] == idata_2.posterior["y_mean"][:, :, 0]).all()

    # When we include group-specific terms, these predictions are different
    assert not (idata_1.posterior["y_mean"] == idata_1.posterior["y_mean"][:, :, 0]).all()


def test_predict_offset():
    # Simple case

    data = bmb.load_data("carclaims")
    model = bmb.Model("numclaims ~ offset(np.log(exposure))", data, family="poisson", link="log")
    idata = model.fit(tune=100, draws=100, chains=2, random_seed=1234)
    model.predict(idata)
    model.predict(idata, kind="pps")

    # More complex case
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.poisson(20, size=100),
            "x": rng.normal(size=100),
            "group": np.tile(np.arange(10), 10),
        }
    )
    data["time"] = data["y"] - rng.normal(loc=1, size=100)
    model = bmb.Model("y ~ offset(np.log(time)) + x + (1 | group)", data, family="poisson")
    idata = model.fit(tune=100, draws=100, chains=2, target_accept=0.9, random_seed=1234)
    model.predict(idata, kind="pps")


def test_posterior_predictive_dirichlet_multinomial(inhaler):
    df = inhaler.groupby(["treat", "rating"], as_index=False).size()
    df = df.pivot(index=["treat"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="dirichlet_multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)

    # With predictor only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 0 + treat", df, family="dirichlet_multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)


def test_posterior_predictive_beta_binomial():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    model = bmb.Model("prop(y, n) ~ x", data, family="beta_binomial")
    idata = model.fit(draws=100, tune=100)
    model.predict(idata, kind="pps")

    n = data["n"].to_numpy()
    assert np.all(idata.posterior_predictive["prop(y, n)"].values <= n[np.newaxis, np.newaxis, :])