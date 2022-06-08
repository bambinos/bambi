from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from bambi.models import Model


@pytest.fixture(scope="module")
def data_numeric_xy():
    x = np.random.uniform(size=100)
    y = x + np.random.normal(scale=0.5, size=100)
    data = pd.DataFrame({"y": y, "x": x})
    return data


@pytest.fixture(scope="module")
def data_bernoulli():
    # Taken from https://juanitorduz.github.io/glm_pymc3/
    n = 250
    x1 = np.random.normal(loc=0.0, scale=2.0, size=n)
    x2 = np.random.normal(loc=0.0, scale=2.0, size=n)
    intercept = -0.5
    beta_x1 = 1
    beta_x2 = -1
    beta_interaction = 2
    z = intercept + beta_x1 * x1 + beta_x2 * x2 + beta_interaction * x1 * x2
    p = 1 / (1 + np.exp(-z))
    y = np.random.binomial(n=1, p=p, size=n)
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
    return df


@pytest.fixture(scope="module")
def data_beta():
    return pd.read_csv(join(dirname(__file__), "data", "gasoline.csv"))


@pytest.fixture(scope="module")
def data_gamma():
    N = 200
    x = np.random.uniform(-1, 1, N)
    a = 0.5
    b = 1.1
    shape = 10
    y = np.random.gamma(shape, np.exp(a + b * x) / shape, N)
    data = pd.DataFrame({"x": x, "y": y})
    return data


@pytest.fixture(scope="module")
def data_count():
    data = pd.DataFrame(
        {"y": np.random.poisson(list(range(10)) * 10), "x": np.random.uniform(size=100)}
    )
    return data


@pytest.fixture(scope="module")
def inhaler():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "inhaler.csv"))
    data["rating"] = pd.Categorical(data["rating"], categories=[1, 2, 3, 4])
    return data


def test_predict_bernoulli(data_bernoulli):
    data = data_bernoulli
    model = Model("y ~ x1*x2", data, family="bernoulli")
    idata = model.fit(tune=100, draws=100, target_accept=0.90)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
    assert (idata.posterior_predictive["y"].isin([0, 1])).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
    assert (idata.posterior_predictive["y"].isin([0, 1])).all()


def test_predict_beta(data_beta):
    data = data_beta
    data["batch"] = pd.Categorical(data["batch"], [10, 1, 2, 3, 4, 5, 6, 7, 8, 9], ordered=True)
    model = Model("yield ~ temp + batch", data, family="beta")
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

    model = Model("y ~ x", data, family="gamma", link="log")
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
    model = Model("y ~ x", data, family="gaussian")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_negativebinomial(data_count):
    data = data_count

    model = Model("y ~ x", data, family="negativebinomial")
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

    model = Model("y ~ x", data, family="negativebinomial")
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
    model = Model("y ~ x", data, family="t")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_wald(data_gamma):
    data = data_gamma

    model = Model("y ~ x", data, family="wald", link="log")
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
    model = Model("rating ~ period + carry + treat", inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model.predict(idata, data=inhaler.iloc[:20, :])
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model = Model("rating ~ period + carry + treat + (1|subject)", inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)

    model.predict(idata, data=inhaler.iloc[:20, :])
    assert np.allclose(idata.posterior["rating_mean"].values.sum(-1), 1)


def test_posterior_predictive_categorical(inhaler):
    model = Model("rating ~ period", data=inhaler, family="categorical")
    idata = model.fit(tune=100, draws=100)
    model.predict(idata, kind="pps")
    pps = idata.posterior_predictive["rating"].values

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

    model = Model("y ~ x1 + (0 + x2|x1) + (0 + x3|x1 + x2)", data, family="bernoulli")

    idata = model.fit(tune=100, draws=100, chains=2)

    model.predict(idata, data=data)

    assert idata.posterior.y_mean.values.shape == (2, 100, 100)
    assert (idata.posterior.y_mean.values > 0).all() and (idata.posterior.y_mean.values < 1).all()


def test_predict_multinomial(inhaler):
    def c(*args):
        return np.column_stack(args)

    df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
    df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    model.predict(idata, data=df.iloc[:3, :])

    # Numerical predictors
    model = Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    model.predict(idata)
    model.predict(idata, data=df.iloc[:3, :])

    # Categorical predictors
    df["treat"] = df["treat"].replace({-0.5: "A", 0.5: "B"})
    df["carry"] = df["carry"].replace({-1: "a", 0: "b", 1: "c"})

    model = Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
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
    model = Model(
        "c(y1, y2, y3, y4) ~ 1 + (1 | state)", data, family="multinomial", noncentered=False
    )
    idata = model.fit(tune=100, draws=100, random_seed=0)

    model.predict(idata)
    model.predict(idata, kind="pps")


def test_posterior_predictive_multinomial(inhaler):
    def c(*args):
        return np.column_stack(args)

    df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
    df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)


def test_predict_include_group_specific():
    rng = np.random.default_rng(1234)
    size = 100

    data = pd.DataFrame(
        {"y": rng.choice([0, 1], size=size), "x1": rng.choice(list("abcd"), size=size),}
    )

    model = Model("y ~ 1 + (1|x1)", data, family="bernoulli")
    idata = model.fit(tune=100, draws=100, chains=2, random_seed=1234)
    idata_1 = model.predict(idata, data=data, inplace=False, include_group_specific=True)
    idata_2 = model.predict(idata, data=data, inplace=False, include_group_specific=False)

    assert not np.isclose(
        idata_1.posterior["y_mean"].values, idata_2.posterior["y_mean"].values,
    ).any()

    # Since it's an intercept-only model, predictions are the same for all observations if
    # we drop group-specific terms.
    assert (idata_2.posterior["y_mean"] == idata_2.posterior["y_mean"][:, :, 0]).all()

    # When we include group-specific terms, these predictions are different
    assert not (idata_1.posterior["y_mean"] == idata_1.posterior["y_mean"][:, :, 0]).all()
