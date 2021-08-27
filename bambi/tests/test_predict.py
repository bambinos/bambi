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


def test_predict_bernoulli(data_bernoulli):
    data = data_bernoulli
    model = Model("y ~ x1*x2", data, family="bernoulli")
    idata = model.fit(target_accept=0.90)

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
    idata = model.fit(target_accept=0.90)

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
    idata = model.fit()

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
    idata = model.fit()

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_negativebinomial(data_count):
    data = data_count

    model = Model("y ~ x", data, family="negativebinomial")
    idata = model.fit()

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
    idata = model.fit()

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
    idata = model.fit()

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])


def test_predict_wald(data_gamma):
    data = data_gamma

    model = Model("y ~ x", data, family="wald", link="log")
    idata = model.fit()

    model.predict(idata, kind="mean")
    model.predict(idata, kind="pps")

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()

    model.predict(idata, kind="mean", data=data.iloc[:20, :])
    model.predict(idata, kind="pps", data=data.iloc[:20, :])

    assert (0 < idata.posterior["y_mean"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()
