import pytest

import bambi as bmb
import numpy as np
import pandas as pd

from helpers import assert_ip_dlogp


@pytest.fixture(scope="module")
def data_gamma():
    rng = np.random.default_rng(121195)
    size = 200
    a, b = 0.5, 1.1
    x = rng.uniform(-1.5, 1.5, size)
    shape = np.exp(0.3 + x * 0.5 + rng.normal(scale=0.1, size=size))
    y = rng.gamma(shape, np.exp(a + b * x) / shape, size)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def data_new_gamma():
    return pd.DataFrame({"x": np.linspace(-1.5, 1.5, num=50)})


@pytest.fixture(scope="module")
def data_normal():
    data = bmb.load_data("bikes")
    data.sort_values(by="hour", inplace=True)
    data_cnt_om = data["count"].mean()
    data_cnt_os = data["count"].std()
    data["count_normalized"] = (data["count"] - data_cnt_om) / data_cnt_os
    data = data[::50]
    data = data.reset_index(drop=True)
    return data


@pytest.fixture(scope="module")
def data_new_normal():
    return pd.DataFrame({"hour": np.linspace(0, 23, num=200)})


def test_normal_with_splines(data_normal, data_new_normal, mock_pymc_sample):
    knots = np.linspace(0, 23, 8)[1:-1]
    knots_s = np.linspace(0, 23, 5)[1:-1]
    formula = bmb.Formula(
        "count_normalized ~ 0 + bs(hour, knots=knots, intercept=True)",
        "sigma ~ 0 + bs(hour, knots=knots_s, intercept=True)",
    )
    model = bmb.Model(formula, data_normal)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, kind="response")
    model.predict(idata, kind="response", data=data_new_normal)


def test_gamma(data_gamma, data_new_gamma, mock_pymc_sample):
    formula = bmb.Formula("y ~ x", "alpha ~ x")

    # NOTE: Inverse link is broken with 'mu'
    # Default links
    # model = bmb.Model(formula, data_gamma, family="gamma")
    # idata = model.fit(tune=100, draws=100, random_seed=1234)
    # model.predict(idata, kind="pps")
    # model.predict(idata, kind="pps", data=data_new_gamma)

    # Custom links
    model = bmb.Model(formula, data_gamma, family="gamma", link={"mu": "log", "alpha": "log"})
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, kind="response")
    model.predict(idata, kind="response", data=data_new_gamma)


def test_gamma_with_splines(data_normal, data_new_normal, mock_pymc_sample):
    formula = bmb.Formula(
        "count ~ 0 + bs(hour, 8, intercept=True)", "alpha ~ 0 + bs(hour, 8, intercept=True)"
    )
    model = bmb.Model(formula, data_normal, family="gamma", link={"mu": "log", "alpha": "log"})
    idata = model.fit(chains=2)
    model.build()
    assert_ip_dlogp(model)
    model.predict(idata, kind="response")
    model.predict(idata, kind="response", data=data_new_normal)
