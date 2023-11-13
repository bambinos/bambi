import bambi as bmb
import numpy as np
import pandas as pd

import pytest


def test_laplace():
    data = pd.DataFrame(np.repeat((0, 1), (30, 60)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    model = bmb.Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(inference_method="laplace")
    mode_n = results.posterior["Intercept"].mean().item()
    std_n = results.posterior["Intercept"].std().item()
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal((mode_n, std_n), (mode_a.item(), std_a.item()), decimal=2)


def test_vi():
    data = pd.DataFrame(np.repeat((0, 1), (30, 60)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    model = bmb.Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(inference_method="vi", method="advi")
    samples = results.sample(1000).posterior["Intercept"]
    mode_n = samples.mean()
    std_n = samples.std()
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal(
        (mode_n.item(), std_n.item()), (mode_a.item(), std_a.item()), decimal=2
    )


@pytest.mark.parametrize(
    "args",
    [
        ("mcmc", {}),
        ("nuts_numpyro", {"chain_method": "vectorized"}),
        ("nuts_blackjax", {"chain_method": "vectorized"}),
    ],
)
def test_logistic_regression_categoric_alternative_samplers(data_100, args):
    model = bmb.Model("b1 ~ n1", data_100, family="bernoulli")
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])


@pytest.mark.parametrize(
    "args",
    [
        ("mcmc", {}),
        ("nuts_numpyro", {"chain_method": "vectorized"}),
        ("nuts_blackjax", {"chain_method": "vectorized"}),
    ],
)
def test_regression_alternative_samplers(data_100, args):
    model = bmb.Model("n1 ~ n2", data_100)
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])
