import bambi as bmb
import numpy as np
import pandas as pd

import pytest


@pytest.fixture(scope="module")
def data_n100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "b1": rng.binomial(n=1, p=0.5, size=size),
            "n1": rng.poisson(lam=2, size=size),
            "n2": rng.poisson(lam=2, size=size),
            "y1": rng.normal(size=size),
            "y2": rng.normal(size=size),
            "y3": rng.normal(size=size),
            "cat2": rng.choice(["a", "b"], size=size),
            "cat4": rng.choice(list("MNOP"), size=size),
            "cat5": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


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
        ("numpyro_nuts", {"chain_method": "vectorized"}),
        ("blackjax_nuts", {"chain_method": "vectorized"}),
    ],
)
def test_logistic_regression_categoric_alternative_samplers(data_n100, args):
    model = bmb.Model("b1 ~ n1", data_n100, family="bernoulli")
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])


@pytest.mark.parametrize(
    "args",
    [
        ("mcmc", {}),
        ("numpyro_nuts", {"chain_method": "vectorized"}),
        ("blackjax_nuts", {"chain_method": "vectorized"}),
    ],
)
def test_regression_alternative_samplers(data_n100, args):
    model = bmb.Model("n1 ~ n2", data_n100)
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])
