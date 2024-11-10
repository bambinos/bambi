import bambi as bmb
import bayeux as bx
import numpy as np
import pandas as pd

import pytest

#pytestmark = pytest.mark.skip("JAX DEPS ARE BROKEN!")

MCMC_METHODS = [getattr(bx.mcmc, k).name for k in bx.mcmc.__all__]
MCMC_METHODS_FILTERED = [
    i for i in MCMC_METHODS if not any(x in i for x in ("flowmc", "chees", "meads"))
]


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


def test_inference_method_names_and_kwargs():
    names = bmb.inference_methods.names

    # Check PyMC inference method family
    assert "mcmc" in names["pymc"].keys()
    assert "vi" in names["pymc"].keys()

    # Check bayeux inference method family. Currently, only MCMC methods are supported
    assert "mcmc" in names["bayeux"].keys()

    # Ensure get_kwargs method raises an error if a non-supported method name is passed
    with pytest.raises(
        ValueError,
        match="Inference method 'not_a_method' not found in the list of available methods. Use `bmb.inference_methods.names` to list the available methods.",
    ):
        bmb.inference_methods.get_kwargs("not_a_method")


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


@pytest.mark.parametrize("sampler", MCMC_METHODS_FILTERED)
def test_logistic_regression_categoric_alternative_samplers(data_n100, sampler):
    model = bmb.Model("b1 ~ n1", data_n100, family="bernoulli")
    model.fit(inference_method=sampler)


@pytest.mark.parametrize("sampler", MCMC_METHODS)
def test_regression_alternative_samplers(data_n100, sampler):
    model = bmb.Model("n1 ~ n2", data_n100)
    model.fit(inference_method=sampler)
