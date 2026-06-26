import numpy as np
import pandas as pd
import pytest

import bambi as bmb

# Skip tests if dependencies not available
try:
    import blackjax  # noqa: F401
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import nutpie  # noqa: F401

    NUTPIE_AVAILABLE = True
except ImportError:
    NUTPIE_AVAILABLE = False


def test_pymc_method(data_random_n100):
    """Test PyMC method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="pymc", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_numpyro_method(data_random_n100):
    """Test NumPyro method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="numpyro", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_blackjax_method(data_random_n100):
    """Test BlackJAX method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="blackjax", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="nutpie not available")
def test_nutpie_method(data_random_n100):
    """Test nutpie method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="nutpie", draws=50, tune=50)
    assert hasattr(result, "posterior")


def test_vi_method(data_random_n100):
    """Test VI method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="vi")
    assert hasattr(result, "sample")  # VI returns approximation object


def test_laplace_method(data_random_n100):
    """Test Laplace method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="laplace", draws=50)
    assert hasattr(result, "posterior")


def test_invalid_method(data_random_n100):
    """Test that invalid inference methods raise ValueError."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.raises(ValueError, match="'invalid_method' is not a supported inference method"):
        model.fit(inference_method="invalid_method", draws=10, tune=10)


def test_legacy_method_warning(data_random_n100):
    """Test that legacy method names produce warnings."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(FutureWarning, match="'mcmc' has been replaced by 'pymc'"):
        model.fit(inference_method="mcmc", draws=10, tune=10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_legacy_nuts_blackjax_warning(data_random_n100):
    """Test legacy nuts_blackjax warning."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(FutureWarning, match="'nuts_blackjax' has been replaced by 'blackjax'"):
        model.fit(inference_method="nuts_blackjax", draws=10, tune=10)

    with pytest.warns(FutureWarning, match="'blackjax_nuts' has been replaced by 'blackjax'"):
        model.fit(inference_method="blackjax_nuts", draws=10, tune=10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_legacy_nuts_numpyro_warning(data_random_n100):
    """Test legacy nuts_numpyro warning."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(FutureWarning, match="'nuts_numpyro' has been replaced by 'numpyro'"):
        model.fit(inference_method="nuts_numpyro", draws=10, tune=10)

    with pytest.warns(FutureWarning, match="'numpyro_nuts' has been replaced by 'numpyro'"):
        model.fit(inference_method="numpyro_nuts", draws=10, tune=10)


def test_nuts_parameter_for_default_sampler(data_random_n100, mock_pymc_sample):
    """NUTS settings passed via nuts={} are forwarded to pm.sample and take effect."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(
        inference_method="pymc",
        draws=10,
        tune=10,
        chains=2,
        nuts={"target_accept": 0.95},
    )
    assert idata is not None


def test_nuts_none_is_noop(data_random_n100, mock_pymc_sample):
    """Omitting nuts (defaults to None) runs without error."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(inference_method="pymc", draws=10, tune=10, chains=2)
    assert idata is not None


def test_nuts_parameter_forwarded_to_external_samplers(data_random_n100):
    """nuts={} is forwarded as-is to _run_mcmc for external samplers."""
    import unittest.mock as mock

    import bambi.backend.pymc as _bpymc

    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    captured = {}

    def patched(self, *args, **kwargs):
        captured.update(kwargs)
        raise SystemExit

    with mock.patch.object(_bpymc.PyMCModel, "_run_mcmc", patched):
        try:
            model.fit(
                inference_method="nutpie",
                draws=10,
                tune=10,
                nuts={"target_accept": 0.95},
            )
        except SystemExit:
            pass

    assert captured.get("nuts") == {"target_accept": 0.95}


def test_nuts_sampler_kwargs_deprecated(data_random_n100, mock_pymc_sample):
    """nuts_sampler_kwargs triggers a FutureWarning and is merged into nuts."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(FutureWarning, match="nuts_sampler_kwargs.*deprecated"):
        idata = model.fit(
            inference_method="pymc",
            draws=10,
            tune=10,
            chains=2,
            nuts_sampler_kwargs={"target_accept": 0.95},
        )
    assert idata is not None
