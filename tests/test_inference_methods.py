import warnings

import bambi as bmb
import numpy as np
import pandas as pd
import pytest

# Skip tests if dependencies not available
try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401
    import blackjax  # noqa: F401

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


# Distinctive phrase from Bambi's own warning, so we don't match PyMC's separate
# `nuts_sampler_kwargs`-deprecation FutureWarning (which also fires on PyMC >= 6).
_NSK_WARNING = "not recommended with the default PyMC sampler"


def test_nuts_sampler_kwargs_on_pymc_warns(data_random_n100):
    """Routing NUTS settings through ``nuts_sampler_kwargs`` on the default PyMC sampler warns.

    Guards the footgun where settings like ``target_accept`` placed in
    ``nuts_sampler_kwargs`` are silently ignored (PyMC < 6) or deprecated (PyMC >= 6),
    instead of being passed directly to ``Model.fit()``.
    """
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(UserWarning, match=_NSK_WARNING):
        model.fit(
            inference_method="pymc",
            draws=10,
            tune=10,
            nuts_sampler_kwargs={"target_accept": 0.95},
        )


def test_target_accept_directly_does_not_warn(data_random_n100):
    """Passing NUTS settings directly is the supported path and must not raise our warning."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        model.fit(inference_method="pymc", draws=10, tune=10, target_accept=0.95)
    assert not any(_NSK_WARNING in str(record.message) for record in records)


def test_empty_nuts_sampler_kwargs_does_not_warn(data_random_n100):
    """An empty / falsy ``nuts_sampler_kwargs`` does not trigger our warning."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        model.fit(inference_method="pymc", draws=10, tune=10, nuts_sampler_kwargs={})
    assert not any(_NSK_WARNING in str(record.message) for record in records)
