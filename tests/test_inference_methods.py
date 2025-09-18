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


@pytest.fixture
def simple_data():
    """Simple dataset for testing."""
    np.random.seed(42)
    size = 50
    data = pd.DataFrame({"y": np.random.normal(0, 1, size), "x": np.random.normal(0, 1, size)})
    return data


def test_pymc_method(simple_data):
    """Test PyMC method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="pymc", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_numpyro_method(simple_data):
    """Test NumPyro method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="numpyro", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_blackjax_method(simple_data):
    """Test BlackJAX method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="blackjax", draws=50, tune=50)
    assert hasattr(result, "posterior")


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="nutpie not available")
def test_nutpie_method(simple_data):
    """Test nutpie method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="nutpie", draws=50, tune=50)
    assert hasattr(result, "posterior")


def test_vi_method(simple_data):
    """Test VI method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="vi")
    assert hasattr(result, "sample")  # VI returns approximation object


def test_laplace_method(simple_data):
    """Test Laplace method runs successfully."""
    model = bmb.Model("y ~ x", simple_data)
    result = model.fit(inference_method="laplace", draws=50)
    assert hasattr(result, "posterior")


def test_invalid_method(simple_data):
    """Test that invalid inference methods raise ValueError."""
    model = bmb.Model("y ~ x", simple_data)
    with pytest.raises(ValueError, match="'invalid_method' is not a supported inference method"):
        model.fit(inference_method="invalid_method", draws=10, tune=10)


def test_legacy_method_warning(simple_data):
    """Test that legacy method names produce warnings."""
    model = bmb.Model("y ~ x", simple_data)
    with pytest.warns(FutureWarning, match="'mcmc' has been replaced by 'pymc'"):
        model.fit(inference_method="mcmc", draws=10, tune=10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_legacy_nuts_blackjax_warning(simple_data):
    """Test legacy nuts_blackjax warning."""
    model = bmb.Model("y ~ x", simple_data)
    with pytest.warns(FutureWarning, match="'nuts_blackjax' has been replaced by 'blackjax'"):
        model.fit(inference_method="nuts_blackjax", draws=10, tune=10)

    with pytest.warns(FutureWarning, match="'blackjax_nuts' has been replaced by 'blackjax'"):
        model.fit(inference_method="blackjax_nuts", draws=10, tune=10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_legacy_nuts_numpyro_warning(simple_data):
    """Test legacy nuts_numpyro warning."""
    model = bmb.Model("y ~ x", simple_data)
    with pytest.warns(FutureWarning, match="'nuts_numpyro' has been replaced by 'numpyro'"):
        model.fit(inference_method="nuts_numpyro", draws=10, tune=10)

    with pytest.warns(FutureWarning, match="'numpyro_nuts' has been replaced by 'numpyro'"):
        model.fit(inference_method="numpyro_nuts", draws=10, tune=10)
