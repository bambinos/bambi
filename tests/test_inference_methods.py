import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
    """Test PyMC NUTS method runs successfully with custom settings."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(
        inference_method="pymc", chains=2, draws=200, tune=200, nuts={"target_accept": 0.95}
    )
    assert hasattr(result, "posterior")
    assert result.posterior.attrs["inference_library"] == "pymc"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_numpyro_method(data_random_n100):
    """Test NumPyro NUTS method runs successfully with custom settings."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(
        inference_method="numpyro", chains=2, draws=200, tune=200, nuts={"target_accept": 0.95}
    )
    assert hasattr(result, "posterior")
    assert result.posterior.attrs["inference_library"] == "numpyro"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX dependencies not available")
def test_blackjax_method(data_random_n100):
    """Test BlackJAX NUTS method runs successfully with custom settings."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    # Both `progressbar=False` and `chain_method="vectorized"` are needed for blackjax to not fail.
    result = model.fit(
        inference_method="blackjax",
        chains=2,
        draws=200,
        tune=200,
        progressbar=False,
        nuts={"target_accept": 0.95, "chain_method": "vectorized"},
    )
    assert hasattr(result, "posterior")
    assert result.posterior.attrs["inference_library"] == "blackjax"


@pytest.mark.skipif(not NUTPIE_AVAILABLE, reason="nutpie not available")
def test_nutpie_method(data_random_n100):
    """Test nutpie NUTS method runs successfully with custom settings."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(
        inference_method="nutpie", chains=2, draws=200, tune=200, nuts={"target_accept": 0.95}
    )
    assert hasattr(result, "posterior")
    assert result.posterior.attrs["inference_library"] == "nutpie"


def test_vi_method(data_random_n100):
    """Test VI method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="vi", chains=2, draws=200, tune=200)
    assert hasattr(result, "sample")  # VI returns approximation object


def test_laplace_method(data_random_n100):
    """Test Laplace method runs successfully."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    result = model.fit(inference_method="laplace", draws=200)
    assert hasattr(result, "posterior")


def test_laplace_postprocesses_offsets_and_response_params(data_random_n100):
    """Test Laplace applies posterior output options after computing deterministics."""
    model = bmb.Model("continuous1 ~ 1 + (1|binary_cat)", data_random_n100)

    idata = model.fit(
        inference_method="laplace",
        draws=200,
        omit_offsets=True,
        include_response_params=False,
    )
    assert not any(var.endswith("_offset") for var in idata.posterior.data_vars)
    assert "mu" not in idata.posterior

    idata = model.fit(
        inference_method="laplace",
        draws=200,
        omit_offsets=False,
        include_response_params=True,
    )
    assert {var for var in idata.posterior.data_vars if var.endswith("_offset")} == {
        "1|binary_cat_offset"
    }
    assert idata.posterior["mu"].shape == (1, 200, len(data_random_n100))


@pytest.mark.parametrize(
    "inference_method",
    [
        "invalid_method",
        "mcmc",
        "nuts_numpyro",
        "numpyro_nuts",
        "nuts_blackjax",
        "blackjax_nuts",
    ],
)
def test_invalid_method(data_random_n100, inference_method):
    """Test that unsupported inference methods raise ValueError."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.raises(
        ValueError, match=f"'{inference_method}' is not a supported inference method"
    ):
        model.fit(inference_method=inference_method, draws=10, tune=10)


def test_nuts_none_is_noop(data_random_n100, mock_pymc_sample):
    """Omitting nuts (defaults to None) runs without error."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(inference_method="pymc", chains=2, draws=200, tune=200)
    assert idata is not None


def test_nuts_sampler_kwargs_deprecated(data_random_n100, mock_pymc_sample):
    """nuts_sampler_kwargs triggers a FutureWarning and is merged into nuts."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.warns(FutureWarning, match="nuts_sampler_kwargs.*deprecated"):
        idata = model.fit(
            inference_method="pymc",
            draws=100,
            tune=100,
            chains=2,
            nuts_sampler_kwargs={"target_accept": 0.95},
        )
    assert idata is not None


def test_explicit_nuts_overrides_legacy_nuts_sampler_kwargs(data_random_n100, monkeypatch):
    """Explicit nuts settings take precedence over legacy nuts_sampler_kwargs values."""
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    model.build()
    captured = {}

    def mock_sample(**kwargs):
        captured.update(kwargs)
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    monkeypatch.setattr("bambi.backend.pymc.model.pm.sample", mock_sample)

    with pytest.warns(FutureWarning, match="nuts_sampler_kwargs.*deprecated"):
        model.fit(
            chains=2,
            draws=200,
            tune=200,
            nuts_sampler_kwargs={"target_accept": 0.8, "max_treedepth": 12},
            nuts={"target_accept": 0.95},
        )

    assert captured["nuts"] == {"target_accept": 0.95, "max_treedepth": 12}
