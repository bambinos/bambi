import numpy as np
import pytest

import bambi as bmb


@pytest.mark.usefixtures("mock_pymc_sample")
def test_inplace(data_beetle):
    model = bmb.Model("prop(y, n) ~ x", data_beetle, family="binomial")
    idata = model.fit(tune=500, draws=100, chains=4)

    assert "log_prior" not in idata
    assert model.compute_log_prior(idata, inplace=True) is None
    assert "log_prior" in idata

    idata_out = model.compute_log_prior(idata, inplace=False)
    assert "log_prior" in idata_out and idata_out is not idata


@pytest.mark.usefixtures("mock_pymc_sample")
def test_inplace_false(mtcars_fixture):
    model, idata = mtcars_fixture
    out = model.compute_log_prior(idata, inplace=False)
    assert "log_prior" not in idata
    assert "log_prior" in out


def test_exclusion(sleep_study):
    model, idata = sleep_study
    out = model.compute_log_prior(idata, inplace=False)

    det_names = {d.name for d in model.backend.model.deterministics}
    free_names = {rv.name for rv in model.backend.model.free_RVs}
    assert {"1|Subject", "Days|Subject", "mu"} <= det_names
    assert set(out.log_prior.data_vars) == (free_names & set(idata.posterior))


def test_categorical(food_choice):
    model, idata = food_choice
    out = model.compute_log_prior(idata, inplace=False)

    assert {d.name for d in model.backend.model.deterministics} == {"p"}
    assert "p" not in out.log_prior.data_vars
    free_names = {rv.name for rv in model.backend.model.free_RVs}
    assert set(out.log_prior.data_vars) == (free_names & set(idata.posterior))
    for v in out.log_prior.data_vars:
        assert np.isfinite(out.log_prior[v].to_numpy()).all()
