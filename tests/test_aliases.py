import pytest

import bambi as bmb


@pytest.fixture(scope="module")
def my_data():
    return bmb.load_data("my_data")


@pytest.fixture(scope="module")
def anes():
    return bmb.load_data("ANES")


def test_non_distributional_model(my_data):
    # Plain model
    formula = bmb.Formula("y ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "y_obs"]
    assert set(idata.posterior.data_vars) == {"Intercept", "x", "y_mean", "y_sigma"}
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]

    # Model with alises
    model.set_alias({"Intercept": "a", "x": "b", "sigma": "s", "y": "response"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "response_obs"]
    assert set(idata.posterior.data_vars) == {"a", "b", "response_mean", "s"}
    assert list(idata.posterior["response_mean"].coords) == ["chain", "draw", "response_obs"]


def test_distributional_model(my_data):
    formula = bmb.Formula("y ~ x", "sigma ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "y_obs"]
    assert set(idata.posterior.data_vars) == {
        "Intercept",
        "x",
        "sigma_Intercept",
        "sigma_x",
        "y_sigma",
        "y_mean",
    }
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]
    assert list(idata.posterior["y_sigma"].coords) == ["chain", "draw", "y_obs"]

    aliases = {
        "y": {"Intercept": "y_a", "x": "y_b", "y": "response"},
        "sigma": {"Intercept": "sigma_a", "x": "sigma_b", "sigma": "s"},
    }
    model.set_alias(aliases)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "response_obs"]
    assert set(idata.posterior.data_vars) == {
        "y_a",
        "y_b",
        "sigma_a",
        "sigma_b",
        "response_s",
        "response_s",
        "response_mean",
    }
    assert list(idata.posterior["response_mean"].coords) == ["chain", "draw", "response_obs"]
    assert list(idata.posterior["response_s"].coords) == ["chain", "draw", "response_obs"]


def test_non_distributional_model_with_categories(anes):
    model = bmb.Model("vote[clinton] ~ age + age:party_id", anes, family="bernoulli")
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "age:party_id_dim", "vote_obs"]
    assert set(idata.posterior.data_vars) == {"Intercept", "age", "age:party_id", "vote_mean"}
    assert list(idata.posterior["vote_mean"].coords) == ["chain", "draw", "vote_obs"]
    assert list(idata.posterior["age:party_id"].coords) == ["chain", "draw", "age:party_id_dim"]
    assert set(idata.posterior["age:party_id_dim"].values) == {"independent", "republican"}

    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "y_obs"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "y_mean"}
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}
