import pytest

import bambi as bmb
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def my_data():
    return bmb.load_data("my_data")


@pytest.fixture(scope="module")
def anes():
    return bmb.load_data("ANES")


@pytest.fixture(scope="module")
def data_100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "n1": rng.normal(size=size),
            "n2": rng.normal(size=size),
            "n3": rng.normal(size=size),
            "b0": rng.binomial(n=1, p=0.5, size=size),
            "b1": rng.choice(["a", "b"], size=size),
            "count1": rng.poisson(lam=2, size=size),
            "count2": rng.poisson(lam=2, size=size),
            "cat1": rng.choice(list("MNOP"), size=size),
            "cat2": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


def test_non_distributional_model(my_data):
    # Plain model
    formula = bmb.Formula("y ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {"Intercept", "x", "mu", "sigma"}
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]

    # Model with alises
    model.set_alias({"Intercept": "a", "x": "b", "sigma": "s", "y": "response"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {"a", "b", "mu", "s"}
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]


def test_distributional_model(my_data):
    formula = bmb.Formula("y ~ x", "sigma ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {
        "Intercept",
        "x",
        "sigma_Intercept",
        "sigma_x",
        "sigma",
        "mu",
    }
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["sigma"].coords) == ["chain", "draw", "__obs__"]

    aliases = {
        "y": "response",
        "mu": {"Intercept": "mu_a", "x": "mu_b"},
        "sigma": {"Intercept": "sigma_a", "x": "sigma_b", "sigma": "s"},
    }
    model.set_alias(aliases)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {
        "mu",
        "mu_a",
        "mu_b",
        "sigma_a",
        "sigma_b",
        "s",
    }
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["s"].coords) == ["chain", "draw", "__obs__"]


def test_non_distributional_model_with_categories(anes):
    model = bmb.Model("vote[clinton] ~ age + age:party_id", anes, family="bernoulli")
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "age:party_id_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"Intercept", "age", "age:party_id", "p"}
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["age:party_id"].coords) == ["chain", "draw", "age:party_id_dim"]
    assert set(idata.posterior["age:party_id_dim"].values) == {"independent", "republican"}

    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "p"}
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}

    # Same as before, but also put an alias for 'p'
    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y", "p": "mean"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "mean"}
    assert list(idata.posterior["mean"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}


def test_alias_equal_to_name(my_data):
    model = bmb.Model("y ~ 1 + x", my_data)
    model.set_alias({"sigma": "sigma"})
    idata = model.fit(tune=100, draws=100)
    set(idata.posterior.data_vars) == {"Intercept", "x", "sigma"}


def test_set_alias_warnings(my_data):
    # Create a model to use aliases on
    formula = bmb.Formula("y ~ x")
    model = bmb.Model(formula, my_data)

    # Define cases that throw the various warnings
    test_cases = [
        # Only one unused alias, explicitly tell user the name
        (
            {"unused_alias": "ua"},
            "The following names do not match any terms, "
            "their aliases were not assigned: unused_alias",
        ),
        # Many unused aliases, generic response
        (
            {f"unused_alias{i}": f"ua{i}" for i in range(6)},
            "There are 6 names that do not match any terms, so their aliases were not assigned.",
        ),
    ]

    # Evaluate each case
    for alias_dict, expected_warning in test_cases:
        with pytest.warns(UserWarning) as record:
            model.set_alias(alias_dict)
            print(model.constant_components)
        assert len(record) == 1
        assert str(record[0].message) == expected_warning


# FIXME: Move somewhere
def test_set_alias(data_100):
    model = bmb.Model("n1 ~ n2 + (n2|cat1)", data_100)
    aliases = {
        "Intercept": "α",
        "n2": "β",
        "1|cat1": "α_group",
        "n2|cat1": "β_group",
        "sigma": "σ",
    }
    model.set_alias(aliases)
    model.build()
    new_names = set(["α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"])
    assert new_names.issubset(set(model.backend.model.named_vars))
