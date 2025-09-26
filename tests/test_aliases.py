import pytest

import bambi as bmb


def test_non_distributional_model(data_random_n100, mock_pymc_sample):
    # Plain model
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {"Intercept", "continuous2", "mu", "sigma"}
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]

    # Model with alises
    model.set_alias({"Intercept": "a", "continuous2": "b", "sigma": "s", "continuous1": "response"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {"a", "b", "mu", "s"}
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]


def test_distributional_model(data_random_n100, mock_pymc_sample):
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")
    model = bmb.Model(formula, data_random_n100)
    idata = model.fit(chains=2)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert set(idata.posterior.data_vars) == {
        "Intercept",
        "continuous2",
        "sigma_Intercept",
        "sigma_continuous2",
        "sigma",
        "mu",
    }
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["sigma"].coords) == ["chain", "draw", "__obs__"]

    aliases = {
        "continuous1": "response",
        "mu": {"Intercept": "mu_a", "continuous2": "mu_b"},
        "sigma": {"Intercept": "sigma_a", "continuous2": "sigma_b", "sigma": "s"},
    }
    model.set_alias(aliases)
    idata = model.fit(chains=2)
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


def test_non_distributional_model_with_categories(data_anes, mock_pymc_sample):
    model = bmb.Model("vote[clinton] ~ age + age:party_id", data_anes, family="bernoulli")
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "age:party_id_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"Intercept", "age", "age:party_id", "p"}
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["age:party_id"].coords) == ["chain", "draw", "age:party_id_dim"]
    assert set(idata.posterior["age:party_id_dim"].values) == {"independent", "republican"}

    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "p"}
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}

    # Same as before, but also put an alias for 'p'
    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y", "p": "mean"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "__obs__"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "mean"}
    assert list(idata.posterior["mean"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}


def test_alias_equal_to_name(data_random_n100, mock_pymc_sample):
    model = bmb.Model("continuous1 ~ 1 + continuous2", data_random_n100)
    model.set_alias({"sigma": "sigma"})
    idata = model.fit(chains=2)
    set(idata.posterior.data_vars) == {"Intercept", "continuous2", "sigma"}


def test_set_alias_warnings(data_random_n100, mock_pymc_sample):
    # Create a model to use aliases on
    formula = bmb.Formula("continuous1 ~ continuous2")
    model = bmb.Model(formula, data_random_n100)

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


def test_set_alias(data_random_n100, mock_pymc_sample):
    model = bmb.Model("continuous1 ~ continuous2 + (continuous2|categorical1)", data_random_n100)
    aliases = {
        "Intercept": "α",
        "continuous2": "β",
        "1|categorical1": "α_group",
        "continuous2|categorical1": "β_group",
        "sigma": "σ",
    }
    model.set_alias(aliases)
    model.build()
    new_names = set(["α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"])
    assert new_names.issubset(set(model.backend.model.named_vars))
