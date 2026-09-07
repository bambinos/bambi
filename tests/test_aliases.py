import pytest

import bambi as bmb


def test_non_distributional_model(data_random_n100, mock_pymc_sample):
    # Plain model
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert {"Intercept", "Intercept_centered", "continuous2", "mu", "sigma"} == set(
        idata.posterior.data_vars
    )
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]

    # Model with alises
    model.set_alias({"Intercept": "a", "continuous2": "b", "sigma": "s", "continuous1": "response"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert {"a", "b", "mu", "s"}.issubset(idata.posterior.data_vars)
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]


def test_distributional_model(data_random_n100, mock_pymc_sample):
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")
    model = bmb.Model(formula, data_random_n100)
    idata = model.fit(chains=2)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "__obs__"]
    assert {
        "Intercept",
        "Intercept_centered",
        "continuous2",
        "sigma_Intercept",
        "sigma_Intercept_centered",
        "sigma_continuous2",
        "sigma",
        "mu",
    } == set(idata.posterior.data_vars)
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
    assert {
        "mu",
        "mu_a",
        "mu_b",
        "sigma_a",
        "sigma_b",
        "s",
    }.issubset(idata.posterior.data_vars)
    assert list(idata.posterior["mu"].coords) == ["chain", "draw", "__obs__"]
    assert list(idata.posterior["s"].coords) == ["chain", "draw", "__obs__"]


def test_non_distributional_model_with_categories(data_anes, mock_pymc_sample):
    model = bmb.Model("vote[clinton] ~ age + age:party_id", data_anes, family="bernoulli")
    idata = model.fit(chains=2)
    model.predict(idata)
    assert set(idata.posterior.data_vars) == {
        "Intercept",
        "Intercept_centered",
        "age",
        "age:party_id",
        "p",
    }
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    interaction_dim = idata.posterior["age:party_id"].dims[-1]
    assert set(idata.posterior.coords[interaction_dim].values) == {"independent", "republican"}

    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert set(idata.posterior.data_vars) == {"α", "α_centered", "β", "γ", "p"}
    assert list(idata.posterior["p"].coords) == ["chain", "draw", "__obs__"]
    interaction_dim = idata.posterior["γ"].dims[-1]
    assert set(idata.posterior.coords[interaction_dim].values) == {"independent", "republican"}

    # Same as before, but also put an alias for 'p'
    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y", "p": "mean"})
    idata = model.fit(chains=2)
    model.predict(idata)
    assert set(idata.posterior.data_vars) == {"α", "α_centered", "β", "γ", "mean"}
    assert list(idata.posterior["mean"].coords) == ["chain", "draw", "__obs__"]
    interaction_dim = idata.posterior["γ"].dims[-1]
    assert set(idata.posterior.coords[interaction_dim].values) == {"independent", "republican"}


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
    new_names = {"α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"}
    assert new_names.issubset(set(model.backend.model.named_vars))


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize(
    "formula,aliases,expected_names",
    [
        (
            "continuous1 ~ (1|categorical1)",
            {"sigma": "sd"},
            {"1|categorical1_sd"},
        ),
        (
            "continuous1 ~ continuous2 + (1 + continuous2|categorical1)",
            {"sigma": "group_sd"},
            {"1|categorical1_group_sd", "continuous2|categorical1_group_sd"},
        ),
        (
            "continuous1 ~ (1|categorical1) + (1|categorical2)",
            {
                "1|categorical1": "by_categorical1",
                "1|categorical2": "by_categorical2",
                "sigma": "sd",
            },
            {"by_categorical1_sd", "by_categorical2_sd"},
        ),
    ],
)
def test_group_specific_hyperprior_aliases(
    data_random_n100, monkeypatch, sparse_dot, formula, aliases, expected_names
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model(formula, data_random_n100)
    model.set_alias(aliases)
    model.build()

    assert expected_names.issubset(model.backend.model.named_vars)


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.usefixtures("mock_pymc_sample")
def test_predict_new_groups_with_hyperprior_alias(data_random_n100, monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    formula = "continuous1 ~ continuous2 + (1 + continuous2|categorical1)"
    model = bmb.Model(formula, data_random_n100)
    model.set_alias({"sigma": "group_sd"})
    idata = model.fit(draws=20, chains=2)
    new_data = data_random_n100.head(10).assign(categorical1="new_group")

    result = model.predict(idata, data=new_data, random_seed=42, inplace=False)

    assert result.predictions["mu"].shape == (2, 20, len(new_data))

    result = model.compute_log_likelihood(idata, inplace=False)

    assert result.log_likelihood["continuous1"].shape == (2, 20, len(data_random_n100))
