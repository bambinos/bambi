"""Per-Prior and per-component non-centered parameterization."""

import numpy as np
import pymc as pm
import pytest
import pytensor

from pytensor.graph.traversal import ancestors

import bambi as bmb


def _hyper_normal(**kwargs):
    return bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1), **kwargs)


def _named_vars(model):
    return set(model.backend.model.named_vars)


def _offsets(model):
    return {v for v in _named_vars(model) if v.endswith("_offset")}


def test_per_prior_true_beats_model_false(data_random_n100):
    priors = {"continuous2|binary_cat": _hyper_normal(noncentered=True)}
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
        noncentered=False,
    )
    model.build()
    assert "continuous2|binary_cat_offset" in _named_vars(model)
    assert "1|binary_cat_offset" not in _named_vars(model)


def test_per_prior_false_beats_model_true(data_random_n100):
    priors = {"continuous2|binary_cat": _hyper_normal(noncentered=False)}
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
        noncentered=True,
    )
    model.build()
    assert "continuous2|binary_cat_offset" not in _named_vars(model)
    assert "1|binary_cat_offset" in _named_vars(model)


def test_none_inherits_model_default_true(data_random_n100):
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        noncentered=True,
    )
    model.build()
    assert "continuous2|binary_cat_offset" in _named_vars(model)
    assert "1|binary_cat_offset" in _named_vars(model)


def test_none_inherits_model_default_false(data_random_n100):
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        noncentered=False,
    )
    model.build()
    assert "continuous2|binary_cat_offset" not in _named_vars(model)
    assert "1|binary_cat_offset" not in _named_vars(model)


def test_mixed_noncentering_two_grouping_terms(data_random_n100):
    priors = {
        "1|binary_cat": _hyper_normal(noncentered=True),
        "continuous2|binary_cat": _hyper_normal(noncentered=False),
    }
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
        noncentered=False,
    )
    model.build()
    assert _offsets(model) == {"1|binary_cat_offset"}


def test_mixed_noncentering_across_distributional_components(data_random_n100):
    formula = bmb.Formula(
        "continuous1 ~ 1 + (1|binary_cat)",
        "sigma ~ 1 + (1|binary_cat)",
    )
    priors = {
        "1|binary_cat": _hyper_normal(noncentered=True),
        "sigma": {"1|binary_cat": _hyper_normal(noncentered=False)},
    }
    model = bmb.Model(formula, data_random_n100, priors=priors)
    model.build()
    assert _offsets(model) == {"1|binary_cat_offset"}


def test_component_dict_sets_per_parameter_default(data_random_n100):
    formula = bmb.Formula(
        "continuous1 ~ 1 + (1|binary_cat)",
        "sigma ~ 1 + (1|binary_cat)",
    )
    model = bmb.Model(
        formula,
        data_random_n100,
        noncentered={"mu": True, "sigma": False},
    )
    model.build()
    assert _offsets(model) == {"1|binary_cat_offset"}


def test_component_dict_missing_key_defaults_to_true(data_random_n100):
    formula = bmb.Formula(
        "continuous1 ~ 1 + (1|binary_cat)",
        "sigma ~ 1 + (1|binary_cat)",
    )
    model = bmb.Model(formula, data_random_n100, noncentered={"sigma": False})
    model.build()
    assert _offsets(model) == {"1|binary_cat_offset"}


def test_per_prior_still_overrides_component_dict(data_random_n100):
    formula = bmb.Formula(
        "continuous1 ~ 1 + (1|binary_cat)",
        "sigma ~ 1 + (1|binary_cat)",
    )
    priors = {"1|binary_cat": _hyper_normal(noncentered=False)}
    model = bmb.Model(
        formula,
        data_random_n100,
        priors=priors,
        noncentered={"mu": True, "sigma": True},
    )
    model.build()
    assert _offsets(model) == {"sigma_1|binary_cat_offset"}


def test_component_dict_rejects_unknown_keys(data_random_n100):
    with pytest.raises(ValueError, match=r"Unknown parameter name\(s\) in `noncentered`"):
        bmb.Model(
            "continuous1 ~ 1 + (1|binary_cat)",
            data_random_n100,
            noncentered={"vv": True},
        )


def test_non_normal_prior_with_noncentered_false_builds(data_random_n100):
    prior = bmb.Prior(
        "StudentT",
        nu=4,
        mu=0,
        sigma=bmb.Prior("HalfNormal", sigma=1),
        noncentered=False,
    )
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors={"continuous2|binary_cat": prior},
    )
    model.build()
    assert "continuous2|binary_cat_offset" not in _named_vars(model)
    assert "continuous2|binary_cat" in _named_vars(model)


def test_non_normal_prior_with_noncentered_true_raises(data_random_n100):
    prior = bmb.Prior(
        "StudentT",
        nu=4,
        mu=0,
        sigma=bmb.Prior("HalfNormal", sigma=1),
        noncentered=True,
    )
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors={"continuous2|binary_cat": prior},
    )
    with pytest.raises(
        NotImplementedError,
        match=r"non-centered parametrization is only supported for Normal priors, got StudentT",
    ):
        model.build()


def test_predict_and_omit_offsets_with_mixed_noncentering(data_random_n100, mock_pymc_sample):
    priors = {
        "1|binary_cat": _hyper_normal(noncentered=True),
        "continuous2|binary_cat": _hyper_normal(noncentered=False),
    }
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
    )

    idata_keep = model.fit(chains=2, omit_offsets=False)
    keep_offsets = {v for v in idata_keep.posterior.data_vars if v.endswith("_offset")}
    assert keep_offsets == {"1|binary_cat_offset"}

    idata_drop = model.fit(chains=2, omit_offsets=True)
    drop_offsets = {v for v in idata_drop.posterior.data_vars if v.endswith("_offset")}
    assert drop_offsets == set()

    model.predict(idata_drop, kind="response")
    model.predict(idata_drop, kind="response_params")


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("location", ["omitted", "zero", "fixed", "hyperprior"])
def test_noncentered_normal_location(data_random_n100, monkeypatch, sparse_dot, location):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    args = {"sigma": bmb.Prior("HalfNormal", sigma=1)}
    if location != "omitted":
        args["mu"] = {
            "zero": 0,
            "fixed": 2,
            "hyperprior": bmb.Prior("Normal", mu=2, sigma=0.5),
        }[location]
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": bmb.Prior("Normal", **args)},
    )
    model.build()
    pm_model = model.backend.model
    coefficient = pm_model["1|binary_cat"]
    offset = pm_model["1|binary_cat_offset"]
    sigma = pm_model["1|binary_cat_sigma"]
    inputs = [offset, sigma]
    values = [np.array([-1.0, 2.0]), np.array(0.5)]
    mu_value = 0 if location in ("omitted", "zero") else 2
    if location == "hyperprior":
        mu = pm_model["1|binary_cat_mu"]
        assert mu in set(ancestors([coefficient]))
        assert mu in set(ancestors(pm_model.observed_RVs))
        inputs.append(mu)
        mu_value = 3
        values.append(np.array(float(mu_value)))
    assert sigma in set(ancestors(pm_model.observed_RVs))

    evaluate = pytensor.function(inputs, [coefficient, pm_model["mu"]])
    coefficients, predictor = evaluate(*values)
    expected = mu_value + values[1] * values[0]
    group_index = data_random_n100["binary_cat"].map({"a": 0, "b": 1}).to_numpy()
    np.testing.assert_allclose(coefficients, expected)
    np.testing.assert_allclose(predictor, expected[group_index])
    np.testing.assert_allclose(
        pm.logp(offset, values[0]).eval(), -0.5 * (np.log(2 * np.pi) + values[0] ** 2)
    )
    if location == "hyperprior":
        values[-1] += 1
        shifted_coefficients, shifted_predictor = evaluate(*values)
        np.testing.assert_allclose(shifted_coefficients, coefficients + 1)
        np.testing.assert_allclose(shifted_predictor, predictor + 1)


@pytest.mark.parametrize("family", ["gaussian", "categorical"])
def test_noncentered_normal_location_broadcasting(data_random_n100, family):
    response = "continuous1" if family == "gaussian" else "categorical2"
    response_size = data_random_n100[response].nunique() - 1
    mu = (
        np.arange(4.0)
        if family == "gaussian"
        else np.arange(4.0 * response_size).reshape(4, response_size)
    )
    prior = bmb.Prior("Normal", mu=mu, sigma=bmb.Prior("HalfNormal", sigma=1))
    model = bmb.Model(
        f"{response} ~ 0 + (0 + categorical1|binary_cat)",
        data_random_n100,
        family=family,
        priors={"categorical1|binary_cat": prior},
    )
    model.build()
    pm_model = model.backend.model
    name = "categorical1|binary_cat"
    coefficient, offset, sigma = pm.draw(
        [pm_model[name], pm_model[name + "_offset"], pm_model[name + "_sigma"]], random_seed=12
    )
    assert coefficient.shape == (2, *mu.shape)
    np.testing.assert_allclose(coefficient, mu + offset * sigma)


def test_noncentered_normal_nested_location_and_alias(data_random_n100):
    location = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=2, sigma=0.5),
        sigma=bmb.Prior("HalfNormal", sigma=1),
        noncentered=True,
    )
    prior = bmb.Prior(
        "Normal", mu=location, sigma=bmb.Prior("HalfNormal", sigma=1), noncentered=True
    )
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": prior},
        noncentered=False,
    )
    model.set_alias({"1|binary_cat": "effect", "sigma": "scale"})
    model.build()
    pm_model = model.backend.model
    names = [
        "effect",
        "effect_offset",
        "effect_scale",
        "effect_mu",
        "effect_mu_offset",
        "effect_mu_sigma",
        "effect_mu_mu",
    ]
    coefficient, offset, sigma, mu, mu_offset, mu_sigma, mu_mu = pm.draw(
        [pm_model[name] for name in names], random_seed=12
    )
    np.testing.assert_allclose(mu, mu_mu + mu_offset * mu_sigma)
    np.testing.assert_allclose(coefficient, mu + offset * sigma)
    assert all(rv in set(ancestors(pm_model.observed_RVs)) for rv in pm_model.free_RVs)


def test_noncentered_normal_location_with_fixed_sigma_still_unsupported(data_random_n100):
    prior = bmb.Prior("Normal", mu=bmb.Prior("Normal", mu=2, sigma=1), sigma=1)
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": prior},
    )
    with pytest.raises(NotImplementedError, match="non-centered parametrization"):
        model.build()


@pytest.mark.parametrize("omit_offsets", [False, True])
def test_noncentered_normal_location_dense_prediction(
    data_random_n100, mock_pymc_sample, monkeypatch, omit_offsets
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", False)
    prior = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=2, sigma=1),
        sigma=bmb.Prior("HalfNormal", sigma=1),
    )
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": prior},
    )
    idata = model.fit(draws=10, chains=2, random_seed=12, omit_offsets=omit_offsets)
    new_data = data_random_n100.head(2).assign(binary_cat=["a", "new_group"])
    result = model.predict(idata, data=new_data, random_seed=42, inplace=False)
    assert result.predictions["mu"].shape == (2, 10, 2)
    np.testing.assert_allclose(
        result.predictions["mu"].values[..., 0], idata.posterior["1|binary_cat"].values[..., 0]
    )
    shifted = idata.copy(deep=True)
    shifted.posterior["1|binary_cat"] += 3
    shifted.posterior["1|binary_cat_mu"] += 3
    shifted_result = model.predict(shifted, data=new_data, random_seed=42, inplace=False)
    np.testing.assert_allclose(shifted_result.predictions["mu"], result.predictions["mu"] + 3)
