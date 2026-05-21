"""Per-Prior and per-component non-centered parameterization."""

import pytest

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
    with pytest.raises(ValueError, match=r"Unknown component name\(s\) in `noncentered`"):
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
