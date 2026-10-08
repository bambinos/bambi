"""Per-Prior and per-component non-centered parameterization."""

import numpy as np
import pytest
import pytensor
import xarray as xr
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


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("location", ["omitted", "zero", "fixed", "free"])
def test_normal_location_affine_graph(data_random_n100, monkeypatch, sparse_dot, location):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    args = {"sigma": bmb.Prior("HalfNormal", sigma=1)}
    if location != "omitted":
        args["mu"] = {
            "zero": 0,
            "fixed": 2,
            "free": bmb.Prior("Normal", mu=2, sigma=0.5),
        }[location]
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": bmb.Prior("Normal", **args)},
    )
    model.build()
    graph = model.backend.model
    coefficient = graph["1|binary_cat"]
    inputs = [graph["1|binary_cat_offset"], graph["1|binary_cat_sigma"]]
    values = [np.array([-1.0, 2.0]), np.array(0.5)]
    mu = 0 if location in ("omitted", "zero") else 2
    if location == "free":
        inputs.append(graph["1|binary_cat_mu"])
        values.append(np.array(3.0))
        mu = 3
        assert inputs[-1] in set(ancestors(graph.observed_RVs))
    result = pytensor.function(inputs, coefficient)(*values)
    np.testing.assert_allclose(result, mu + values[0] * values[1])


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize(
    "location, prior_nc, model_nc",
    [
        ("omitted", None, True),
        ("fixed", None, True),
        ("vector", None, True),
        ("free", None, True),
        ("nested", None, True),
        ("nested", False, True),
        ("nested", True, {"mu": False}),
        ("fixed", True, False),
        ("free", False, True),
        ("nested-scale", None, True),
    ],
)
def test_normal_location_omitted_offsets(
    data_random_n100, mock_pymc_sample, monkeypatch, sparse_dot, location, prior_nc, model_nc
):
    """Natural draws must recover the same offsets, predictions, and prior densities."""
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    args = {"sigma": bmb.Prior("HalfNormal", sigma=1)}
    if location != "omitted":
        args["mu"] = np.array([1.0, 3.0]) if location == "vector" else 2.0
    if location in ("free", "nested"):
        args["mu"] = bmb.Prior("Normal", mu=2, sigma=0.5)
    if location == "nested":
        args["mu"] = bmb.Prior(
            "Normal", mu=args["mu"], sigma=bmb.Prior("HalfNormal", sigma=0.5), noncentered=True
        )
    if location == "nested-scale":
        args["sigma"] = bmb.Prior(
            "Normal", mu=3, sigma=bmb.Prior("HalfNormal", sigma=0.01), noncentered=True
        )
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": bmb.Prior("Normal", **args, noncentered=prior_nc)},
        noncentered=model_nc,
    )
    model.set_alias({"1|binary_cat": "effect", "mu": "location"})
    retained = model.fit(draws=4, chains=2, random_seed=12, omit_offsets=False)
    offset_names = [name for name in retained.posterior if name.endswith("_offset")]
    omitted = retained.copy(deep=True)
    omitted["posterior"] = omitted.posterior.to_dataset().drop_vars(offset_names)
    original = omitted.copy(deep=True)
    restored = model._re_center_intercept(omitted)
    for name in offset_names:
        xr.testing.assert_allclose(restored.posterior[name], retained.posterior[name])
    xr.testing.assert_identical(model._re_center_intercept(retained), retained)

    expected_prior = model.compute_log_prior(retained, inplace=False)
    actual_prior = model.compute_log_prior(omitted, inplace=False)
    kept_names = set(expected_prior.log_prior.data_vars) - set(offset_names)
    assert set(actual_prior.log_prior.data_vars) == kept_names
    for name in kept_names:
        xr.testing.assert_allclose(actual_prior.log_prior[name], expected_prior.log_prior[name])

    for new_data in (None, data_random_n100.iloc[[2, 0, 1]].copy()):
        expected = model.predict(
            retained, data=new_data, kind="response", inplace=False, random_seed=42
        )
        actual = model.predict(
            omitted, data=new_data, kind="response", inplace=False, random_seed=42
        )
        xr.testing.assert_allclose(actual.posterior["location"], expected.posterior["location"])
        xr.testing.assert_allclose(actual.posterior_predictive, expected.posterior_predictive)
        expected_ll = model.compute_log_likelihood(retained, data=new_data, inplace=False)
        actual_ll = model.compute_log_likelihood(omitted, data=new_data, inplace=False)
        xr.testing.assert_allclose(actual_ll.log_likelihood, expected_ll.log_likelihood)
    xr.testing.assert_identical(omitted, original)


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("family", ["gaussian", "categorical"])
@pytest.mark.parametrize("nested", [False, True])
def test_normal_location_offset_dimensions(
    data_random_n100, mock_pymc_sample, monkeypatch, family, sparse_dot, nested
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    response = "continuous1" if family == "gaussian" else "categorical2"
    response_size = data_random_n100[response].nunique() - 1
    # Cover expression and response axes separately: categorical slopes use a
    # different construction path from categorical group intercepts.
    term = "categorical1|binary_cat" if family == "gaussian" else "1|binary_cat"
    location = np.arange(4.0) if family == "gaussian" else np.arange(float(response_size))
    if nested:
        location = bmb.Prior("Normal", mu=2, sigma=bmb.Prior("HalfNormal", sigma=1))
    model = bmb.Model(
        f"{response} ~ 0 + (0 + {term})",
        data_random_n100,
        family=family,
        priors={term: bmb.Prior("Normal", mu=location, sigma=bmb.Prior("HalfNormal", sigma=1))},
    )
    retained = model.fit(draws=3, chains=2, random_seed=12, omit_offsets=False)
    names = [name for name in retained.posterior if name.endswith("_offset")]
    omitted = retained.copy(deep=True)
    omitted["posterior"] = (
        omitted.posterior.to_dataset().drop_vars(names).transpose(..., "draw", "chain")
    )
    restored = model._re_center_intercept(omitted)
    for name in names:
        xr.testing.assert_allclose(
            restored.posterior[name].transpose(*retained.posterior[name].dims),
            retained.posterior[name],
        )


@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("omit_offsets", [False, True])
def test_normal_location_new_group_prediction(
    data_random_n100, mock_pymc_sample, monkeypatch, sparse_dot, omit_offsets
):
    """Main's donor-based new-group predictions use the stored natural coefficients."""
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={
            "1|binary_cat": bmb.Prior(
                "Normal",
                mu=bmb.Prior("Normal", mu=2, sigma=1),
                sigma=bmb.Prior("HalfNormal", sigma=1),
            )
        },
    )
    trace = model.fit(draws=4, chains=2, random_seed=12, omit_offsets=omit_offsets)
    shifted = trace.copy(deep=True)
    for name in ("1|binary_cat", "1|binary_cat_mu"):
        shifted["posterior"][name] = shifted.posterior[name] + 3
    new_data = data_random_n100.iloc[:2].copy()
    new_data["binary_cat"] = ["a", "new"]
    with pytest.warns(FutureWarning, match="sample_new_groups"):
        expected = model.predict(
            trace, data=new_data, sample_new_groups=True, random_seed=42, inplace=False
        )
    with pytest.warns(FutureWarning, match="sample_new_groups"):
        actual = model.predict(
            shifted, data=new_data, sample_new_groups=True, random_seed=42, inplace=False
        )
    xr.testing.assert_allclose(actual.posterior["mu"], expected.posterior["mu"] + 3)


def test_nested_location_log_prior_matches_normal_density(data_random_n100, mock_pymc_sample):
    """An omitted location offset is an input, not an additional density target."""
    model = bmb.Model(
        "continuous1 ~ 0 + (1|binary_cat)",
        data_random_n100,
        priors={
            "1|binary_cat": bmb.Prior(
                "Normal",
                mu=bmb.Prior(
                    "Normal", mu=2, sigma=bmb.Prior("HalfNormal", sigma=1), noncentered=True
                ),
                sigma=bmb.Prior("HalfNormal", sigma=1),
                noncentered=False,
            )
        },
    )
    trace = model.fit(draws=4, chains=2, random_seed=12)
    assert "1|binary_cat_mu_offset" not in trace.posterior
    result = model.compute_log_prior(trace, inplace=False)
    b, mu, sigma = (
        trace.posterior[name] for name in ("1|binary_cat", "1|binary_cat_mu", "1|binary_cat_sigma")
    )
    expected = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((b - mu) / sigma) ** 2
    xr.testing.assert_allclose(result.log_prior["1|binary_cat"], expected)
    assert "1|binary_cat_mu_offset" not in result.log_prior
    assert "1|binary_cat_mu" not in result.log_prior


def test_normal_offset_subset_and_rebuild(data_random_n100, mock_pymc_sample):
    model = bmb.Model(
        "continuous1 ~ 1 + (1|binary_cat)",
        data_random_n100,
        priors={"1|binary_cat": bmb.Prior("Normal", mu=2, sigma=bmb.Prior("HalfNormal", sigma=1))},
    )
    trace = model.fit(draws=3, chains=2, random_seed=12)
    expected = model.compute_log_prior(trace, inplace=False)
    trace["posterior"] = trace.posterior.to_dataset()[["Intercept"]]
    result = model.compute_log_prior(trace, inplace=False)
    assert set(result.log_prior.data_vars) == {"Intercept"}
    xr.testing.assert_allclose(result.log_prior["Intercept"], expected.log_prior["Intercept"])
    model.set_priors(
        {
            "1|binary_cat": bmb.Prior(
                "Normal", mu=3, sigma=bmb.Prior("HalfNormal", sigma=1), noncentered=False
            )
        }
    )
    trace = model.fit(draws=3, chains=2, random_seed=12)
    assert model._re_center_intercept(trace) is trace
    assert not any(rv.name.endswith("_offset") for rv in model.backend.model.free_RVs)
