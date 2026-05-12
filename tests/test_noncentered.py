"""Tests for per-Prior non-centered parameterization control.

Covers the tri-state semantics of ``Prior.noncentered`` (``None`` inherits the model
default, ``True``/``False`` overrides) and the relaxed centered fallback that now
accepts non-Normal priors when noncentering is not requested.
"""

import pytest

import bambi as bmb


def _hyper_normal(**kwargs):
    """Helper: Normal prior with a random HalfNormal sigma hyperprior."""
    return bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1), **kwargs)


def _named_vars(model):
    return set(model.backend.model.named_vars)


def test_per_prior_true_beats_model_false(data_random_n100):
    """Prior(noncentered=True) overrides Model(noncentered=False)."""
    priors = {"continuous2|binary_cat": _hyper_normal(noncentered=True)}
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
        noncentered=False,
    )
    model.build()
    named_vars = _named_vars(model)
    # The explicitly noncentered slope-by-group term should produce an _offset RV.
    assert "continuous2|binary_cat_offset" in named_vars
    # The intercept-by-group term inherits the model default (False) → no offset.
    assert "1|binary_cat_offset" not in named_vars


def test_per_prior_false_beats_model_true(data_random_n100):
    """Prior(noncentered=False) overrides Model(noncentered=True) (the default)."""
    priors = {"continuous2|binary_cat": _hyper_normal(noncentered=False)}
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
    )
    model.build()
    named_vars = _named_vars(model)
    # The explicitly centered term should NOT produce an _offset RV.
    assert "continuous2|binary_cat_offset" not in named_vars
    # The intercept-by-group term inherits the model default (True) → offset exists.
    assert "1|binary_cat_offset" in named_vars


def test_none_inherits_model_default_true(data_random_n100):
    """Prior(noncentered=None) (the default) inherits Model.noncentered=True."""
    # No explicit per-prior setting → both group-specific terms should be noncentered.
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        noncentered=True,
    )
    model.build()
    named_vars = _named_vars(model)
    assert "continuous2|binary_cat_offset" in named_vars
    assert "1|binary_cat_offset" in named_vars


def test_none_inherits_model_default_false(data_random_n100):
    """Prior(noncentered=None) inherits Model.noncentered=False."""
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        noncentered=False,
    )
    model.build()
    named_vars = _named_vars(model)
    assert "continuous2|binary_cat_offset" not in named_vars
    assert "1|binary_cat_offset" not in named_vars


def test_mixed_noncentering_two_grouping_terms(data_random_n100):
    """Mixed per-prior settings across two group-specific terms in one component."""
    priors = {
        "1|binary_cat": _hyper_normal(noncentered=True),
        "continuous2|binary_cat": _hyper_normal(noncentered=False),
    }
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
        # Set Model.noncentered to an opposite-of-each value to confirm both per-prior
        # decisions win independently of the model default.
        noncentered=False,
    )
    model.build()
    named_vars = _named_vars(model)
    offsets = {v for v in named_vars if v.endswith("_offset")}
    assert offsets == {"1|binary_cat_offset"}


def test_mixed_noncentering_across_distributional_components(data_random_n100):
    """HSSM-style: per-prior noncentering on group-specific terms in different
    distributional components (parent + auxiliary).
    """
    formula = bmb.Formula(
        "continuous1 ~ 1 + (1|binary_cat)",
        "sigma ~ 1 + (1|binary_cat)",
    )
    priors = {
        # Parent component's grouping term: explicit noncentered=True.
        "1|binary_cat": _hyper_normal(noncentered=True),
        # Auxiliary `sigma` component's grouping term: explicit noncentered=False.
        "sigma": {"1|binary_cat": _hyper_normal(noncentered=False)},
    }
    model = bmb.Model(formula, data_random_n100, priors=priors)
    model.build()
    named_vars = _named_vars(model)
    offsets = {v for v in named_vars if v.endswith("_offset")}
    # Exactly one offset, for the parent component's grouping term.
    assert offsets == {"1|binary_cat_offset"}


def test_non_normal_prior_with_noncentered_false_builds(data_random_n100):
    """Non-Normal priors with random hyperpriors must build under explicit
    noncentered=False (previously raised NotImplementedError).
    """
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
    model.build()  # Should NOT raise.
    named_vars = _named_vars(model)
    assert "continuous2|binary_cat_offset" not in named_vars
    assert "continuous2|binary_cat" in named_vars


def test_non_normal_prior_with_noncentered_true_raises(data_random_n100):
    """Non-Normal priors with noncentered=True still raise, with an informative message."""
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
    with pytest.raises(NotImplementedError) as excinfo:
        model.build()
    msg = str(excinfo.value)
    # The improved error message names both the requested prior and the constraint.
    assert "StudentT" in msg
    assert "Normal" in msg
    assert "noncentered=False" in msg


def test_predict_and_omit_offsets_with_mixed_noncentering(
    data_random_n100, mock_pymc_sample
):
    """End-to-end regression: a mixed-noncentering model must fit, predict, and
    its omit_offsets filter must behave correctly on the heterogeneous posterior.
    """
    priors = {
        "1|binary_cat": _hyper_normal(noncentered=True),
        "continuous2|binary_cat": _hyper_normal(noncentered=False),
    }
    model = bmb.Model(
        "continuous1 ~ continuous2 + (continuous2|binary_cat)",
        data_random_n100,
        priors=priors,
    )

    # omit_offsets=False: only the noncentered term contributes an offset variable.
    idata_keep = model.fit(chains=2, omit_offsets=False)
    offsets = {v for v in idata_keep.posterior.data_vars if v.endswith("_offset")}
    assert offsets == {"1|binary_cat_offset"}

    # omit_offsets=True (default): no offsets at all.
    idata_drop = model.fit(chains=2, omit_offsets=True)
    drop_offsets = {v for v in idata_drop.posterior.data_vars if v.endswith("_offset")}
    assert drop_offsets == set()

    # predict paths run cleanly on the mixed-noncentering posterior.
    model.predict(idata_drop, kind="response")
    model.predict(idata_drop, kind="response_params")
