import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bambi as bmb
from bambi.interpret import (
    comparisons,
    plot_comparisons,
    plot_predictions,
    plot_slopes,
    predictions,
    slopes,
)
from bambi.transformations import counts


@pytest.fixture
def model():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "z": [0.0, 0.0, 1.0]})
    return bmb.Model("y ~ x + z", data)


@pytest.mark.parametrize(
    ("deprecated_name", "replacement_name"),
    [
        ("components", "parameters"),
        ("distributional_components", "conditional_parameters"),
        ("constant_components", "marginal_parameters"),
    ],
)
def test_model_component_names_deprecated(model, deprecated_name, replacement_name):
    replacement = getattr(model, replacement_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated = getattr(model, deprecated_name)

    assert deprecated == replacement
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert f"Model.{deprecated_name}" in str(caught[0].message)
    assert f"Model.{replacement_name}" in str(caught[0].message)
    assert "future version" in str(caught[0].message)
    assert Path(caught[0].filename) == Path(__file__)


@pytest.mark.parametrize("name", ["parameters", "conditional_parameters", "marginal_parameters"])
def test_new_model_component_names_do_not_warn(model, name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(model, name)

    assert caught == []


def test_parameters_is_stored_components_dictionary(model):
    assert model.parameters is model.parameters


def test_internal_model_operations_do_not_warn(model, mock_pymc_sample):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repr(model)
        model.set_alias({"x": "slope"})
        model.build()
        idata = model.fit(chains=2)
        model.predict(idata)

    assert [warning for warning in caught if warning.category is FutureWarning] == []


def test_response_component_deprecated(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response_component = model.response_component

    assert response_component.term is model.response_term
    assert response_component.response is model._response_component.response
    assert response_component.spec is model
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert "Model.response_component" in str(caught[0].message)
    assert "Model.response_term" in str(caught[0].message)
    assert "future version" in str(caught[0].message)
    assert Path(caught[0].filename) == Path(__file__)


def test_response_term_does_not_warn(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response_term = model.response_term

    assert response_term is model._response_component.term
    assert caught == []


@pytest.mark.parametrize(("value", "normalized"), [(None, False), (False, False), (True, True)])
def test_predict_sample_new_groups_deprecated(model, mocker, value, normalized):
    compute = mocker.patch.object(model, "_compute_likelihood_params", return_value=object())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(object(), sample_new_groups=value)

    assert compute.call_args.kwargs["sample_new_groups"] is normalized
    sample_new_groups_warnings = [
        warning for warning in caught if "sample_new_groups" in str(warning.message)
    ]
    assert len(sample_new_groups_warnings) == (value is not None)
    if sample_new_groups_warnings:
        warning = sample_new_groups_warnings[0]
        assert warning.category is FutureWarning
        assert "handled automatically" in str(warning.message)
        assert "future version" in str(warning.message)
        assert Path(warning.filename) == Path(__file__)


@pytest.fixture
def fitted_model(model, mock_pymc_sample):
    return model, model.fit(chains=2)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (predictions, {"conditional": "x"}),
        (plot_predictions, {"conditional": "x"}),
        (comparisons, {"contrast": "x"}),
        (plot_comparisons, {"contrast": "x", "conditional": "z"}),
        (slopes, {"wrt": "x"}),
        (plot_slopes, {"wrt": "x", "conditional": "z"}),
    ],
)
@pytest.mark.parametrize("value", [None, False, True])
def test_interpret_sample_new_groups_deprecated(fitted_model, function, kwargs, value):
    model, idata = fitted_model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function(model, idata, sample_new_groups=value, **kwargs)

    sample_new_groups_warnings = [
        warning for warning in caught if "sample_new_groups" in str(warning.message)
    ]
    assert len(sample_new_groups_warnings) == (value is not None)
    if sample_new_groups_warnings:
        warning = sample_new_groups_warnings[0]
        assert warning.category is FutureWarning


@pytest.fixture
def count_data():
    return pd.DataFrame(
        {
            "y1": [1, 2, 3],
            "y2": [3, 4, 3],
            "x1": [0.0, 1.0, 2.0],
            "x2": [2.0, 1.0, 0.0],
        }
    )


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_c_response_deprecated_for_count_families(count_data, family):
    formula = bmb.Formula("c(y1, y2) ~ x1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = bmb.Model(formula, count_data, family=family)
        model.build()

    future_warnings = [warning for warning in caught if warning.category is FutureWarning]
    assert len(future_warnings) == 1
    assert "c(...)" in str(future_warnings[0].message)
    assert "counts(...)" in str(future_warnings[0].message)
    assert Path(future_warnings[0].filename) == Path(__file__)
    assert model.formula.main == "counts(y1, y2) ~ x1"
    assert model.response_term.name == "counts(y1, y2)"
    assert model.response_term.levels == ["y1", "y2"]
    assert np.array_equal(model.response_term.data, count_data[["y1", "y2"]].to_numpy())


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_counts_response_and_c_predictor_do_not_warn(count_data, family):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = bmb.Model("counts(y1, y2) ~ c(x1, x2)", count_data, family=family)
        model.build()

    assert [warning for warning in caught if warning.category is FutureWarning] == []
    assert model.response_term.name == "counts(y1, y2)"
    assert model.response_term.levels == ["y1", "y2"]
    parameter = model.conditional_parameters[model.family.likelihood.parent]
    assert "c(x1, x2)" in parameter.common_terms


def test_counts_stacks_count_columns():
    y1 = np.array([1, 2])
    y2 = np.array([3, 2])
    expected = np.array([[1, 3], [2, 2]])

    assert np.array_equal(counts(y1, y2), expected)


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_counts_response_fit_and_predict(count_data, mock_pymc_sample, family):
    model = bmb.Model("counts(y1, y2) ~ x1", count_data, family=family)
    idata = model.fit(chains=2)
    predictions = model.predict(idata, data=count_data, kind="response", inplace=False)

    assert model.response_term.name == "counts(y1, y2)"
    assert model.response_term.levels == ["y1", "y2"]
    assert np.array_equal(model.response_term.data, count_data[["y1", "y2"]].to_numpy())
    assert "counts(y1, y2)" in predictions.posterior_predictive


@pytest.mark.parametrize("kind", ["mean", "pps"])
def test_removed_predict_kinds(model, kind):
    with pytest.raises(ValueError, match="'kind' must be one of 'response_params' or 'response'"):
        model.predict(object(), kind=kind)
