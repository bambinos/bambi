import warnings
from pathlib import Path

import pandas as pd
import pytest

import bambi as bmb


@pytest.fixture
def model():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    return bmb.Model("y ~ x", data)


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
