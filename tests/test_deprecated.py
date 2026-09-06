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


def test_internal_model_operations_do_not_warn(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repr(model)
        model.set_alias({"x": "slope"})
        model.build()

    assert [warning for warning in caught if warning.category is FutureWarning] == []
