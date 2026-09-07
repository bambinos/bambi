import warnings

import pandas as pd
import pytest

import bambi as bmb


@pytest.mark.parametrize(
    ("deprecated_name", "replacement_name"),
    [
        ("components", "parameters"),
        ("distributional_components", "conditional_parameters"),
        ("constant_components", "marginal_parameters"),
    ],
)
def test_deprecated_component_properties(deprecated_name, replacement_name):
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = bmb.Model("y ~ x", data)

    with pytest.warns(FutureWarning, match=f"'{deprecated_name}'.*'{replacement_name}'") as record:
        value = getattr(model, deprecated_name)

    assert value == getattr(model, replacement_name)
    assert record[0].filename == __file__


def test_deprecated_response_component():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = bmb.Model("y ~ x", data)

    with pytest.warns(FutureWarning, match="'response_component'.*'response_term'") as record:
        response_component = model.response_component

    assert response_component.term is model.response_term
    assert response_component.response.term.term is model.response_term.term
    assert response_component.spec is model
    assert record[0].filename == __file__


@pytest.mark.parametrize("sample_new_groups", [None, True, False])
def test_predict_sample_new_groups_is_deprecated_and_has_no_effect(sample_new_groups, mocker):
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = bmb.Model("y ~ x", data)
    backend = mocker.Mock()
    mocker.patch.object(model, "backend", backend)
    idata = object()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.predict(idata, sample_new_groups=sample_new_groups)

    future_warnings = [warning for warning in record if warning.category is FutureWarning]
    assert len(future_warnings) == (sample_new_groups is not None)
    if future_warnings:
        assert "automatically" in str(future_warnings[0].message)
        assert future_warnings[0].filename == __file__
    assert "sample_new_groups" not in backend.predict.call_args.kwargs


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_c_response_is_deprecated_alias_for_counts(family):
    data = pd.DataFrame(
        {
            "y1": [1, 2, 3],
            "y2": [3, 4, 3],
            "x": [0.0, 1.0, 2.0],
        }
    )

    with pytest.warns(FutureWarning, match="Use 'counts") as record:
        model = bmb.Model(bmb.Formula("c(y1, y2) ~ x"), data, family=family)

    assert record[0].filename == __file__
    assert model.formula.main == "counts(y1, y2) ~ x"
    assert model.response_term.is_counts is True
    assert model.response_term.name == "y1_y2"


def test_c_predictor_is_not_deprecated_for_count_families():
    data = pd.DataFrame(
        {
            "y1": [1, 2, 3],
            "y2": [3, 4, 3],
            "x1": [0.0, 1.0, 2.0],
            "x2": [2.0, 1.0, 0.0],
        }
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = bmb.Model("counts(y1, y2) ~ c(x1, x2)", data, family="multinomial")

    assert not [warning for warning in record if warning.category is FutureWarning]
    assert "c(x1, x2)" in model.parameters["p"].common_terms


@pytest.mark.parametrize(
    ("function_name", "args"),
    [
        ("predictions", ()),
        ("plot_predictions", ()),
        ("comparisons", ("x",)),
        ("plot_comparisons", ("x",)),
        ("slopes", ("x",)),
        ("plot_slopes", ("x",)),
    ],
)
def test_interpret_sample_new_groups_is_deprecated_and_not_forwarded(function_name, args, mocker):
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = bmb.Model("y ~ x", data)
    predict = mocker.patch.object(
        model, "predict", side_effect=RuntimeError("stop after prediction call")
    )
    function = getattr(bmb.interpret, function_name)

    with pytest.warns(FutureWarning, match="sample_new_groups.*automatically") as record:
        with pytest.raises(RuntimeError, match="stop after prediction call"):
            function(model, object(), *args, sample_new_groups=True)

    assert len(record) == 1
    assert record[0].filename == __file__
    assert "sample_new_groups" not in predict.call_args.kwargs
