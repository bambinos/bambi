"""
Tests for the computation functions of the 'interpret' sub-package.
Tests here do not test the plotting functionality.
"""

import numpy as np
import pandas as pd
import pytest
from arviz import InferenceData

from bambi.interpret import comparisons, predictions, slopes
from bambi.interpret.types import Result

# -------------------------------------------------------------------
#                       Tests for `predictions`
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "conditional",
    [
        "hp",
        ["hp", "drat"],
        {"hp": [110, 175], "am": [0, 1]},
    ],
    ids=["str", "list", "dict"],
)
def test_predictions_returns_result(mtcars_fixture, conditional):
    model, idata = mtcars_fixture
    result = predictions(model, idata, conditional=conditional)

    assert isinstance(result, Result)
    assert isinstance(result.summary, pd.DataFrame)
    assert isinstance(result.draws, InferenceData)
    assert "estimate" in result.summary.columns
    assert any("lower" in col for col in result.summary.columns)
    assert any("upper" in col for col in result.summary.columns)


def test_predictions_unit_level(mtcars_fixture):
    """Unit-level predictions (conditional=None) require average_by."""
    model, idata = mtcars_fixture
    result = predictions(model, idata, conditional=None, average_by="am")

    assert isinstance(result, Result)
    assert len(result.summary) > 0


@pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
def test_predictions_average_by(mtcars_fixture, average_by):
    model, idata = mtcars_fixture

    result_full = predictions(model, idata, conditional=["hp", "am", "drat"])
    result_avg = predictions(
        model, idata, conditional=["hp", "am", "drat"], average_by=average_by
    )

    assert len(result_avg.summary) < len(result_full.summary)


@pytest.mark.parametrize("pps", [False, True])
def test_predictions_pps(mtcars_fixture, pps):
    model, idata = mtcars_fixture
    result = predictions(model, idata, conditional="hp", pps=pps)
    assert isinstance(result, Result)


def test_predictions_prob_validation(mtcars_fixture):
    model, idata = mtcars_fixture

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        predictions(model, idata, conditional="hp", prob=1.1)

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        predictions(model, idata, conditional="hp", prob=-0.1)


def test_predictions_average_by_all(mtcars_fixture):
    """Test average_by='all' reduces to a single-row summary."""
    model, idata = mtcars_fixture
    result = predictions(model, idata, conditional=["hp", "am"], average_by="all")
    assert isinstance(result, Result)
    assert len(result.summary) == 1


@pytest.mark.parametrize(
    "transforms",
    [
        {"mpg": np.log},
        {"hp": np.log},
        {"mpg": np.log, "hp": np.log},
    ],
)
def test_predictions_transforms(mtcars_fixture, transforms):
    model, idata = mtcars_fixture
    result = predictions(model, idata, conditional="hp", transforms=transforms)
    assert isinstance(result, Result)


def test_predictions_group_effects(sleep_study):
    model, idata = sleep_study
    result = predictions(
        model, idata, conditional=["Days", "Subject"], sample_new_groups=True
    )
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_predictions_categorical_response(food_choice):
    model, idata = food_choice
    result = predictions(model, idata, conditional="length")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_predictions_distributional_target(distributional_fixture):
    model, idata = distributional_fixture
    result = predictions(model, idata, conditional="x", target="alpha")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_predictions_integer_predictor(integer_data_fixture):
    model, idata = integer_data_fixture
    result = predictions(model, idata, conditional="x_int")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


# -------------------------------------------------------------------
#                       Tests for `comparisons`
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "contrast, conditional",
    [
        ("hp", "am"),
        ("am", "hp"),
    ],
    ids=["numeric_contrast", "categorical_contrast"],
)
def test_comparisons_returns_result(mtcars_fixture, contrast, conditional):
    model, idata = mtcars_fixture
    result = comparisons(model, idata, contrast=contrast, conditional=conditional)

    assert isinstance(result, Result)
    assert isinstance(result.summary, pd.DataFrame)
    assert isinstance(result.draws, InferenceData)
    assert "term" in result.summary.columns
    assert "estimate_type" in result.summary.columns
    assert "value" in result.summary.columns
    assert "estimate" in result.summary.columns


@pytest.mark.parametrize(
    "contrast, conditional",
    [
        ({"hp": [110, 175]}, ["am", "drat"]),
        ({"hp": [110, 175]}, {"am": [0, 1], "drat": [3, 4, 5]}),
    ],
)
def test_comparisons_with_user_values(mtcars_fixture, contrast, conditional):
    model, idata = mtcars_fixture
    result = comparisons(model, idata, contrast=contrast, conditional=conditional)
    assert isinstance(result, Result)


@pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
def test_comparisons_average_by(mtcars_fixture, average_by):
    model, idata = mtcars_fixture

    result_full = comparisons(model, idata, contrast="hp", conditional=["am", "drat"])
    result_avg = comparisons(
        model, idata, contrast="hp", conditional=["am", "drat"], average_by=average_by
    )

    assert len(result_avg.summary) < len(result_full.summary)


def test_comparisons_prob_validation(mtcars_fixture):
    model, idata = mtcars_fixture

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        comparisons(model, idata, contrast="hp", conditional="am", prob=1.1)

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        comparisons(model, idata, contrast="hp", conditional="am", prob=-0.1)


@pytest.mark.parametrize("comparison", ["ratio", "lift"])
def test_comparisons_comparison_types(mtcars_fixture, comparison):
    model, idata = mtcars_fixture
    result = comparisons(
        model, idata, contrast="hp", conditional="am", comparison=comparison
    )
    assert isinstance(result, Result)
    assert result.summary["estimate_type"].iloc[0] == comparison


def test_comparisons_custom_callable(mtcars_fixture):
    model, idata = mtcars_fixture

    def my_comparison(reference, contrast):
        return contrast - 2 * reference

    result = comparisons(
        model, idata, contrast="hp", conditional="am", comparison=my_comparison
    )
    assert isinstance(result, Result)
    assert result.summary["estimate_type"].iloc[0] == "my_comparison"


def test_comparisons_unit_level(mtcars_fixture):
    model, idata = mtcars_fixture
    result = comparisons(model, idata, contrast="hp", conditional=None, average_by="am")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_comparisons_pps(mtcars_fixture):
    model, idata = mtcars_fixture
    result = comparisons(model, idata, contrast="hp", conditional="am", pps=True)
    assert isinstance(result, Result)


def test_comparisons_group_effects(sleep_study):
    model, idata = sleep_study
    result = comparisons(
        model,
        idata,
        contrast={"Days": [2, 4]},
        conditional={"Subject": [308, 335, 352, 372]},
    )
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_comparisons_categorical_response(food_choice):
    model, idata = food_choice
    result = comparisons(model, idata, contrast="sex", conditional="length")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_comparisons_integer_contrast(integer_data_fixture):
    model, idata = integer_data_fixture
    result = comparisons(model, idata, contrast="x_int", conditional="x_float")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


@pytest.mark.parametrize(
    "transforms",
    [
        {"mpg": np.log},
        {"hp": np.log},
    ],
)
def test_comparisons_transforms(mtcars_fixture, transforms):
    model, idata = mtcars_fixture
    result = comparisons(
        model, idata, contrast="drat", conditional=["hp", "am"], transforms=transforms
    )
    assert isinstance(result, Result)


# -------------------------------------------------------------------
#                       Tests for `slopes`
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrt, conditional",
    [
        ("hp", "am"),
        ("hp", ["am", "drat"]),
    ],
    ids=["str_wrt", "list_conditional"],
)
def test_slopes_returns_result(mtcars_fixture, wrt, conditional):
    model, idata = mtcars_fixture
    result = slopes(model, idata, wrt=wrt, conditional=conditional)

    assert isinstance(result, Result)
    assert isinstance(result.summary, pd.DataFrame)
    assert isinstance(result.draws, InferenceData)
    assert "term" in result.summary.columns
    assert "estimate_type" in result.summary.columns
    assert "value" in result.summary.columns
    assert "estimate" in result.summary.columns


def test_slopes_with_user_values(mtcars_fixture):
    model, idata = mtcars_fixture
    result = slopes(model, idata, wrt={"hp": 150}, conditional=["am", "drat"])
    assert isinstance(result, Result)


@pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
def test_slopes_average_by(mtcars_fixture, average_by):
    model, idata = mtcars_fixture

    result_full = slopes(model, idata, wrt="hp", conditional=["am", "drat"])
    result_avg = slopes(
        model, idata, wrt="hp", conditional=["am", "drat"], average_by=average_by
    )

    assert len(result_avg.summary) < len(result_full.summary)


@pytest.mark.parametrize("slope", ["dydx", "dyex", "eyex", "eydx"])
def test_slopes_elasticity(mtcars_fixture, slope):
    model, idata = mtcars_fixture
    result = slopes(model, idata, wrt="hp", conditional="drat", slope=slope)
    assert isinstance(result, Result)


def test_slopes_prob_validation(mtcars_fixture):
    model, idata = mtcars_fixture

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        slopes(model, idata, wrt="hp", conditional="am", prob=1.1)

    with pytest.raises(
        ValueError, match="'prob' must be greater than 0 and smaller than 1"
    ):
        slopes(model, idata, wrt="hp", conditional="am", prob=-0.1)


def test_slopes_unit_level(mtcars_fixture):
    model, idata = mtcars_fixture
    result = slopes(model, idata, wrt="hp", conditional=None, average_by="am")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_slopes_custom_callable(mtcars_fixture):
    model, idata = mtcars_fixture

    def my_slope(derivative, x, y):
        return derivative * 2

    result = slopes(model, idata, wrt="hp", conditional="drat", slope=my_slope)
    assert isinstance(result, Result)
    assert result.summary["estimate_type"].iloc[0] == "my_slope"


def test_slopes_pps(mtcars_fixture):
    model, idata = mtcars_fixture
    result = slopes(model, idata, wrt="hp", conditional="am", pps=True)
    assert isinstance(result, Result)


def test_slopes_group_effects(sleep_study):
    model, idata = sleep_study
    result = slopes(
        model,
        idata,
        wrt={"Days": 2},
        conditional={"Subject": [308, 335, 352, 372]},
    )
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_slopes_categorical_response(food_choice):
    model, idata = food_choice
    result = slopes(model, idata, wrt="length", conditional="sex")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


def test_slopes_integer_wrt(integer_data_fixture):
    model, idata = integer_data_fixture
    result = slopes(model, idata, wrt="x_int", conditional="x_float")
    assert isinstance(result, Result)
    assert len(result.summary) > 0


@pytest.mark.parametrize(
    "transforms",
    [
        {"mpg": np.log},
        {"hp": np.log},
    ],
)
def test_slopes_transforms(mtcars_fixture, transforms):
    model, idata = mtcars_fixture
    result = slopes(
        model, idata, wrt="drat", conditional=["hp", "am"], transforms=transforms
    )
    assert isinstance(result, Result)
