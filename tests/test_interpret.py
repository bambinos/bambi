"""
This module contains tests for the non-plotting functions of the 'interpret'
sub-package. In some cases, 'comparisons()', 'predictions()' and 'slopes()' 
contain arguments not in their respective plotting functions. Such arguments
are tested here.
"""
import pandas as pd
import pytest

import bambi as bmb


@pytest.fixture(scope="module")
def mtcars():
    "Model with common level effects only"
    data = bmb.load_data("mtcars")
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


@pytest.fixture(scope="module")
def sleep_study():
    "Model with common and group specific effects"
    data = bmb.load_data("sleepstudy")
    model = bmb.Model("Reaction ~ 1 + Days + (Days | Subject)", data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


@pytest.fixture(scope="module")
def food_choice():
    """
    Model a categorical response using the 'categorical' family to test 'interpret'
    plotting functions for a model whose predictions have multiple response
    dimensions (levels).
    """
    length = [
        1.3,
        1.32,
        1.32,
        1.4,
        1.42,
        1.42,
        1.47,
        1.47,
        1.5,
        1.52,
        1.63,
        1.65,
        1.65,
        1.65,
        1.65,
        1.68,
        1.7,
        1.73,
        1.78,
        1.78,
        1.8,
        1.85,
        1.93,
        1.93,
        1.98,
        2.03,
        2.03,
        2.31,
        2.36,
        2.46,
        3.25,
        3.28,
        3.33,
        3.56,
        3.58,
        3.66,
        3.68,
        3.71,
        3.89,
        1.24,
        1.3,
        1.45,
        1.45,
        1.55,
        1.6,
        1.6,
        1.65,
        1.78,
        1.78,
        1.8,
        1.88,
        2.16,
        2.26,
        2.31,
        2.36,
        2.39,
        2.41,
        2.44,
        2.56,
        2.67,
        2.72,
        2.79,
        2.84,
    ]
    choice = [
        "I",
        "F",
        "F",
        "F",
        "I",
        "F",
        "I",
        "F",
        "I",
        "I",
        "I",
        "O",
        "O",
        "I",
        "F",
        "F",
        "I",
        "O",
        "F",
        "O",
        "F",
        "F",
        "I",
        "F",
        "I",
        "F",
        "F",
        "F",
        "F",
        "F",
        "O",
        "O",
        "F",
        "F",
        "F",
        "F",
        "O",
        "F",
        "F",
        "I",
        "I",
        "I",
        "O",
        "I",
        "I",
        "I",
        "F",
        "I",
        "O",
        "I",
        "I",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "O",
        "F",
        "I",
        "F",
        "F",
    ]
    sex = ["Male"] * 32 + ["Female"] * 31
    data = pd.DataFrame({"choice": choice, "length": length, "sex": sex})
    data["choice"] = pd.Categorical(
        data["choice"].map({"I": "Invertebrates", "F": "Fish", "O": "Other"}),
        ["Other", "Invertebrates", "Fish"],
        ordered=True,
    )

    model = bmb.Model("choice ~ length + sex", data, family="categorical")
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


@pytest.mark.parametrize("return_idata", [True, False])
def test_return_idata_common_effects(mtcars, return_idata):
    model, idata = mtcars

    bmb.interpret.predictions(
        model, idata, ["hp", "wt"], return_idata=return_idata
        )
    bmb.interpret.comparisons(
        model, idata, "hp", "wt", return_idata=return_idata
    )
    bmb.interpret.slopes(
        model, idata, "hp", "wt", return_idata=return_idata
    )


@pytest.mark.parametrize("return_idata", [True, False])
def test_return_idata_group_effects(sleep_study, return_idata):
    model, idata = sleep_study

    bmb.interpret.predictions(
        model, idata, ["Days", "Subject"], sample_new_groups=True, return_idata=return_idata
        )
    bmb.interpret.comparisons(
        model, idata, "Days", "Subject", sample_new_groups=True, return_idata=return_idata
    )
    bmb.interpret.slopes(
        model, idata, "Days", "Subject", sample_new_groups=True, return_idata=return_idata
    )


@pytest.mark.parametrize("return_idata", [True, False])
def test_return_idata_categorical(food_choice, return_idata):
    model, idata = food_choice

    bmb.interpret.predictions(
        model, idata, ["length", "sex"], return_idata=return_idata
        )
    bmb.interpret.comparisons(
        model, idata, "sex", "length", return_idata=return_idata
    )
    bmb.interpret.slopes(
        model, idata, "length", "sex", return_idata=return_idata
    )