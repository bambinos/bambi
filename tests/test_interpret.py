"""
This module contains tests for the helper functions of the 'interpret' sub-package.
Tests here do not test any of the plotting functionality.
"""
import numpy as np
import pandas as pd
import pytest

import bambi as bmb
from bambi.interpret.helpers import data_grid, select_draws

CHAINS = 4
TUNE = 500
DRAWS = 500


@pytest.fixture(scope="module")
def mtcars():
    "Model with common level effects only"
    data = bmb.load_data("mtcars")
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=TUNE, draws=DRAWS, chains=CHAINS, random_seed=1234)
    return model, idata


# -------------------------------------------------------------------
#                       Tests for `data_grid`
#
# `data_grid` serves several functions: (1) the ability to create a pairwise
# grid of data passing an argument to 'conditional' with no regard to the effect
# type, and (2) the ability to create a grid of data with respect to an effect
# type by passing an argument to 'conditional', 'variable', and 'effect_type'.
# The tests below test these functionalities.
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "conditional",
    [
        # default values for 'hp', 'drat', and 'am'
        (["hp", "drat"]),
        # default values computed for 'am' and user-passed for 'hp' and 'drat'
        ({"hp": np.linspace(50, 350, 7), "drat": [2.5, 3.5]}),
        # user-passed for 'hp', 'drat', and 'am'
        ({"hp": np.linspace(50, 350, 7), "drat": [2.5, 3.5], "am": [0, 1]}),
    ],
    ids=["1", "2", "3"],
)
def test_data_grid_no_effect(request, mtcars, conditional):
    model, idata = mtcars
    grid = data_grid(model, conditional)

    id = request.node.name
    if id == "1":
        assert grid.shape == (48, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]
    elif id == "2":
        assert grid.shape == (14, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]
    elif id == "3":
        assert grid.shape == (28, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]


@pytest.mark.parametrize(
    "conditional, variable",
    [
        # default values for 'conditional' and 'variable'
        (["drat", "am"], "hp"),
        # user-passed for 'conditional' and 'variable'
        ({"drat": np.arange(1, 5, 1), "am": np.array([0, 1])}, {"hp": np.array([150, 300])}),
        # user-passed for 'conditional' and default value for 'variable'
        ({"drat": np.arange(1, 5, 1), "am": np.array([0, 1])}, "hp"),
    ],
    ids=["1", "2", "3"],
)
def test_data_grid_comparisons(request, mtcars, conditional, variable):
    model, idata = mtcars
    grid = data_grid(model, conditional, variable=variable, effect_type="comparisons")

    id = request.node.name
    if id == "1":
        assert grid.shape == (200, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]
    elif id == "2" or id == "3":
        assert grid.shape == (16, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]

    with pytest.raises(
        ValueError, match="'effect_type' must be specified if argument for 'variable' is passed."
    ):
        data_grid(model, conditional, variable=variable, effect_type=None)


@pytest.mark.parametrize(
    "conditional, variable",
    [
        # default values for 'conditional' and 'variable'
        (["drat", "am"], "hp"),
        # user-passed for 'conditional' and 'variable'
        ({"drat": np.arange(1, 5, 1), "am": np.array([0, 1])}, {"hp": np.array([150])}),
        # user-passed for 'conditional' and default value for 'variable'
        ({"drat": np.arange(1, 5, 1), "am": np.array([0, 1])}, "hp"),
    ],
    ids=["1", "2", "3"],
)
def test_data_grid_slopes(request, mtcars, conditional, variable):
    model, idata = mtcars
    grid = data_grid(model, conditional, variable=variable, effect_type="slopes")

    id = request.node.name
    if id == "1":
        assert grid.shape == (200, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]
    elif id == "2" or id == "3":
        assert grid.shape == (16, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]

    with pytest.raises(
        ValueError, match="'effect_type' must be specified if argument for 'variable' is passed."
    ):
        data_grid(model, conditional, variable=variable, effect_type=None)


# -------------------------------------------------------------------
#                      Tests for `select_draws`
#
# Select posterior or posterior predictive draws conditioned on the
# observation that produced that draw by passing a `condition` dictionary.
# `data_grid` is used to create the grid of data that is passed to `model.predict`.
# Then, a variety of 'condition' dictionaries are passed to `select_draws` to
# test the output shape of the draws is correct.
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition",
    [
        ({"hp": 225}),
        ({"drat": 2.5}),
        ({"hp": 250, "drat": 2.5}),
    ],
    ids=["1", "2", "3"],
)
def test_select_draws_no_effect(request, mtcars, conditional, condition):
    model, idata = mtcars

    conditional = {"hp": np.linspace(50, 350, 7), "drat": [2.5, 3.5], "am": [0, 1]}
    grid = data_grid(model, conditional)

    idata = model.predict(idata, data=grid, inplace=False)
    draws = select_draws(idata, grid, condition=condition, data_var="mpg_mean")

    # (CHAINS, DRAWS, n) where n is the number of observations that satisfy the condition
    id = request.node.name
    if id == "1":
        assert draws.shape == (CHAINS, DRAWS, 2)
    elif id == "2":
        assert draws.shape == (CHAINS, DRAWS, 7)
    elif id == "3":
        assert draws.shape == (CHAINS, DRAWS, 1)
