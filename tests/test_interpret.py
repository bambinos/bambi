"""
This module contains tests for the helper functions of the 'interpret' sub-package.
Tests here do not test any of the plotting functionality.
"""

import numpy as np
import pandas as pd
import pytest

import bambi as bmb
from bambi.interpret.helpers import data_grid, select_draws
from bambi.interpret.utils import get_model_covariates


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
# The tests below test these two functionalities.
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
    ids=["defaults", "defaults_and_user_passed", "user_passed"],
)
def test_data_grid_no_effect(request, mtcars, conditional):
    model, idata = mtcars
    grid = data_grid(model, conditional)

    id = request.node.name
    if id == "defaults":
        assert grid.shape == (48, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]
    elif id == "defaults_and_user_passed":
        assert grid.shape == (14, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]
    elif id == "user_passed":
        assert grid.shape == (28, 3)
        assert grid.columns.tolist() == ["hp", "drat", "am"]


def test_data_grid_no_effect_kwargs(request, mtcars):
    model, idata = mtcars
    grid = data_grid(model, ["hp", "drat"], num=10)

    assert grid.shape == (100, 3)
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
    ids=["defaults", "user_passed", "defaults_and_user_passed"],
)
def test_data_grid_comparisons(request, mtcars, conditional, variable):
    model, idata = mtcars
    grid = data_grid(model, conditional, variable=variable, effect_type="comparisons")

    id = request.node.name
    if id == "defaults":
        assert grid.shape == (200, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]
    elif id == "user_passed" or id == "defaults_and_user_passed":
        assert grid.shape == (16, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]

    with pytest.raises(
        ValueError,
        match="'If passing an argument to 'variable', the parameter 'effect_type' must be either "
        f"'comparisons' or 'slopes'. Received: {None}",
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
    ids=["defaults", "user_passed", "defaults_and_user_passed"],
)
def test_data_grid_slopes(request, mtcars, conditional, variable):
    model, idata = mtcars
    grid = data_grid(model, conditional, variable=variable, effect_type="slopes")

    id = request.node.name
    if id == "defaults":
        assert grid.shape == (200, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]
    elif id == "user_passed" or id == "defaults_and_user_passed":
        assert grid.shape == (16, 3)
        assert grid.columns.tolist() == ["drat", "am", "hp"]

    with pytest.raises(
        ValueError,
        match="'If passing an argument to 'variable', the parameter 'effect_type' must be either "
        f"'comparisons' or 'slopes'. Received: {None}",
    ):
        data_grid(model, conditional, variable=variable, effect_type=None)


@pytest.mark.parametrize(
    "effect_type, eps", [("comparisons", 1), ("slopes", 1e-2)], ids=["comparisons", "slopes"]
)
def test_data_grid_eps(request, mtcars, effect_type, eps):
    model, idata = mtcars
    grid = data_grid(model, ["drat", "am"], "hp", effect_type, eps=eps)
    unit_difference = np.unique(np.abs(np.diff(grid["hp"])))

    id = request.node.name
    if id == "comparisons":
        # centered difference 'eps' of 1 adds and subtracts 1 to the default "hp" value
        assert unit_difference == np.array([2])
    elif id == "slopes":
        # finite difference 'eps' of 1e-2 adds 0.01 to the default "hp" value
        assert unit_difference == np.array([0.01])


# -------------------------------------------------------------------
#                      Tests for `select_draws`
#
# Select posterior or posterior predictive draws conditioned on the
# observation that produced that draw by passing a `condition` dictionary.
# `data_grid` is used to create the grid of data that is passed to `model.predict`.
# Then, different 'condition' dictionaries are passed to `select_draws` to
# ensure the output shape of the selected draws is correct.
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition",
    [
        ({"hp": 250}),
        ({"drat": 2.5}),
        ({"hp": 250, "drat": 2.5}),
    ],
    ids=["1", "2", "3"],
)
def test_select_draws_no_effect(request, mtcars, condition):
    model, idata = mtcars

    conditional = {"hp": np.linspace(50, 350, 7), "drat": [2.5, 3.5], "am": [0, 1]}
    grid = data_grid(model, conditional)

    idata = model.predict(idata, data=grid, inplace=False)
    draws = select_draws(idata, grid, condition=condition, data_var="mu")

    # (CHAINS, DRAWS, n) where n is the number of observations that satisfy the condition
    id = request.node.name
    if id == "1":
        assert draws.shape == (CHAINS, DRAWS, 4)
    elif id == "2":
        assert draws.shape == (CHAINS, DRAWS, 14)
    elif id == "3":
        assert draws.shape == (CHAINS, DRAWS, 2)


# ------------------------------------------------------------------------------------------------ #
#                                         Tests for utils                                          #
# ------------------------------------------------------------------------------------------------ #


def test_get_model_covariates():
    """Tests `get_model_covariates()` does not include non-covariate names"""
    # See issue 797
    df = pd.DataFrame({"y": np.arange(10), "x": np.random.normal(size=10)})
    knots = np.linspace(np.min(df["x"]), np.max(df["x"]), 4 + 2)[1:-1]
    formula = "y ~ 1 + bs(x, degree=3, knots=knots)"
    model = bmb.Model(formula, df)
    assert set(get_model_covariates(model)) == {"x"}
