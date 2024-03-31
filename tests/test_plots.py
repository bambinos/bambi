from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import bambi as bmb
from bambi.interpret import plot_comparisons, plot_predictions, plot_slopes


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


@pytest.fixture(scope="module")
def formulae_transform():
    """
    A model with a 'formulae' stateful transformation (polynomial) on a term.
    """
    np.random.seed(0)
    x1 = np.random.normal(size=100)
    x2 = np.random.normal(size=100)
    y = 2 + 3*x1 + 1.5*x1**2 + 2*x2 + np.random.normal(scale=1, size=100)
    data = pd.DataFrame({'x1': x1, "x2": x2, 'y': y})
    model = bmb.Model('y ~ poly(x1, 2) + x2', data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


@pytest.fixture(scope="module")
def nonformulae_transform():
    """
    A model with a non-formulae transformation on a term. 
    """
    np.random.seed(0)
    x1 = np.random.uniform(1, 50, 50)
    noise = np.random.normal(0, 1, 50)
    y = 3 * np.log(x1) + noise
    data = pd.DataFrame({'x1': x1, 'y': y})

    model = bmb.Model('y ~ np.log(x1)', data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


# Improvement:
# * Test the actual plots are what we are indeed the desired result.
# * Test using the dictionary and the list gives the same plot
# * Use the same function for different models, e.g. average by, transforms, etc.


class TestCommon:
    """
    Tests argments that are common to both 'plot_predictions', 'plot_comparisons',
    and 'plot_slopes' such as figure object and uncertainty arguments.
    """

    @pytest.mark.parametrize("pps", [False, True])
    def test_use_hdi(self, mtcars, pps):
        model, idata = mtcars
        plot_comparisons(model, idata, "hp", "am", use_hdi=False)
        plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, use_hdi=False)
        plot_slopes(model, idata, "hp", "am", use_hdi=False)

    @pytest.mark.parametrize("pps", [False, True])
    def test_hdi_prob(self, mtcars, pps):
        model, idata = mtcars
        plot_comparisons(model, idata, "am", "hp", prob=0.8)
        plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, prob=0.8)
        plot_slopes(model, idata, "hp", "am", prob=0.8)

        with pytest.raises(
            ValueError, match="'prob' must be greater than 0 and smaller than 1. It is 1.1."
        ):
            plot_comparisons(model, idata, "am", "hp", prob=1.1)
            plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, prob=1.1)
            plot_slopes(model, idata, "hp", "am", prob=1.1)

        with pytest.raises(
            ValueError, match="'prob' must be greater than 0 and smaller than 1. It is -0.1."
        ):
            plot_comparisons(model, idata, "am", "hp", prob=-0.1)
            plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, prob=-0.1)
            plot_slopes(model, idata, "hp", "am", prob=0.1)

    @pytest.mark.parametrize("pps", [False, True])
    def test_legend(self, mtcars, pps):
        model, idata = mtcars
        plot_comparisons(model, idata, "am", "hp", legend=False)
        plot_predictions(model, idata, ["hp"], pps=pps, legend=False)
        plot_slopes(model, idata, "hp", "am", legend=False)

    @pytest.mark.parametrize("pps", [False, True])
    def test_ax(self, mtcars, pps):
        model, idata = mtcars
        fig, ax = plt.subplots()
        fig_r, ax_r = plot_comparisons(model, idata, "am", "hp", ax=ax)

        assert isinstance(ax_r, np.ndarray)
        assert fig is fig_r
        assert ax is ax_r[0]

        fig, ax = plt.subplots()
        fig_r, ax_r = plot_predictions(model, idata, ["hp"], pps=pps, ax=ax)

        assert isinstance(ax_r, np.ndarray)
        assert fig is fig_r
        assert ax is ax_r[0]

        fig, ax = plt.subplots()
        fig_r, ax_r = plot_slopes(model, idata, "hp", "am", ax=ax)

        assert isinstance(ax_r, np.ndarray)
        assert fig is fig_r
        assert ax is ax_r[0]


class TestPredictions:
    """
    Tests the 'plot_predictions' function for different combinations of main, group,
    and panel variables.
    """

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "covariates",
        (
            "hp",  # Main variable is numeric
            "gear",  # Main variable is categorical
            ["hp"],  # Using list
            ["gear"],  # Using list
        ),
    )
    def test_basic(self, mtcars, covariates, pps):
        model, idata = mtcars
        plot_predictions(model, idata, covariates, pps=pps)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "covariates",
        (
            ["hp", "wt"],  # Main: numeric. Group: numeric
            ["hp", "cyl"],  # Main: numeric. Group: categorical
            ["gear", "wt"],  # Main: categorical. Group: numeric
            ["gear", "cyl"],  # Main: categorical. Group: categorical
        ),
    )
    def test_with_groups(self, mtcars, covariates, pps):
        model, idata = mtcars
        plot_predictions(model, idata, covariates, pps=pps)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "covariates", (["hp", "cyl", "gear"], ["cyl", "hp", "gear"], ["cyl", "gear", "hp"])
    )
    def test_with_group_and_panel(self, mtcars, covariates, pps):
        model, idata = mtcars
        plot_predictions(model, idata, covariates, pps=pps)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "conditional",
        [
            ({"hp": [110, 175], "am": [0, 1], "drat": [3, 4, 5]}),
            ({"hp": 150, "am": 1, "drat": [3, 4, 5]}),
        ],
    )
    def test_with_user_values(self, mtcars, conditional, pps):
        model, idata = mtcars
        plot_predictions(model, idata, conditional=conditional, pps=pps)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars, average_by):
        model, idata = mtcars

        # grid of values with average_by
        plot_predictions(model, idata, ["hp", "am", "drat"], average_by)

        # unit level with average by covariates
        plot_predictions(model, idata, None, average_by)

    @pytest.mark.parametrize("pps", [False, True])
    def test_fig_kwargs(self, mtcars, pps):
        model, idata = mtcars
        plot_predictions(
            model,
            idata,
            ["hp", "cyl", "gear"],
            pps=pps,
            fig_kwargs={"figsize": (15, 5), "dpi": 120, "sharey": True},
        )

    @pytest.mark.parametrize("pps", [False, True])
    def test_subplot_kwargs(self, mtcars, pps):
        model, idata = mtcars
        plot_predictions(
            model,
            idata,
            ["hp", "drat"],
            pps=pps,
            subplot_kwargs={"main": "hp", "group": "drat", "panel": "drat"},
        )

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "transforms",
        (
            {"mpg": np.log},
            {"hp": np.log},
            {"mpg": np.log, "hp": np.log},
        ),
    )
    def test_transforms(self, mtcars, transforms, pps):
        model, idata = mtcars
        plot_predictions(model, idata, ["hp"], pps=pps, transforms=transforms)

    @pytest.mark.parametrize("pps", [False, True])
    def test_multiple_outputs_with_alias(self, pps):
        """Test plot cap default and specified values for target argument"""
        rng = np.random.default_rng(121195)
        N = 200
        a, b = 0.5, 1.1
        x = rng.uniform(-1.5, 1.5, N)
        shape = np.exp(0.3 + x * 0.5 + rng.normal(scale=0.1, size=N))
        y = rng.gamma(shape, np.exp(a + b * x) / shape, N)
        data_gamma = pd.DataFrame({"x": x, "y": y})

        formula = bmb.Formula("y ~ x", "alpha ~ x")
        model = bmb.Model(formula, data_gamma, family="gamma")

        # Without alias
        idata = model.fit(tune=100, draws=100, random_seed=1234)
        # Test default target
        plot_predictions(model, idata, "x", pps=pps)
        # Test user supplied target argument
        plot_predictions(model, idata, "x", target="alpha", pps=False)

        # With alias
        alias = {"alpha": {"Intercept": "sd_intercept", "x": "sd_x", "alpha": "sd_alpha"}}
        model.set_alias(alias)
        idata = model.fit(tune=100, draws=100, random_seed=1234)

        # Test user supplied target argument
        plot_predictions(model, idata, "x", target="alpha", pps=False)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        plot_predictions(model, idata, ["Days", "Subject"], sample_new_groups=True)

        with pytest.raises(
            ValueError,
            match="There are new groups for the factors \('Subject',\) and 'sample_new_groups' is False.",
        ):
            # default: sample_new_groups=False
            plot_predictions(model, idata, ["Days", "Subject"])

    @pytest.mark.parametrize(
        "covariates",
        (
            "length",  # Main variable is numeric
            "sex",  # Main variable is categorical
            ["length", "sex"],  # Using both covariates
        ),
    )
    def test_categorical_response(self, food_choice, covariates):
        model, idata = food_choice
        plot_predictions(model, idata, covariates)
    
    
    def test_term_transformations(self, formulae_transform, nonformulae_transform):
        model, idata = formulae_transform

        # Test that the plot works with a formulae transformation
        plot_predictions(model, idata, ["x2", "x1"])

        model, idata = nonformulae_transform
    
        # Test that the plot works with a non-formulae transformation
        plot_predictions(model, idata, "x1")


class TestComparison:
    """
    Tests the plot_comparisons function for different combinations of
    contrast and conditional variables, and user inputs.
    """

    @pytest.mark.parametrize(
        "contrast, conditional",
        [("hp", "am"), ("am", "hp")],  # numeric & categorical  # categorical & numeric
    )
    def test_basic(self, mtcars, contrast, conditional):
        model, idata = mtcars
        plot_comparisons(model, idata, contrast, conditional)

    @pytest.mark.parametrize(
        "contrast, conditional",
        [
            ("hp", ["am", "drat"]),  # numeric & [categorical, numeric]
            ("hp", ["drat", "am"]),  # numeric & [numeric, categorical]
        ],
    )
    def test_with_groups(self, mtcars, contrast, conditional):
        model, idata = mtcars
        plot_comparisons(model, idata, contrast, conditional)

    @pytest.mark.parametrize(
        "contrast, conditional",
        [
            ({"hp": [110, 175]}, ["am", "drat"]),  # user provided values
            ({"hp": [110, 175]}, {"am": [0, 1], "drat": [3, 4, 5]}),  # user provided values
        ],
    )
    def test_with_user_values(self, mtcars, contrast, conditional):
        model, idata = mtcars
        plot_comparisons(model, idata, contrast, conditional)

    @pytest.mark.parametrize(
        "contrast, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(self, mtcars, contrast, conditional, subplot_kwargs):
        model, idata = mtcars
        plot_comparisons(model, idata, contrast, conditional, subplot_kwargs=subplot_kwargs)

    @pytest.mark.parametrize(
        "contrast, conditional, transforms",
        [
            ("drat", ["hp", "am"], {"hp": np.log}),  # transform main numeric
            ("drat", ["hp", "am"], {"mpg": np.log}),  # transform response
        ],
    )
    def test_transforms(self, mtcars, contrast, conditional, transforms):
        model, idata = mtcars
        plot_comparisons(model, idata, contrast, conditional, transforms=transforms)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars, average_by):
        model, idata = mtcars

        # grid of values with average_by
        plot_comparisons(model, idata, "hp", ["am", "drat"], average_by)

        # unit level with average by
        plot_comparisons(model, idata, "hp", None, average_by)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        plot_comparisons(model, idata, "Days", "Subject", sample_new_groups=True)
        # user passed values seen in observed data
        plot_comparisons(
            model,
            idata,
            contrast={"Days": [2, 4]},
            conditional={"Subject": [308, 335, 352, 372]},
        )

        with pytest.raises(
            ValueError,
            match="There are new groups for the factors \('Subject',\) and 'sample_new_groups' is False.",
        ):
            # default: sample_new_groups=False
            plot_comparisons(model, idata, "Days", "Subject")

    @pytest.mark.parametrize(
        "contrast, conditional",
        [("sex", "length"), ("length", "sex")],  # Categorical & numeric  # Numeric & categorical
    )
    def test_categorical_response(self, food_choice, contrast, conditional):
        model, idata = food_choice
        plot_comparisons(model, idata, contrast, conditional)


class TestSlopes:
    """
    Tests the 'plot_slopes' function for different combinations, elasticity,
    and effect types (unit and average slopes) of 'wrt' and 'conditional'
    variables.
    """

    @pytest.mark.parametrize(
        "wrt, conditional",
        [("hp", "am"), ("am", "hp")],  # numeric & categorical  # categorical & numeric
    )
    def test_basic(self, mtcars, wrt, conditional):
        model, idata = mtcars
        plot_slopes(model, idata, wrt, conditional)

    @pytest.mark.parametrize(
        "wrt, conditional",
        [
            ("hp", ["am", "drat"]),  # numeric & [categorical, numeric]
            ("hp", ["drat", "am"]),  # numeric & [numeric, categorical]
        ],
    )
    def test_with_groups(self, mtcars, wrt, conditional):
        model, idata = mtcars
        plot_slopes(model, idata, wrt, conditional)

    @pytest.mark.parametrize(
        "wrt, conditional, average_by",
        [
            ({"hp": 150}, ["am", "drat"], None),  # single 'wrt' values
            ({"hp": 150}, {"am": [0, 1], "drat": [3, 4, 5]}, None),  # single 'wrt' values
            ({"hp": [150, 200]}, ["am", "drat"], "am"),  # multiple 'wrt' values
            (
                {"hp": [150, 200]},
                {"am": [0, 1], "drat": [3, 4, 5]},
                "drat",
            ),  # multiple 'wrt' values
        ],
    )
    def test_with_user_values(self, mtcars, wrt, conditional, average_by):
        model, idata = mtcars
        # need to average by if greater than 1 value is passed with 'wrt'
        plot_slopes(model, idata, wrt, conditional, average_by=average_by)

    @pytest.mark.parametrize("slope", ["dydx", "dyex", "eyex", "eydx"])
    def test_elasticity(self, mtcars, slope):
        model, idata = mtcars
        plot_slopes(model, idata, "hp", "drat", slope=slope)

    @pytest.mark.parametrize(
        "wrt, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(self, mtcars, wrt, conditional, subplot_kwargs):
        model, idata = mtcars
        plot_slopes(model, idata, wrt, conditional, subplot_kwargs=subplot_kwargs)

    @pytest.mark.parametrize(
        "wrt, conditional, transforms",
        [
            ("drat", ["hp", "am"], {"hp": np.log}),  # transform main numeric
            ("drat", ["hp", "am"], {"mpg": np.log}),  # transform response
        ],
    )
    def test_transforms(self, mtcars, wrt, conditional, transforms):
        model, idata = mtcars
        plot_slopes(model, idata, wrt, conditional, transforms=transforms)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars, average_by):
        model, idata = mtcars

        # grid of values with average_by
        plot_slopes(model, idata, "hp", ["am", "drat"], average_by)

        # unit level with average by
        plot_slopes(model, idata, "hp", None, average_by)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        plot_slopes(model, idata, "Days", "Subject", sample_new_groups=True)
        # user passed values seen in observed data
        plot_slopes(model, idata, wrt={"Days": 2}, conditional={"Subject": 308})

        with pytest.raises(
            ValueError,
            match="There are new groups for the factors \('Subject',\) and 'sample_new_groups' is False.",
        ):
            # default: sample_new_groups=False
            plot_slopes(model, idata, "Days", "Subject")

    @pytest.mark.parametrize(
        "wrt, conditional",
        [("sex", "length"), ("length", "sex")],  # Categorical & numeric  # Numeric & categorical
    )
    def test_categorical_response(self, food_choice, wrt, conditional):
        model, idata = food_choice
        plot_slopes(model, idata, wrt, conditional)
