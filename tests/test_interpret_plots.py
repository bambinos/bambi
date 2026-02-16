import matplotlib
import numpy as np
import pandas as pd
import pytest
from seaborn.objects import Plot

import bambi as bmb
from bambi.interpret import plot_comparisons, plot_predictions, plot_slopes

# Render plots to a buffer instead of rendering to stddout
matplotlib.use("Agg")


# Improvement:
# * Test the actual plots are what we are indeed the desired result.
# * Test using the dictionary and the list gives the same plot
# * Use the same function for different models, e.g. average by, transforms, etc.


class TestCommon:
    """
    Tests arguments that are common to both 'plot_predictions', 'plot_comparisons',
    and 'plot_slopes' such as figure object and uncertainty arguments.
    """

    @pytest.mark.parametrize("pps", [False, True])
    def test_use_hdi(self, mtcars_fixture, pps):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", "am", use_hdi=False)
        assert isinstance(result, Plot)
        result = plot_predictions(
            model, idata, ["hp", "cyl", "gear"], pps=pps, use_hdi=False
        )
        assert isinstance(result, Plot)
        result = plot_slopes(model, idata, "hp", "am", use_hdi=False)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    def test_hdi_prob(self, mtcars_fixture, pps):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "am", "hp", prob=0.8)
        assert isinstance(result, Plot)
        result = plot_predictions(
            model, idata, ["hp", "cyl", "gear"], pps=pps, prob=0.8
        )
        assert isinstance(result, Plot)
        result = plot_slopes(model, idata, "hp", "am", prob=0.8)
        assert isinstance(result, Plot)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is 1.1.",
        ):
            plot_comparisons(model, idata, "am", "hp", prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is 1.1.",
        ):
            plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is 1.1.",
        ):
            plot_slopes(model, idata, "hp", "am", prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is -0.1.",
        ):
            plot_comparisons(model, idata, "am", "hp", prob=-0.1)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is -0.1.",
        ):
            plot_predictions(model, idata, ["hp", "cyl", "gear"], pps=pps, prob=-0.1)

        with pytest.raises(
            ValueError,
            match="'prob' must be greater than 0 and smaller than 1. It is -0.1.",
        ):
            plot_slopes(model, idata, "hp", "am", prob=-0.1)


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
    def test_basic(self, mtcars_fixture, covariates, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, pps=pps)
        assert isinstance(result, Plot)

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
    def test_with_groups(self, mtcars_fixture, covariates, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, pps=pps)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "covariates",
        (["hp", "cyl", "gear"], ["cyl", "hp", "gear"], ["cyl", "gear", "hp"]),
    )
    def test_with_group_and_panel(self, mtcars_fixture, covariates, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, pps=pps)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "conditional",
        [
            ({"hp": [110, 175], "am": [0, 1], "drat": [3, 4, 5]}),
            ({"hp": [150], "am": [1], "drat": [3, 4, 5]}),
        ],
    )
    def test_with_user_values(self, mtcars_fixture, conditional, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, conditional=conditional, pps=pps)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_predictions(model, idata, ["hp", "am", "drat"], average_by)
        assert isinstance(result, Plot)

        # unit level with average by covariates
        result = plot_predictions(model, idata, None, average_by)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    def test_fig_kwargs(self, mtcars_fixture, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(
            model,
            idata,
            ["hp", "cyl", "gear"],
            pps=pps,
            fig_kwargs={"sharey": True, "theme": {"font.size": 12}},
        )
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    def test_subplot_kwargs(self, mtcars_fixture, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(
            model,
            idata,
            ["hp", "drat"],
            pps=pps,
            subplot_kwargs={"main": "hp", "group": "drat", "panel": "drat"},
        )
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("pps", [False, True])
    @pytest.mark.parametrize(
        "transforms",
        (
            {"mpg": np.log},
            {"hp": np.log},
            {"mpg": np.log, "hp": np.log},
        ),
    )
    def test_transforms(self, mtcars_fixture, transforms, pps):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, ["hp"], pps=pps, transforms=transforms)
        assert isinstance(result, Plot)

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
        result = plot_predictions(model, idata, "x", pps=pps)
        assert isinstance(result, Plot)
        # Test user supplied target argument
        result = plot_predictions(model, idata, "x", target="alpha", pps=False)
        assert isinstance(result, Plot)

        # With alias
        alias = {
            "alpha": {"Intercept": "sd_intercept", "x": "sd_x", "alpha": "sd_alpha"}
        }
        model.set_alias(alias)
        idata = model.fit(tune=100, draws=100, random_seed=1234)

        # Test user supplied target argument
        result = plot_predictions(model, idata, "x", target="alpha", pps=False)
        assert isinstance(result, Plot)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        result = plot_predictions(
            model, idata, ["Days", "Subject"], sample_new_groups=True
        )
        assert isinstance(result, Plot)

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
        result = plot_predictions(model, idata, covariates)
        assert isinstance(result, Plot)

    def test_term_transformations(self, formulae_transform, nonformulae_transform):
        model, idata = formulae_transform

        # Test that the plot works with a formulae transformation
        result = plot_predictions(model, idata, ["x2", "x1"])
        assert isinstance(result, Plot)

        model, idata = nonformulae_transform

        # Test that the plot works with a non-formulae transformation
        result = plot_predictions(model, idata, "x1")
        assert isinstance(result, Plot)

    def test_same_variable_conditional_and_group(self, mtcars_fixture):
        model, idata = mtcars_fixture

        # Plot predictions where a categorical variable is passed to both
        # `conditional` and as the `group` variable
        result = plot_predictions(
            model=model,
            idata=idata,
            conditional="am",
            subplot_kwargs={"main": "am", "group": "am"},
        )
        assert isinstance(result, Plot)


class TestComparisons:
    """
    Tests the plot_comparisons function for different combinations of
    contrast and conditional variables, and user inputs.
    """

    @pytest.mark.parametrize(
        "contrast, conditional",
        [("hp", "am"), ("am", "hp")],  # numeric & categorical  # categorical & numeric
    )
    def test_basic(self, mtcars_fixture, contrast, conditional):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, contrast, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "contrast, conditional",
        [
            ("hp", ["am", "drat"]),  # numeric & [categorical, numeric]
            ("hp", ["drat", "am"]),  # numeric & [numeric, categorical]
        ],
    )
    def test_with_groups(self, mtcars_fixture, contrast, conditional):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, contrast, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "contrast, conditional",
        [
            ({"hp": [110, 175]}, ["am", "drat"]),  # user provided values
            (
                {"hp": [110, 175]},
                {"am": [0, 1], "drat": [3, 4, 5]},
            ),  # user provided values
        ],
    )
    def test_with_user_values(self, mtcars_fixture, contrast, conditional):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, contrast, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "contrast, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(
        self, mtcars_fixture, contrast, conditional, subplot_kwargs
    ):
        model, idata = mtcars_fixture
        result = plot_comparisons(
            model, idata, contrast, conditional, subplot_kwargs=subplot_kwargs
        )
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "contrast, conditional, transforms",
        [
            ("drat", ["hp", "am"], {"hp": np.log}),  # transform main numeric
            ("drat", ["hp", "am"], {"mpg": np.log}),  # transform response
        ],
    )
    def test_transforms(self, mtcars_fixture, contrast, conditional, transforms):
        model, idata = mtcars_fixture
        result = plot_comparisons(
            model, idata, contrast, conditional, transforms=transforms
        )
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_comparisons(model, idata, "hp", ["am", "drat"], average_by)
        assert isinstance(result, Plot)

        # unit level with average by
        result = plot_comparisons(model, idata, "hp", None, average_by)
        assert isinstance(result, Plot)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        result = plot_comparisons(
            model, idata, "Days", "Subject", sample_new_groups=True
        )
        assert isinstance(result, Plot)
        # user passed values seen in observed data
        result = plot_comparisons(
            model,
            idata,
            contrast={"Days": [2, 4]},
            conditional={"Subject": [308, 335, 352, 372]},
        )
        assert isinstance(result, Plot)

        with pytest.raises(
            ValueError,
            match="There are new groups for the factors \('Subject',\) and 'sample_new_groups' is False.",
        ):
            # default: sample_new_groups=False
            plot_comparisons(model, idata, "Days", "Subject")

    @pytest.mark.parametrize(
        "contrast, conditional",
        [
            ("sex", "length"),
            ("length", "sex"),
        ],  # Categorical & numeric  # Numeric & categorical
    )
    def test_categorical_response(self, food_choice, contrast, conditional):
        model, idata = food_choice
        result = plot_comparisons(model, idata, contrast, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("comparison", ["ratio", "lift"])
    def test_comparison_types(self, mtcars_fixture, comparison):
        model, idata = mtcars_fixture
        result = plot_comparisons(
            model, idata, "hp", "am", comparison=comparison
        )
        assert isinstance(result, Plot)

    def test_pps(self, mtcars_fixture):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", "am", pps=True)
        assert isinstance(result, Plot)


class TestSlopes:
    """
    Tests the 'plot_slopes' function for different combinations, elasticity,
    and effect types (unit and average slopes) of 'wrt' and 'conditional'
    variables.
    """

    def test_basic(self, mtcars_fixture):
        model, idata = mtcars_fixture
        # numeric wrt & categorical conditional
        result = plot_slopes(model, idata, "hp", "am")
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "wrt, conditional",
        [
            ("hp", ["am", "drat"]),  # numeric & [categorical, numeric]
            ("hp", ["drat", "am"]),  # numeric & [numeric, categorical]
        ],
    )
    def test_with_groups(self, mtcars_fixture, wrt, conditional):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, wrt, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "wrt, conditional",
        [
            ({"hp": 150}, ["am", "drat"]),  # single 'wrt' value
            ({"hp": 150}, {"am": [0, 1], "drat": [3, 4, 5]}),  # single 'wrt' value
        ],
    )
    def test_with_user_values(self, mtcars_fixture, wrt, conditional):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, wrt, conditional)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("slope", ["dydx", "dyex", "eyex", "eydx"])
    def test_elasticity(self, mtcars_fixture, slope):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, "hp", "drat", slope=slope)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "wrt, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(self, mtcars_fixture, wrt, conditional, subplot_kwargs):
        model, idata = mtcars_fixture
        result = plot_slopes(
            model, idata, wrt, conditional, subplot_kwargs=subplot_kwargs
        )
        assert isinstance(result, Plot)

    @pytest.mark.parametrize(
        "wrt, conditional, transforms",
        [
            ("drat", ["hp", "am"], {"hp": np.log}),  # transform main numeric
            ("drat", ["hp", "am"], {"mpg": np.log}),  # transform response
        ],
    )
    def test_transforms(self, mtcars_fixture, wrt, conditional, transforms):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, wrt, conditional, transforms=transforms)
        assert isinstance(result, Plot)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_slopes(model, idata, "hp", ["am", "drat"], average_by)
        assert isinstance(result, Plot)

        # unit level with average by
        result = plot_slopes(model, idata, "hp", None, average_by)
        assert isinstance(result, Plot)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        # contains new unseen data
        result = plot_slopes(model, idata, "Days", "Subject", sample_new_groups=True)
        assert isinstance(result, Plot)
        # user passed values seen in observed data
        result = plot_slopes(
            model, idata, wrt={"Days": 2}, conditional={"Subject": 308}
        )
        assert isinstance(result, Plot)

        with pytest.raises(
            ValueError,
            match="There are new groups for the factors \('Subject',\) and 'sample_new_groups' is False.",
        ):
            # default: sample_new_groups=False
            plot_slopes(model, idata, "Days", "Subject")

    def test_categorical_response(self, food_choice):
        model, idata = food_choice
        # Only numeric wrt is supported
        result = plot_slopes(model, idata, "length", "sex")
        assert isinstance(result, Plot)

    def test_pps(self, mtcars_fixture):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, "hp", "am", pps=True)
        assert isinstance(result, Plot)
