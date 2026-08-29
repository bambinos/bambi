import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import bambi as bmb
from bambi.interpret import plot_comparisons, plot_predictions, plot_slopes
from bambi.interpret.effects import comparisons, predictions, slopes
from bambi.interpret.plots import PlottingConfig, plot

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

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    def test_use_hdi(self, mtcars_fixture, target):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", "am", use_hdi=False)
        assert isinstance(result, Figure)
        result = plot_predictions(model, idata, ["hp", "cyl", "gear"], target=target, use_hdi=False)
        assert isinstance(result, Figure)
        result = plot_slopes(model, idata, "hp", "am", use_hdi=False)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    def test_hdi_prob(self, mtcars_fixture, target):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "am", "hp", prob=0.8)
        assert isinstance(result, Figure)
        result = plot_predictions(model, idata, ["hp", "cyl", "gear"], target=target, prob=0.8)
        assert isinstance(result, Figure)
        result = plot_slopes(model, idata, "hp", "am", prob=0.8)
        assert isinstance(result, Figure)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got 1.1.",
        ):
            plot_comparisons(model, idata, "am", "hp", prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got 1.1.",
        ):
            plot_predictions(model, idata, ["hp", "cyl", "gear"], target=target, prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got 1.1.",
        ):
            plot_slopes(model, idata, "hp", "am", prob=1.1)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got -0.1.",
        ):
            plot_comparisons(model, idata, "am", "hp", prob=-0.1)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got -0.1.",
        ):
            plot_predictions(model, idata, ["hp", "cyl", "gear"], target=target, prob=-0.1)

        with pytest.raises(
            ValueError,
            match="'prob' values must be between 0 and 1 \\(exclusive\\). Got -0.1.",
        ):
            plot_slopes(model, idata, "hp", "am", prob=-0.1)

    def test_multiple_prob(self, mtcars_fixture):
        model, idata = mtcars_fixture
        # Numeric main (uses Band) and categorical main (uses Range)
        result = plot_predictions(model, idata, "hp", prob=[0.5, 0.94])
        assert isinstance(result, Figure)
        result = plot_predictions(model, idata, "gear", prob=[0.5, 0.94])
        assert isinstance(result, Figure)
        result = plot_comparisons(model, idata, "hp", "am", prob=[0.5, 0.94])
        assert isinstance(result, Figure)
        result = plot_slopes(model, idata, "hp", "am", prob=[0.5, 0.94])
        assert isinstance(result, Figure)

    def test_none_prob_omits_credible_intervals(self, mtcars_fixture):
        model, idata = mtcars_fixture

        for effect, args in (
            (predictions, ("hp",)),
            (comparisons, ("hp", "am")),
            (slopes, ("hp", "am")),
        ):
            summary = effect(model, idata, *args, prob=None).summary
            assert list(summary.filter(regex="^(lower|upper)_").columns) == []

        for plot_effect, args in (
            (plot_predictions, ("hp",)),
            (plot_comparisons, ("hp", "am")),
            (plot_slopes, ("hp", "am")),
        ):
            figure = plot_effect(model, idata, *args, prob=None)
            assert isinstance(figure, Figure)

    def test_plot_customization(self, mtcars_fixture):
        """Verify returned figures can be customized after creation."""
        model, idata = mtcars_fixture

        # Test plot_predictions
        plot = plot_predictions(model, idata, "hp")
        assert isinstance(plot, Figure)
        plot.axes[0].set_title("Custom Title")
        plot.axes[0].set_xlabel("Horsepower")
        assert plot.axes[0].get_title() == "Custom Title"
        assert plot.axes[0].get_xlabel() == "Horsepower"

        # Test plot_comparisons
        plot = plot_comparisons(model, idata, "hp", "am")
        assert isinstance(plot, Figure)
        plot.axes[0].set_title("Custom Comparison")
        assert plot.axes[0].get_title() == "Custom Comparison"

        # Test plot_slopes
        plot = plot_slopes(model, idata, "hp", "am")
        assert isinstance(plot, Figure)
        plot.axes[0].set_title("Custom Slopes")
        assert plot.axes[0].get_title() == "Custom Slopes"

    def test_plot_accepts_matplotlib_targets(self, mtcars_fixture):
        model, idata = mtcars_fixture

        axes_figure = Figure()
        axes = axes_figure.subplots(1, 2, sharey=True)
        assert plot_predictions(model, idata, "hp", on=axes[0]) is axes_figure
        assert axes[0].get_xlabel() == "hp"
        assert axes[1].get_xlabel() == ""

        figure = Figure()
        assert plot_predictions(model, idata, "hp", on=figure) is figure
        assert figure.axes

        parent_figure = Figure()
        subfigure = parent_figure.subfigures()
        assert plot_predictions(model, idata, "hp", on=subfigure) is parent_figure
        assert subfigure.axes

    def test_plot_uses_active_matplotlib_style(self, mtcars_fixture):
        model, idata = mtcars_fixture

        with matplotlib.rc_context({"axes.facecolor": "#123456"}):
            figure = plot_predictions(model, idata, "hp")

        assert figure.axes[0].get_facecolor() == matplotlib.colors.to_rgba("#123456")

    def test_group_legend_stays_within_figure(self):
        """Figure-targeted seaborn legends are explicitly repositioned."""
        data = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0],
                "group": ["a", "a", "b", "b"],
                "estimate": [0.0, 1.0, 0.2, 1.2],
                "lower_94%": [-0.1, 0.9, 0.1, 1.1],
                "upper_94%": [0.1, 1.1, 0.3, 1.3],
            }
        )

        config = PlottingConfig.from_params(
            ["x", "group"], fig_kwargs={"theme": {"figure.figsize": (12, 4)}}
        )
        figure = plot(data, config)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        legend_bbox = figure.legends[0].get_window_extent(canvas.get_renderer())

        assert tuple(figure.get_size_inches()) == (12.0, 4.0)
        assert 0.8 * figure.bbox.width <= legend_bbox.x0
        assert legend_bbox.x1 <= figure.bbox.x1
        assert figure.bbox.y0 <= legend_bbox.y0
        assert legend_bbox.y1 <= figure.bbox.y1
        interval_patches = figure.legends[0].findobj(
            match=lambda artist: isinstance(artist, Rectangle)
        )
        assert len(interval_patches) == 2

    def test_legend_can_be_disabled_with_fig_kwargs(self):
        data = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0],
                "group": ["a", "a", "b", "b"],
                "estimate": [0.0, 1.0, 0.2, 1.2],
                "lower_94%": [-0.1, 0.9, 0.1, 1.1],
                "upper_94%": [0.1, 1.1, 0.3, 1.3],
            }
        )
        config = PlottingConfig.from_params(["x", "group"], fig_kwargs={"legend": False})

        figure = plot(data, config)

        assert not figure.legends

    def test_panel_order_follows_data_order(self):
        pigs = [4602, 8437, 4817]
        data = pd.DataFrame(
            {
                "Time": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                "Pig": pigs * 2,
                "estimate": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                "lower_94%": [0.9, 1.9, 2.9, 1.4, 2.4, 3.4],
                "upper_94%": [1.1, 2.1, 3.1, 1.6, 2.6, 3.6],
            }
        )
        config = PlottingConfig.from_params(
            ["Time", "Pig"], subplot_kwargs={"main": "Time", "panel": "Pig"}
        )

        figure = plot(data, config)

        assert [ax.get_title() for ax in figure.axes] == [str(pig) for pig in pigs]


class TestPredictions:
    """
    Tests the 'plot_predictions' function for different combinations of main, group,
    and panel variables.
    """

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    @pytest.mark.parametrize(
        "covariates",
        (
            "hp",  # Main variable is numeric
            "gear",  # Main variable is categorical
            ["hp"],  # Using list
            ["gear"],  # Using list
        ),
    )
    def test_basic(self, mtcars_fixture, covariates, target):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, target=target)
        assert isinstance(result, Figure)

    def test_binomial_predictions_use_one_trial(self, data_beetle, mock_pymc_sample):
        model = bmb.Model("p(y, n) ~ x", data_beetle, family="binomial")
        idata = model.fit(chains=2)

        result = predictions(model, idata, conditional="x")
        assert (result.summary["n"] == 1).all()
        assert isinstance(plot_predictions(model, idata, conditional="x"), Figure)

    def test_binomial_predictions_keep_literal_trials(self, data_beetle, mock_pymc_sample):
        model = bmb.Model("p(y, 62) ~ x", data_beetle, family="binomial")
        idata = model.fit(chains=2)

        result = predictions(model, idata, conditional="x")
        assert "n" not in result.summary

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    @pytest.mark.parametrize(
        "covariates",
        (
            ["hp", "wt"],  # Main: numeric. Group: numeric
            ["hp", "cyl"],  # Main: numeric. Group: categorical
            ["gear", "wt"],  # Main: categorical. Group: numeric
            ["gear", "cyl"],  # Main: categorical. Group: categorical
        ),
    )
    def test_with_groups(self, mtcars_fixture, covariates, target):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, target=target)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    @pytest.mark.parametrize(
        "covariates",
        (["hp", "cyl", "gear"], ["cyl", "hp", "gear"], ["cyl", "gear", "hp"]),
    )
    def test_with_group_and_panel(self, mtcars_fixture, covariates, target):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, covariates, target=target)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    @pytest.mark.parametrize(
        "conditional",
        [
            ({"hp": [110, 175], "am": [0, 1], "drat": [3, 4, 5]}),
            ({"hp": [150], "am": [1], "drat": [3, 4, 5]}),
        ],
    )
    def test_with_user_values(self, mtcars_fixture, conditional, target):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, conditional=conditional, target=target)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_predictions(model, idata, ["hp", "am", "drat"], average_by)
        assert isinstance(result, Figure)

        # unit level with average by covariates
        result = plot_predictions(model, idata, None, average_by)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    def test_fig_kwargs(self, mtcars_fixture, target):
        model, idata = mtcars_fixture
        result = plot_predictions(
            model,
            idata,
            ["hp", "cyl", "gear"],
            target=target,
            fig_kwargs={"sharey": True, "theme": {"font.size": 12}},
        )
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    def test_subplot_kwargs(self, mtcars_fixture, target):
        model, idata = mtcars_fixture
        result = plot_predictions(
            model,
            idata,
            ["hp", "drat"],
            target=target,
            subplot_kwargs={"main": "hp", "group": "drat", "panel": "drat"},
        )
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "mpg"])
    @pytest.mark.parametrize(
        "transforms",
        (
            {"mpg": np.log},
            {"hp": np.log},
            {"mpg": np.log, "hp": np.log},
        ),
    )
    def test_transforms(self, mtcars_fixture, transforms, target):
        model, idata = mtcars_fixture
        result = plot_predictions(model, idata, ["hp"], target=target, transforms=transforms)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("target", ["mean", "y"])
    def test_multiple_outputs_with_alias(self, target):
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
        initvals = {"Intercept_centered": 1 / y.mean()}

        # Without alias
        idata = model.fit(tune=100, draws=100, random_seed=1234, initvals=initvals)
        # Test default target
        result = plot_predictions(model, idata, "x", target=target)
        assert isinstance(result, Figure)
        # Test user supplied target argument
        result = plot_predictions(model, idata, "x", target="alpha")
        assert isinstance(result, Figure)

        # With alias
        alias = {"alpha": {"Intercept": "sd_intercept", "x": "sd_x", "alpha": "sd_alpha"}}
        model.set_alias(alias)
        idata = model.fit(tune=100, draws=100, random_seed=1234, initvals=initvals)

        # Test user supplied target argument
        result = plot_predictions(model, idata, "x", target="alpha")
        assert isinstance(result, Figure)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        result = plot_predictions(model, idata, ["Days", "Subject"])
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    def test_categorical_response_summary_keeps_all_categories(self, food_choice):
        model, idata = food_choice

        result = predictions(model, idata, conditional={"length": [30.0, 50.0, 70.0]})
        summary = result.summary

        assert len(summary) == 9
        assert (summary.groupby("length")["choice_dim"].nunique() == 3).all()
        np.testing.assert_allclose(summary.groupby("length")["estimate"].sum(), 1)

    @pytest.mark.parametrize(
        "group, label", [(None, "choice"), ("choice", "choice"), ("choice_dim", "choice_dim")]
    )
    def test_categorical_response_uses_response_name_in_plot(self, food_choice, group, label):
        model, idata = food_choice
        subplot_kwargs = None
        if group is not None:
            subplot_kwargs = {"main": "length", "group": group}

        figure = plot_predictions(model, idata, "length", subplot_kwargs=subplot_kwargs)

        assert figure.legends[0].get_title().get_text() == label

    def test_term_transformations(self, formulae_transform, nonformulae_transform):
        model, idata = formulae_transform

        # Test that the plot works with a formulae transformation
        result = plot_predictions(model, idata, ["x2", "x1"])
        assert isinstance(result, Figure)

        model, idata = nonformulae_transform

        # Test that the plot works with a non-formulae transformation
        result = plot_predictions(model, idata, "x1")
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    def test_distributional_target(self, distributional_fixture):
        model, idata = distributional_fixture
        result = plot_predictions(model, idata, "x", target="alpha")
        assert isinstance(result, Figure)

    def test_integer_predictor(self, integer_data_fixture):
        model, idata = integer_data_fixture
        result = plot_predictions(model, idata, "x_int")
        assert isinstance(result, Figure)


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
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize(
        "conditional",
        [["am", "drat", "gear"], ["drat", "am", "gear"], ["drat", "gear", "am"]],
    )
    def test_with_group_and_panel(self, mtcars_fixture, conditional):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", conditional)
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize(
        "contrast, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(self, mtcars_fixture, contrast, conditional, subplot_kwargs):
        model, idata = mtcars_fixture
        result = plot_comparisons(
            model, idata, contrast, conditional, subplot_kwargs=subplot_kwargs
        )
        assert isinstance(result, Figure)

    @pytest.mark.parametrize(
        "contrast, conditional, transforms",
        [
            ("drat", ["hp", "am"], {"hp": np.log}),  # transform main numeric
            ("drat", ["hp", "am"], {"mpg": np.log}),  # transform response
        ],
    )
    def test_transforms(self, mtcars_fixture, contrast, conditional, transforms):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, contrast, conditional, transforms=transforms)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_comparisons(model, idata, "hp", ["am", "drat"], average_by)
        assert isinstance(result, Figure)

        # unit level with average by
        result = plot_comparisons(model, idata, "hp", None, average_by)
        assert isinstance(result, Figure)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        result = plot_comparisons(model, idata, "Days", "Subject")
        assert isinstance(result, Figure)
        # user passed values seen in observed data
        result = plot_comparisons(
            model,
            idata,
            contrast={"Days": [2, 4]},
            conditional={"Subject": [308, 335, 352, 372]},
        )
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("comparison", ["ratio", "lift"])
    def test_comparison_types(self, mtcars_fixture, comparison):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", "am", comparison=comparison)
        assert isinstance(result, Figure)

    def test_target_response(self, mtcars_fixture):
        model, idata = mtcars_fixture
        result = plot_comparisons(model, idata, "hp", "am", target="mpg")
        assert isinstance(result, Figure)

    def test_custom_callable(self, mtcars_fixture):
        model, idata = mtcars_fixture

        def my_comparison(reference, contrast):
            return contrast - 2 * reference

        result = plot_comparisons(model, idata, "hp", "am", comparison=my_comparison)
        assert isinstance(result, Figure)

    def test_integer_contrast(self, integer_data_fixture):
        model, idata = integer_data_fixture
        result = plot_comparisons(model, idata, "x_int", "x_float")
        assert isinstance(result, Figure)


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
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize(
        "conditional",
        [["am", "drat", "gear"], ["drat", "am", "gear"], ["drat", "gear", "am"]],
    )
    def test_with_group_and_panel(self, mtcars_fixture, conditional):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, "hp", conditional)
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("slope", ["dydx", "dyex", "eyex", "eydx"])
    def test_elasticity(self, mtcars_fixture, slope):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, "hp", "drat", slope=slope)
        assert isinstance(result, Figure)

    @pytest.mark.parametrize(
        "wrt, conditional, subplot_kwargs",
        [("drat", ["hp", "am"], {"main": "hp", "group": "am", "panel": "am"})],
    )
    def test_subplot_kwargs(self, mtcars_fixture, wrt, conditional, subplot_kwargs):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, wrt, conditional, subplot_kwargs=subplot_kwargs)
        assert isinstance(result, Figure)

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
        assert isinstance(result, Figure)

    @pytest.mark.parametrize("average_by", ["am", "drat", ["am", "drat"]])
    def test_average_by(self, mtcars_fixture, average_by):
        model, idata = mtcars_fixture

        # grid of values with average_by
        result = plot_slopes(model, idata, "hp", ["am", "drat"], average_by)
        assert isinstance(result, Figure)

        # unit level with average by
        result = plot_slopes(model, idata, "hp", None, average_by)
        assert isinstance(result, Figure)

    def test_group_effects(self, sleep_study):
        model, idata = sleep_study

        result = plot_slopes(model, idata, "Days", "Subject")
        assert isinstance(result, Figure)
        # user passed values seen in observed data
        result = plot_slopes(model, idata, wrt={"Days": 2}, conditional={"Subject": [308]})
        assert isinstance(result, Figure)

    def test_categorical_response(self, food_choice):
        model, idata = food_choice
        # Only numeric wrt is supported
        result = plot_slopes(model, idata, "length", "sex")
        assert isinstance(result, Figure)

    def test_target_response(self, mtcars_fixture):
        model, idata = mtcars_fixture
        result = plot_slopes(model, idata, "hp", "am", target="mpg")
        assert isinstance(result, Figure)

    def test_custom_callable(self, mtcars_fixture):
        model, idata = mtcars_fixture

        def my_slope(derivative, x, y):
            return derivative * 2

        result = plot_slopes(model, idata, "hp", "drat", slope=my_slope)
        assert isinstance(result, Figure)

    def test_integer_wrt(self, integer_data_fixture):
        model, idata = integer_data_fixture
        result = plot_slopes(model, idata, "x_int", "x_float")
        assert isinstance(result, Figure)
