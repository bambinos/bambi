from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import bambi as bmb
from bambi.plots import plot_cap


@pytest.fixture(scope="module")
def mtcars():
    data = pd.read_csv(join(dirname(__file__), "data", "mtcars.csv"))
    data["cyl"] = data["cyl"].replace({4: "low", 6: "medium", 8: "high"})
    data["cyl"] = pd.Categorical(data["cyl"], categories=["low", "medium", "high"], ordered=True)
    data["gear"] = data["gear"].replace({3: "A", 4: "B", 5: "C"})
    model = bmb.Model("mpg ~ 0 + hp * wt + cyl + gear", data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


# Improvement:
# * Test the actual plots are what we are indeed the desired result.
# * Test using the dictionary and the list gives the same plot


def test_basic(mtcars):
    model, idata = mtcars

    # Using dictionary
    # Horizontal variable is numeric
    plot_cap(model, idata, {"horizontal": "hp"})

    # Horizontal variable is categorical
    plot_cap(model, idata, {"horizontal": "gear"})

    # Using list
    plot_cap(model, idata, ["hp"])
    plot_cap(model, idata, ["gear"])


def test_with_groups(mtcars):
    model, idata = mtcars

    # Dictionary
    # Horizontal: numeric. Group: numeric
    plot_cap(model, idata, {"horizontal": "hp", "color": "wt"})

    # Horizontal: numeric. Group: categorical
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl"})

    # Horizontal: categorical. Group: numeric
    plot_cap(model, idata, {"horizontal": "gear", "color": "wt"})

    # Horizontal: categorical. Group: categorical
    plot_cap(model, idata, {"horizontal": "gear", "color": "cyl"})

    # List
    plot_cap(model, idata, ["hp", "wt"])
    plot_cap(model, idata, ["hp", "cyl"])
    plot_cap(model, idata, ["gear", "wt"])
    plot_cap(model, idata, ["gear", "cyl"])


def test_with_panel(mtcars):
    model, idata = mtcars

    # Dictionary is the only possibility
    # Horizontal: numeric. Group: numeric
    plot_cap(model, idata, {"horizontal": "hp", "panel": "wt"})

    # Horizontal: numeric. Group: categorical
    plot_cap(model, idata, {"horizontal": "hp", "panel": "cyl"})

    # Horizontal: categorical. Group: numeric
    plot_cap(model, idata, {"horizontal": "gear", "panel": "wt"})

    # Horizontal: categorical. Group: categorical
    plot_cap(model, idata, {"horizontal": "gear", "panel": "cyl"})


def test_with_group_and_panel(mtcars):
    model, idata = mtcars

    # Dictionary
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"})
    plot_cap(model, idata, {"horizontal": "cyl", "color": "hp", "panel": "gear"})
    plot_cap(model, idata, {"horizontal": "cyl", "color": "gear", "panel": "hp"})

    # List
    plot_cap(model, idata, ["hp", "cyl", "gear"])
    plot_cap(model, idata, ["cyl", "hp", "gear"])
    plot_cap(model, idata, ["cyl", "gear", "hp"])


def test_ax(mtcars):
    model, idata = mtcars
    fig, ax = plt.subplots()
    fig_r, ax_r = plot_cap(model, idata, ["hp"], ax=ax)

    assert isinstance(ax_r, np.ndarray)
    assert fig is fig_r
    assert ax is ax_r[0]


def test_fig_kwargs(mtcars):
    model, idata = mtcars
    plot_cap(
        model,
        idata,
        {"horizontal": "hp", "color": "cyl", "panel": "gear"},
        fig_kwargs={"figsize": (15, 5), "dpi": 120, "sharey": True},
    )


def test_use_hdi(mtcars):
    model, idata = mtcars
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, use_hdi=False)


def test_hdi_prob(mtcars):
    model, idata = mtcars
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, hdi_prob=0.9)

    with pytest.raises(
        ValueError, match="'hdi_prob' must be greater than 0 and smaller than 1. It is 1.1."
    ):
        plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, hdi_prob=1.1)

    with pytest.raises(
        ValueError, match="'hdi_prob' must be greater than 0 and smaller than 1. It is -0.1."
    ):
        plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, hdi_prob=-0.1)


def test_legend(mtcars):
    model, idata = mtcars
    plot_cap(model, idata, ["hp"], legend=False)


def test_transforms(mtcars):
    model, idata = mtcars

    transforms = {"mpg": np.log}
    plot_cap(model, idata, ["hp"], transforms=transforms)

    transforms = {"hp": np.log}
    plot_cap(model, idata, ["hp"], transforms=transforms)

    transforms = {"mpg": np.log, "hp": np.log}
    plot_cap(model, idata, ["hp"], transforms=transforms)


def test_multiple_outputs():
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
    idata = model.fit(tune=100, draws=100, random_seed=1234)
    # Test default target
    plot_cap(model, idata, "x")
    # Test user supplied target argument
    plot_cap(model, idata, "x", "alpha")
