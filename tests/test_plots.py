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


@pytest.mark.parametrize("pps", [False, True])
def test_basic(mtcars, pps):
    model, idata = mtcars

    # Using dictionary
    # Horizontal variable is numeric
    plot_cap(model, idata, {"horizontal": "hp"}, pps=pps)

    # Horizontal variable is categorical
    plot_cap(model, idata, {"horizontal": "gear"}, pps=pps)

    # Using list
    plot_cap(model, idata, ["hp"], pps=pps)
    plot_cap(model, idata, ["gear"], pps=pps)


@pytest.mark.parametrize("pps", [False, True])
def test_with_groups(mtcars, pps):
    model, idata = mtcars

    # Dictionary
    # Horizontal: numeric. Group: numeric
    plot_cap(model, idata, {"horizontal": "hp", "color": "wt"}, pps=pps)

    # Horizontal: numeric. Group: categorical
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl"}, pps=pps)

    # Horizontal: categorical. Group: numeric
    plot_cap(model, idata, {"horizontal": "gear", "color": "wt"}, pps=pps)

    # Horizontal: categorical. Group: categorical
    plot_cap(model, idata, {"horizontal": "gear", "color": "cyl"}, pps=pps)

    # List
    plot_cap(model, idata, ["hp", "wt"], pps=pps)
    plot_cap(model, idata, ["hp", "cyl"], pps=pps)
    plot_cap(model, idata, ["gear", "wt"], pps=pps)
    plot_cap(model, idata, ["gear", "cyl"], pps=pps)


@pytest.mark.parametrize("pps", [False, True])
def test_with_panel(mtcars, pps):
    model, idata = mtcars

    # Dictionary is the only possibility
    # Horizontal: numeric. Group: numeric
    plot_cap(model, idata, {"horizontal": "hp", "panel": "wt"}, pps=pps)

    # Horizontal: numeric. Group: categorical
    plot_cap(model, idata, {"horizontal": "hp", "panel": "cyl"}, pps=pps)

    # Horizontal: categorical. Group: numeric
    plot_cap(model, idata, {"horizontal": "gear", "panel": "wt"}, pps=pps)

    # Horizontal: categorical. Group: categorical
    plot_cap(model, idata, {"horizontal": "gear", "panel": "cyl"}, pps=pps)


@pytest.mark.parametrize("pps", [False, True])
def test_with_group_and_panel(mtcars, pps):
    model, idata = mtcars

    # Dictionary
    plot_cap(model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, pps=pps)
    plot_cap(model, idata, {"horizontal": "cyl", "color": "hp", "panel": "gear"}, pps=pps)
    plot_cap(model, idata, {"horizontal": "cyl", "color": "gear", "panel": "hp"}, pps=pps)

    # List
    plot_cap(model, idata, ["hp", "cyl", "gear"], pps=pps)
    plot_cap(model, idata, ["cyl", "hp", "gear"], pps=pps)
    plot_cap(model, idata, ["cyl", "gear", "hp"], pps=pps)


@pytest.mark.parametrize("pps", [False, True])
def test_ax(mtcars, pps):
    model, idata = mtcars
    fig, ax = plt.subplots()
    fig_r, ax_r = plot_cap(model, idata, ["hp"], pps=pps, ax=ax)

    assert isinstance(ax_r, np.ndarray)
    assert fig is fig_r
    assert ax is ax_r[0]


@pytest.mark.parametrize("pps", [False, True])
def test_fig_kwargs(mtcars, pps):
    model, idata = mtcars
    plot_cap(
        model,
        idata,
        {"horizontal": "hp", "color": "cyl", "panel": "gear"},
        pps=pps,
        fig_kwargs={"figsize": (15, 5), "dpi": 120, "sharey": True},
    )


@pytest.mark.parametrize("pps", [False, True])
def test_use_hdi(mtcars, pps):
    model, idata = mtcars
    plot_cap(
        model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, pps=pps, use_hdi=False
    )


@pytest.mark.parametrize("pps", [False, True])
def test_hdi_prob(mtcars, pps):
    model, idata = mtcars
    plot_cap(
        model, idata, {"horizontal": "hp", "color": "cyl", "panel": "gear"}, pps=pps, hdi_prob=0.9
    )

    with pytest.raises(
        ValueError, match="'hdi_prob' must be greater than 0 and smaller than 1. It is 1.1."
    ):
        plot_cap(
            model,
            idata,
            {"horizontal": "hp", "color": "cyl", "panel": "gear"},
            pps=pps,
            hdi_prob=1.1,
        )

    with pytest.raises(
        ValueError, match="'hdi_prob' must be greater than 0 and smaller than 1. It is -0.1."
    ):
        plot_cap(
            model,
            idata,
            {"horizontal": "hp", "color": "cyl", "panel": "gear"},
            pps=pps,
            hdi_prob=-0.1,
        )


@pytest.mark.parametrize("pps", [False, True])
def test_legend(mtcars, pps):
    model, idata = mtcars
    plot_cap(model, idata, ["hp"], pps=pps, legend=False)


@pytest.mark.parametrize("pps", [False, True])
def test_transforms(mtcars, pps):
    model, idata = mtcars

    transforms = {"mpg": np.log}
    plot_cap(model, idata, ["hp"], pps=pps, transforms=transforms)

    transforms = {"hp": np.log}
    plot_cap(model, idata, ["hp"], pps=pps, transforms=transforms)

    transforms = {"mpg": np.log, "hp": np.log}
    plot_cap(model, idata, ["hp"], pps=pps, transforms=transforms)


@pytest.mark.parametrize("pps", [False, True])
def test_multiple_outputs(pps):
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
    plot_cap(model, idata, "x", pps=pps)
    # Test user supplied target argument
    plot_cap(model, idata, "x", "alpha", pps=pps)
