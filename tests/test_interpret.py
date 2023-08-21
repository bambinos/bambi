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
    data = bmb.load_data('mtcars')
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


@pytest.mark.parametrize("return_posterior", [True, False])
def test_return_posterior(mtcars, return_posterior):
    model, idata = mtcars

    bmb.interpret.predictions(
        model, idata, ["hp", "wt"], return_posterior=return_posterior
        )
    bmb.interpret.comparisons(
        model, idata, "hp", "wt", return_posterior=return_posterior
    )
    bmb.interpret.slopes(
        model, idata, "hp", "wt", return_posterior=return_posterior
    )
