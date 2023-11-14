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


@pytest.mark.parametrize("return_idata", [True, False])
def test_return_idata(mtcars, return_idata):
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
    