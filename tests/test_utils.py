import pytest

import numpy as np
import pandas as pd

import bambi as bmb
from bambi.utils import listify
from bambi.backend.pymc import probit, cloglog
from bambi.transformations import censored

def test_listify():
    assert listify(None) == []
    assert listify([1, 2, 3]) == [1, 2, 3]
    assert listify("giraffe") == ["giraffe"]


def test_probit():
    x = probit(np.random.normal(scale=10000, size=100)).eval()
    assert (x > 0).all() and (x < 1).all()


def test_cloglog():
    x = cloglog(np.random.normal(scale=10000, size=100)).eval()
    assert (x > 0).all() and (x < 1).all()


@pytest.mark.skip(reason="Censored still not ported")
def test_censored():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 3, 4, 5, 6],
            "status": ["none", "right", "interval", "left", "none"],
        }
    )

    df_bad = pd.DataFrame({"x": [1, 2], "status": ["foo", "bar"]})

    x = censored(df["x"], df["status"])
    assert x.shape == (5, 2)

    x = censored(df["x"], df["y"], df["status"])
    assert x.shape == (5, 3)

    with pytest.raises(AssertionError):
        censored(df_bad["x"], df_bad["status"])
