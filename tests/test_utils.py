import pytest

import numpy as np
import pandas as pd

from bambi.utils import listify
from bambi.backend.pymc import probit, cloglog
from bambi.transformations import censored, truncated


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
    assert (x[:, -1] == np.array([0, 1, 2, -1, 0])).all()

    x = censored(df["x"], df["y"], df["status"])
    assert x.shape == (5, 3)
    assert (x[:, -1] == np.array([0, 1, 2, -1, 0])).all()

    # Statuses are not the expected
    with pytest.raises(AssertionError, match="Statuses must be in"):
        censored(df_bad["x"], df_bad["status"])

    # Upper bound is not always larger than lower bound
    df_bad = pd.DataFrame({"l": [1, 2], "r": [1, 1], "status": ["foo", "bar"]})

    with pytest.raises(AssertionError, match="Upper bound must be larger than lower bound"):
        censored(df_bad["l"], df_bad["r"], df_bad["status"])

    # Bad number of arguments
    with pytest.raises(ValueError, match="needs 2 or 3 argument values"):
        censored(df["x"])

    with pytest.raises(ValueError, match="needs 2 or 3 argument values"):
        censored(df["x"], df["x"], df["x"], df["x"])


def test_truncated():
    x = np.array([-3, -2, -1, 0, 0, 0, 1, 1, 2, 3])
    lower = -5
    upper = 4.5
    lower_arr = np.array([-5] * 6 + [-4] * 4)
    upper_arr = np.array([5] * 6 + [5.35] * 4)

    # Arguments and expected outcomes
    iterable = {
        "lower": (lower, None, lower, lower_arr, None, lower_arr),
        "upper": (None, upper, upper, None, upper_arr, upper_arr),
        "elower": (lower, -np.inf, lower, lower_arr, -np.inf, lower_arr),
        "eupper": (np.inf, upper, upper, np.inf, upper_arr, upper_arr),
    }

    for l, u, el, eu in zip(*iterable.values()):
        result = truncated(x, lb=l, ub=u)
        assert result.shape == (10, 3)
        assert (result[:, 0] == x).all()
        assert (result[:, 1] == el).all()
        assert (result[:, 2] == eu).all()

    with pytest.raises(ValueError, match="'lb' and 'ub' cannot both be None"):
        truncated(x)

    with pytest.raises(ValueError, match="'truncated' only works with 1-dimensional arrays"):
        truncated(np.column_stack([x, x]))

    with pytest.raises(AssertionError, match="The length of 'lb' must be equal to the one of 'x'"):
        truncated(x, np.array([-5, -6]))

    with pytest.raises(AssertionError, match="The length of 'ub' must be equal to the one of 'x'"):
        truncated(x, ub=np.array([5, 6]))

    with pytest.raises(ValueError, match="'lb' must be 0 or 1 dimensional."):
        truncated(x, np.column_stack([lower_arr, lower_arr]))

    with pytest.raises(ValueError, match="'ub' must be 0 or 1 dimensional."):
        truncated(x, ub=np.column_stack([upper_arr, upper_arr]))
