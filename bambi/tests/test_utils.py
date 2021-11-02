import numpy as np

from bambi.utils import listify
from bambi.backend.pymc import probit, cloglog


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
