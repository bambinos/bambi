import pytest

import bambi as bmb


def test_regular_formula():
    f1 = bmb.Formula("y ~ x1 + x2")
    assert f1.main == "y ~ x1 + x2"
    assert f1.additionals == tuple()
    assert f1.additionals_lhs == list()


def test_additional_empty_response():
    with pytest.raises(ValueError, match="Additional formulas must contain a response name"):
        bmb.Formula("y ~ x1", "x1")


def test_additional_call_response():
    with pytest.raises(ValueError, match="The response must be a name"):
        bmb.Formula("y ~ x1", "log(sigma) ~ x1")


def test_access_additional_names():
    f1 = bmb.Formula("y ~ x")
    f2 = bmb.Formula("y ~ x1", "sigma ~ 1", "gamma ~ x")

    assert f1.additionals_lhs == []
    assert f2.additionals_lhs == ["sigma", "gamma"]


def test_formula_str():
    f1 = bmb.Formula("y ~ x")
    f2 = bmb.Formula("y ~ x", "sigma ~ 1", "gamma ~ x")

    assert str(f1) == "Formula(y ~ x)"
    assert str(f2) == "Formula(y ~ x, sigma ~ 1, gamma ~ x)"


def test_formula_repr():
    f1 = bmb.Formula("y ~ x")
    f2 = bmb.Formula("y ~ x", "sigma ~ 1", "gamma ~ x")

    assert repr(f1) == "Formula('y ~ x')"
    assert repr(f2) == "Formula('y ~ x', 'sigma ~ 1', 'gamma ~ x')"
