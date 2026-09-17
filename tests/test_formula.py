import pytest

import bambi as bmb


def test_regular_formula():
    f1 = bmb.Formula("y ~ x1 + x2")
    assert f1.main == "y ~ x1 + x2"
    assert f1.additionals == tuple()
    assert f1.additionals_lhs == list()
    assert f1.nlpars == tuple()


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


def test_empty_nlpars_preserves_regular_formula_representation():
    formula = bmb.Formula("y ~ x", nlpars=[])

    assert formula.nlpars == ()
    assert str(formula) == "Formula(y ~ x)"
    assert repr(formula) == "Formula('y ~ x')"


def test_nonlinear_formula_repr():
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + z", nlpars=["a", "b"])

    assert formula.nlpars == ("a", "b")
    assert str(formula) == "Formula(y ~ a + b * x, a ~ 1 + z, nlpars=('a', 'b'))"
    assert repr(formula) == "Formula('y ~ a + b * x', 'a ~ 1 + z', nlpars=('a', 'b'))"


@pytest.mark.parametrize("nlpars", ["a", {"a"}, 1])
def test_nlpars_rejects_non_sequences(nlpars):
    with pytest.raises(TypeError, match="list or tuple"):
        bmb.Formula("y ~ a", nlpars=nlpars)


@pytest.mark.parametrize("nlpars", [["a", 1], ["a-b"], [""], ["for"]])
def test_nlpars_rejects_invalid_entries(nlpars):
    with pytest.raises(ValueError, match="valid Python identifiers"):
        bmb.Formula("y ~ a", nlpars=nlpars)


def test_nlpars_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate nonlinear parameter name"):
        bmb.Formula("y ~ a", nlpars=("a", "a"))


def test_nonlinear_keyword_is_not_supported():
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonlinear'"):
        bmb.Formula("y ~ a", nonlinear=True)


@pytest.mark.parametrize(
    "additionals",
    [
        ("a ~ 1", "a ~ x"),
        ("sigma ~ z", "sigma ~ 1"),
    ],
)
def test_nonlinear_formula_rejects_duplicate_parameter_formulas(additionals):
    with pytest.raises(ValueError, match="Duplicate parameter formula"):
        bmb.Formula("y ~ a", *additionals, nlpars=("a",))
