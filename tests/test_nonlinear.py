import formulae as fm
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy.special import erf, erfc, expit, logit, ndtr, ndtri

from bambi.backend.pymc.nonlinear import build_nonlinear_predictor, evaluate_expression
from bambi.nonlinear import (
    NonlinearExpression,
    parameter_dependency_order,
    split_predictor,
)
from bambi.parameters import ConditionalCoefficient


def test_nonlinear_coefficient_rejects_incompatible_input_dimensions():
    parameter = ConditionalCoefficient(
        "curve",
        design=None,
        priors={},
        spec=None,
        expression=NonlinearExpression.parse("a + b"),
    )
    with pm.Model(coords={"__obs__": range(3), "first": range(2), "second": range(2)}) as model:
        values = {
            "a": pm.Data("a", [1.0, 2.0], dims="first"),
            "b": pm.Data("b", [3.0, 4.0], dims="second"),
        }
        with pytest.raises(ValueError, match="incompatible input dimensions"):
            build_nonlinear_predictor(parameter, values, pd.DataFrame(), model, dims=None)


def test_parameter_dependency_order_is_stable_and_topological():
    dependencies = {
        "sigma": ("mu", "sigma_y"),
        "mu": ("p_angle", "p_distance"),
        "sigma_y": (),
        "p_distance": (),
        "p_angle": (),
    }

    result = parameter_dependency_order(
        dependencies, ("sigma", "mu", "sigma_y", "p_distance", "p_angle")
    )

    assert result == ("p_distance", "p_angle", "mu", "sigma_y", "sigma")


@pytest.mark.parametrize(
    ("dependencies", "message"),
    [
        ({"mu": ("mu",)}, "cannot depend on themselves: \\['mu'\\]"),
        ({"mu": ("sigma",), "sigma": ("mu",)}, "mu -> sigma -> mu"),
        (
            {"mu": ("scale",), "scale": ("sigma",), "sigma": ("mu",)},
            "mu -> scale -> sigma -> mu",
        ),
    ],
)
def test_parameter_dependency_order_errors(dependencies, message):
    with pytest.raises(ValueError, match=message):
        parameter_dependency_order(dependencies, tuple(dependencies))


def evaluate_nonlinear_expression(expression, data):
    values = {name: pt.as_tensor_variable(data[name].to_numpy()) for name in data}
    return evaluate_expression(NonlinearExpression.parse(expression).root, values).eval()


def test_supported_expression_operations():
    data = pd.DataFrame({"x": [1.0, 2.0], "a": -3.0, "b": np.exp(2.0)})
    result = evaluate_nonlinear_expression("sqrt(a ** 2) + log(b) / x", data)
    np.testing.assert_allclose(result, [5.0, 4.0])


@pytest.mark.parametrize(
    "function",
    [
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "log1p",
        "expm1",
        "softplus",
        "erf",
        "erfc",
    ],
)
def test_supported_unary_math_functions(function):
    x = np.array([0.2, 0.5])
    z = np.array([1.2, 1.5])
    references = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
        "arcsinh": np.arcsinh,
        "arccosh": np.arccosh,
        "arctanh": np.arctanh,
        "log1p": np.log1p,
        "expm1": np.expm1,
        "softplus": lambda value: np.logaddexp(0, value),
        "erf": erf,
        "erfc": erfc,
    }
    argument_name = "z" if function in {"acosh", "arccosh"} else "x"
    argument = z if argument_name == "z" else x

    result = evaluate_nonlinear_expression(
        f"{function}({argument_name})", pd.DataFrame({"x": x, "z": z})
    )

    expected = references[function](argument)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("function", ["atan2", "arctan2"])
def test_supported_two_argument_functions(function):
    x = np.array([0.2, 0.5])
    z = np.array([1.2, 1.5])

    result = evaluate_nonlinear_expression(f"{function}(x, z)", pd.DataFrame({"x": x, "z": z}))

    np.testing.assert_allclose(result, np.arctan2(x, z))


@pytest.mark.parametrize(
    "function",
    [
        "logit",
        "invlogit",
        "expit",
        "normal_cdf",
        "norm_cdf",
        "normal_ppf",
        "norm_ppf",
        "probit",
        "invprobit",
        "cloglog",
        "invcloglog",
    ],
)
def test_supported_probability_transforms(function):
    x = np.array([0.2, 0.5])
    references = {
        "logit": logit,
        "invlogit": expit,
        "expit": expit,
        "normal_cdf": ndtr,
        "norm_cdf": ndtr,
        "normal_ppf": ndtri,
        "norm_ppf": ndtri,
        "probit": ndtri,
        "invprobit": ndtr,
        "cloglog": lambda value: np.log(-np.log1p(-value)),
        "invcloglog": lambda value: -np.expm1(-np.exp(value)),
    }

    result = evaluate_nonlinear_expression(f"{function}(x)", pd.DataFrame({"x": x}))

    np.testing.assert_allclose(result, references[function](x))


@pytest.mark.parametrize(
    "expression, message",
    [
        ("atan2(x)", "requires exactly 2 positional arguments"),
        ("atan2(x, x, x)", "requires exactly 2 positional arguments"),
        ("atan2(x=x, y=x)", "requires exactly 2 positional arguments"),
        ("sin(x, x)", "requires exactly 1 positional argument"),
    ],
)
def test_nonlinear_function_arity(expression, message):
    with pytest.raises(ValueError, match=message):
        NonlinearExpression.parse(expression)


def test_dependency_check_ignores_string_literals():
    source = "C(category, Treatment(reference='a'))"
    linear, expression = split_predictor(source, ("a",))
    assert linear == source
    assert expression is None


@pytest.mark.parametrize(
    "expression",
    [
        "a + np.exp(x)",
        "a + x[0]",
        "a + (x > 0)",
        "a + [x]",
        "a + True",
        "a + 'x'",
        "a % x",
        "a // x",
        "a + (lambda: x)()",
        "a + exp(x, x)",
        "a + exp(value=x)",
    ],
)
def test_unsupported_expression_syntax_is_rejected(expression):
    with pytest.raises(ValueError, match="Nonlinear|nonlinear|Unsupported"):
        NonlinearExpression.parse(expression)


@pytest.mark.parametrize(
    "source, linear_formula, nlexpr",
    [
        ("nl(exp(b * x) + x) + z", "z", "exp(b * x) + x"),
        ("z - exp(b * x) + (1 + x | group)", "z + (1 + x | group)", "-exp(b * x)"),
        ("nl(exp(b * x) - 1) + z", "z", "exp(b * x) - 1"),
        ("nl(exp(b * x) + x) - log1p(b ** 2) + z", "z", "exp(b * x) + x - log1p(b**2)"),
        ("(exp(b * x) + x) + z", "x + z", "exp(b * x)"),
        ("0 + nl(b * x) + x:z", "0 + x:z", "b * x"),
        ("nl(b * x) - 1", "-1", "b * x"),
        ("nl(x + 1)", "1", "x + 1"),
        ("a + b * x", None, "a + b * x"),
        ("a + b * x + 2", None, "a + b * x + 2"),
        ("a + b * x - 0.5", None, "a + b * x - 0.5"),
        ("a * x + 1e-3", None, "a * x + 1e-3"),
        ("nl(a * x + 1e-3) + z", "z", "a * x + 1e-3"),
        ("(nl(b * x) + (1 | group))", "(1 | group)", "b * x"),
    ],
)
def test_split_linear_formula_and_nlexpr(source, linear_formula, nlexpr):
    actual_linear_formula, actual_nlexpr = split_predictor(source, ("a", "b"))
    if linear_formula is None:
        assert actual_linear_formula is None
    else:
        assert actual_linear_formula.replace(" ", "") == linear_formula.replace(" ", "")
    values = {"a": 0.5, "b": 0.3, "x": pt.as_tensor_variable([0.2, 0.8])}
    actual = evaluate_expression(actual_nlexpr.root, values).eval()
    expected = evaluate_expression(NonlinearExpression.parse(nlexpr).root, values).eval()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "source", ["nl()", "nl(a, x)", "nl(a=x)", "2 * nl(a*x)", "nl(a*x):z", "nl(nl(a*x))"]
)
def test_invalid_nl_wrapper(source):
    with pytest.raises(ValueError):
        split_predictor(source, ("a",))


@pytest.mark.parametrize("predictor", ["a*x", "nl(a*x)"])
@pytest.mark.parametrize("linear", ["z - (x - w)", "z - (x + w - 1)", "(z - (x - w))"])
def test_split_predictor_preserves_formula_subtraction(predictor, linear):
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "z": [2.0, 4.0, 1.0, 3.0],
            "w": [4.0, 1.0, 3.0, 2.0],
        }
    )
    actual_linear, expression = split_predictor(f"{predictor} + {linear}", ("a",))
    assert expression is not None
    expected = fm.design_matrices(f"y ~ {linear}", data).common.terms
    actual = fm.design_matrices(f"y ~ {actual_linear}", data).common.terms
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name].data, expected[name].data)


@pytest.mark.parametrize("predictor", ["a*x", "nl(a*x)"])
def test_split_predictor_preserves_subtraction_of_mixed_group(predictor):
    linear, expression = split_predictor(f"z - ({predictor} + x - w)", ("a",))
    assert linear.replace(" ", "") == "z-(x-w)"
    values = {"a": pt.as_tensor_variable(1.0), "x": pt.as_tensor_variable(2.0)}
    np.testing.assert_allclose(evaluate_expression(expression.root, values).eval(), -2.0)


@pytest.mark.parametrize("name", ["0x10", "0XFF", "0o17", "0b10", "123abc", "1e"])
@pytest.mark.parametrize("source", ["a*x + {name}", "nl(a*x + {name})", "a*x + z + {name}"])
def test_digit_prefixed_variable_names_are_rejected(name, source):
    with pytest.raises(ValueError, match=f"Invalid variable name: '{name}'"):
        split_predictor(source.format(name=name), ("a",))


@pytest.mark.parametrize("name", ["0x10", "0XFF", "0o17", "0b10", "123abc", "1e"])
def test_nonlinear_expression_rejects_digit_prefixed_names(name):
    with pytest.raises(ValueError, match=f"Invalid variable name: '{name}'"):
        NonlinearExpression.parse(f"a*x + {name}")


@pytest.mark.parametrize(
    "literal", ["0", "1", "15", "1.5", "0.3", ".5", "1.", "1e-3", "1E+3", "1_000.25"]
)
@pytest.mark.parametrize("wrapper", [False, True])
def test_decimal_numeric_literals(literal, wrapper):
    assert NonlinearExpression.parse(literal).root.value == float(literal)
    source = f"a*x + {literal}"
    if wrapper:
        source = f"nl({source})"
    _, expression = split_predictor(source, ("a",))
    values = {"a": pt.as_tensor_variable(1.0), "x": pt.as_tensor_variable(2.0)}
    result = evaluate_expression(expression.root, values).eval()
    np.testing.assert_allclose(result, 2 + float(literal))
