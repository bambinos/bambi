import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.special import erf, erfc, expit, logit, ndtr, ndtri

import bambi as bmb
from bambi.backend.pymc.nonlinear import evaluate_expression
from bambi.nonlinear import (
    NonlinearExpression,
    parameter_dependency_order,
    resolve_nonlinear_symbols,
    split_predictor,
)

from helpers import assert_ip_dlogp, linear_data, normal_prior


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
        ({"mu": ("missing",)}, "Unknown nonlinear parameter reference.*missing"),
    ],
)
def test_parameter_dependency_order_errors(dependencies, message):
    with pytest.raises(ValueError, match=message):
        parameter_dependency_order(dependencies)


def test_resolve_nonlinear_symbols_separates_parameters_and_data():
    expression = NonlinearExpression.parse("sqrt(mu * (1 - mu) / attempts + sigma_y ** 2)")
    data = pd.DataFrame({"attempts": [10, 20]})

    dependencies, data_names = resolve_nonlinear_symbols(
        expression, ("mu", "sigma", "sigma_y"), data
    )

    assert dependencies == ("mu", "sigma_y")
    assert data_names == ("attempts",)


def test_resolve_nonlinear_symbols_rejects_ambiguous_name():
    expression = NonlinearExpression.parse("mu + x")
    data = pd.DataFrame({"mu": [0.2], "x": [1.0]})

    with pytest.raises(ValueError, match="both modeled parameters and data columns.*mu"):
        resolve_nonlinear_symbols(expression, ("mu",), data)


def evaluate_nonlinear_expression(expression, data):
    data = data.assign(y=0.0)
    model = bmb.Model(
        bmb.Formula(f"y ~ 0 + nl({expression} + a)", nlpars=("a",)),
        data,
        priors={"a": normal_prior()},
    )
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[0.0]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        return pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]


def test_supported_expression_operations():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0]})
    formula = bmb.Formula(
        "y ~ sqrt(a ** 2) + log(b) / x",
        nlpars=("a", "b"),
    )
    model = bmb.Model(
        formula,
        data,
        priors={
            "a": normal_prior(),
            "b": bmb.Prior("LogNormal", mu=0, sigma=1),
        },
    )
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[-3.0]]),
            "b": (("chain", "draw"), [[np.exp(2.0)]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        result = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]

    np.testing.assert_allclose(result, [[[5.0, 4.0]]])


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
    np.testing.assert_allclose(result, expected[None, None, :])


@pytest.mark.parametrize("function", ["atan2", "arctan2"])
def test_supported_two_argument_functions(function):
    x = np.array([0.2, 0.5])
    z = np.array([1.2, 1.5])

    result = evaluate_nonlinear_expression(f"{function}(x, z)", pd.DataFrame({"x": x, "z": z}))

    np.testing.assert_allclose(result, np.arctan2(x, z)[None, None, :])


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

    np.testing.assert_allclose(result, references[function](x)[None, None, :])


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
    formula = bmb.Formula(f"y ~ 0 + nl(a + {expression})", nlpars=("a",))

    with pytest.raises(ValueError, match=message):
        bmb.Model(formula, linear_data())


def test_dependency_check_ignores_string_literals():
    data = linear_data(4)
    data["category"] = ["a", "b", "a", "b"]
    formula = bmb.Formula("y ~ a * x", "a ~ C(category, Treatment(reference='a'))", nlpars=("a",))
    model = bmb.Model(formula, data)
    model.build()
    assert_ip_dlogp(model)


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
    formula = bmb.Formula(f"y ~ 0 + nl({expression})", nlpars=("a",))
    with pytest.raises(ValueError, match="Nonlinear|nonlinear|Unsupported"):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize(
    "source, additive, expression",
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
def test_split_additive_and_nonlinear_predictor(source, additive, expression):
    actual_additive, actual_expression = split_predictor(source, ("a", "b"))
    if additive is None:
        assert actual_additive is None
    else:
        assert actual_additive.replace(" ", "") == additive.replace(" ", "")
    values = {"a": 0.5, "b": 0.3, "x": pt.as_tensor_variable([0.2, 0.8])}
    actual = evaluate_expression(actual_expression.root, values).eval()
    expected = evaluate_expression(NonlinearExpression.parse(expression).root, values).eval()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "source", ["nl()", "nl(a, x)", "nl(a=x)", "2 * nl(a*x)", "nl(a*x):z", "nl(nl(a*x))"]
)
def test_invalid_nl_wrapper(source):
    with pytest.raises(ValueError):
        split_predictor(source, ("a",))
