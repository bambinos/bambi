import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy.special import erf, erfc, expit, logit, ndtr, ndtri  # pylint: disable=no-name-in-module

import bambi as bmb
from bambi.nonlinear import (
    NonlinearExpression,
    parameter_dependency_order,
    resolve_nonlinear_symbols,
)

from helpers import assert_ip_dlogp


def normal_prior(sigma=2):
    return bmb.Prior("Normal", mu=0, sigma=sigma)


def linear_data(size=30):
    x = np.linspace(-1, 1, size)
    return pd.DataFrame({"x": x, "z": x**2, "y": 1 + 2 * x})


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


@pytest.mark.parametrize(
    ("main", "additionals", "message"),
    [
        ("y ~ mu + a * x", (), "cannot depend on themselves.*mu"),
        ("y ~ sigma * a", ("sigma ~ mu + 1",), "mu -> sigma -> mu"),
        (
            "y ~ a * x",
            ("a ~ sigma", "sigma ~ mu + 1"),
            "a -> sigma -> mu -> a",
        ),
        (
            "y ~ a * x",
            ("sigma ~ mu + unknown",),
            "No nonlinear parameter formula or data column.*unknown",
        ),
    ],
)
def test_parameter_dependency_validation(main, additionals, message):
    formula = bmb.Formula(main, *additionals, nlpars=("a",))
    with pytest.raises(ValueError, match=message):
        bmb.Model(formula, linear_data())


def test_likelihood_parameter_and_expression_data_collision_is_rejected():
    data = linear_data().assign(mu=0.5)
    formula = bmb.Formula("y ~ a * x", "sigma ~ mu + z", nlpars=("a",))

    with pytest.raises(ValueError, match="both modeled parameters and data columns.*mu"):
        bmb.Model(formula, data)


@pytest.mark.parametrize("response", ["mu", "sigma"])
def test_likelihood_parameter_and_response_data_collision_is_rejected(response):
    data = linear_data().rename(columns={"y": response})
    formula = bmb.Formula(f"{response} ~ a * x", nlpars=("a",))

    with pytest.raises(ValueError, match=f"modeled likelihood parameters.*{response}"):
        bmb.Model(formula, data)


def exponential_formula(group_specific=False):
    a_formula = "a ~ 1 + (1 | group)" if group_specific else "a ~ 1 + z"
    return bmb.Formula(
        "y ~ a + b * exp(-k * x)",
        a_formula,
        nlpars=("a", "b", "k"),
    )


def exponential_priors(group_specific=False):
    a_priors = {"Intercept": normal_prior(), "z": normal_prior()}
    if group_specific:
        a_priors = {
            "Intercept": normal_prior(),
            "1|group": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        }
    return {
        "a": a_priors,
        "b": {"Intercept": normal_prior()},
        "k": {"Intercept": normal_prior()},
    }


def evaluate_nonlinear_expression(expression, data):
    data = data.assign(y=0.0)
    model = bmb.Model(
        bmb.Formula(f"y ~ {expression} + a", nlpars=("a",)),
        data,
        priors={"a": {"Intercept": normal_prior()}},
    )
    model.build()
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.0]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with model.backend.model:
        return pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]


def test_constant_parameters_match_linear_regression():
    data = linear_data()
    nonlinear = bmb.Model(
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        data,
        priors={
            "a": {"Intercept": normal_prior()},
            "b": {"Intercept": normal_prior()},
        },
        center_predictors=False,
    )
    linear = bmb.Model(
        "y ~ x",
        data,
        priors={"Intercept": normal_prior(), "x": normal_prior()},
        center_predictors=False,
    )
    nonlinear.build()
    linear.build()

    nonlinear_draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[1.25]]),
            "b_Intercept": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    linear_draws = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), [[1.25]]),
            "x": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with nonlinear.backend.model:
        nonlinear_mu = pm.compute_deterministics(
            nonlinear_draws, var_names=["mu"], progressbar=False
        )["mu"]
    with linear.backend.model:
        linear_mu = pm.compute_deterministics(linear_draws, var_names=["mu"], progressbar=False)[
            "mu"
        ]

    xr.testing.assert_allclose(nonlinear_mu, linear_mu)


def test_omitted_constant_formulas_match_explicit_intercepts():
    data = linear_data()
    formulas = [
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        bmb.Formula("y ~ a + b * x", "a ~ 1", "b ~ 1", nlpars=("a", "b")),
    ]
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[1.25]]),
            "b_Intercept": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    predictions = []

    for formula in formulas:
        model = bmb.Model(formula, data, center_predictors=False)
        model.build()
        with model.backend.model:
            predictions.append(
                pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)["mu"]
            )

    xr.testing.assert_allclose(predictions[0], predictions[1])


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
            "a": {"Intercept": normal_prior()},
            "b": {"Intercept": bmb.Prior("LogNormal", mu=0, sigma=1)},
        },
    )
    model.build()
    draws = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[-3.0]]),
            "b_Intercept": (("chain", "draw"), [[np.exp(2.0)]]),
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
    formula = bmb.Formula(f"y ~ a + {expression}", nlpars=("a",))

    with pytest.raises(ValueError, match=message):
        bmb.Model(formula, linear_data())


def test_predictor_dependent_parameter_builds_expected_graph():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    model.build()

    assert_ip_dlogp(model)
    assert set(model.nonlinear_predictors) == {"a", "b", "k"}
    assert set(model.nonlinear_predictors["a"].terms) == {"Intercept", "z"}
    assert set(model.nonlinear_predictors["b"].terms) == {"Intercept"}
    assert set(model.nonlinear_predictors["k"].terms) == {"Intercept"}
    assert model.backend.model.named_vars_to_dims["mu"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["a"] == ("__obs__",)
    assert model.backend.model.named_vars_to_dims["mu__x_data"] == ("__obs__",)
    assert {"a_Intercept", "a_z", "b_Intercept", "k_Intercept"} <= set(
        model.backend.model.named_vars
    )


def test_predictor_dependent_parameter_recovers_simulated_values():
    rng = np.random.default_rng(8)
    size = 120
    x = rng.uniform(0, 3, size)
    z = rng.normal(size=size)
    a = 0.7 + 0.4 * z
    y = a + 1.5 * np.exp(-0.8 * x) + rng.normal(0, 0.15, size)
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    priors = exponential_priors()
    priors["b"]["Intercept"] = bmb.Prior("Normal", mu=1.5, sigma=0.5)
    priors["k"]["Intercept"] = bmb.Prior("LogNormal", mu=np.log(0.8), sigma=0.35)
    priors["sigma"] = bmb.Prior("HalfNormal", sigma=0.5)
    model = bmb.Model(exponential_formula(), data, priors=priors)

    idata = model.fit(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        random_seed=11,
        inference_method="pymc",
        target_accept=0.9,
    )

    expected = {
        "a_Intercept": 0.7,
        "a_z": 0.4,
        "b_Intercept": 1.5,
        "k_Intercept": 0.8,
    }
    for name, value in expected.items():
        assert float(idata.posterior[name].mean()) == pytest.approx(value, abs=0.15)

    predicted = model.predict(idata, kind="response", random_seed=4, inplace=False)
    assert predicted.posterior_predictive["y"].shape == (2, 200, size)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_posterior_predictive_and_new_data():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    idata = model.fit(draws=5, chains=2)
    assert {"a", "b", "k", "mu"}.isdisjoint(idata.posterior.data_vars)

    predicted = model.predict(idata, kind="response", inplace=False)
    assert predicted.posterior["mu"].shape == (2, 5, len(data))
    assert predicted.posterior_predictive["y"].shape == (2, 5, len(data))

    new_data = pd.DataFrame({"x": np.linspace(1, 2, 7), "z": np.linspace(-0.5, 0.5, 7)})
    predicted = model.predict(idata, kind="response", data=new_data, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, len(new_data))
    assert predicted.predictions["y"].shape == (2, 5, len(new_data))

    with pytest.raises(ValueError, match="missing nonlinear expression column"):
        model.predict(idata, data=new_data.drop(columns="x"), inplace=False)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_include_response_params_only_keeps_likelihood_parameter():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    idata = model.fit(draws=5, chains=2, include_response_params=True)

    assert "mu" in idata.posterior
    assert {"a", "b", "k"}.isdisjoint(idata.posterior.data_vars)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_group_specific_parameter_predicts_new_data(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = linear_data(24)
    data["group"] = np.repeat(["a", "b", "c"], 8)
    formula = exponential_formula(group_specific=True)
    model = bmb.Model(formula, data, priors=exponential_priors(group_specific=True))
    idata = model.fit(draws=5, chains=2)

    assert model.backend.model.named_vars_to_dims["a_1|group"] == ("group_dim",)
    assert idata.posterior["a_1|group"].shape == (2, 5, 3)
    new_data = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "group": ["a", "new", "new"],
        }
    )
    predicted = model.predict(idata, data=new_data, random_seed=123, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, 3)


@pytest.mark.parametrize(
    "formula, error",
    [
        (
            bmb.Formula("y ~ a + b * x", nlpars=("b",)),
            "No nonlinear parameter formula or data column",
        ),
        (
            bmb.Formula("y ~ a + x", nlpars=("a", "b")),
            "not used by the expression",
        ),
        (
            bmb.Formula("y ~ a + unknown", nlpars=("a",)),
            "No nonlinear parameter formula or data column",
        ),
    ],
)
def test_validation_errors(formula, error):
    with pytest.raises(ValueError, match=error):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize(
    "additionals",
    [("a ~ 1 + b",), ("a ~ 1", "sigma ~ 1 + b")],
)
def test_data_name_collision_is_reported_before_parameter_dependency(additionals):
    data = linear_data()
    data["b"] = 1.0
    formula = bmb.Formula("y ~ a + b * x", *additionals, nlpars=("a", "b"))

    with pytest.raises(ValueError, match=r"must not also be data columns: \['b'\]"):
        bmb.Model(formula, data)


def test_malformed_expression():
    formula = bmb.Formula("y ~ a +", nlpars=("a",))
    with pytest.raises(ValueError, match="Malformed nonlinear expression"):
        bmb.Model(formula, linear_data())


def test_nonnumeric_expression_data():
    data = linear_data()
    data["label"] = "a"
    formula = bmb.Formula("y ~ a + label", nlpars=("a",))

    with pytest.raises(ValueError, match="Nonlinear expression data must be numeric"):
        bmb.Model(formula, data)


def test_nested_priors_are_assigned():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    prior = model.nonlinear_predictors["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 0, "sigma": 2}


def test_set_priors_uses_nested_parameter_names():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())
    updated = bmb.Prior("Normal", mu=1, sigma=0.25)

    model.set_priors({"a": {"z": updated}})

    prior = model.nonlinear_predictors["a"].terms["z"].prior
    assert prior.name == "Normal"
    assert prior.args == {"mu": 1, "sigma": 0.25}


def test_nonlinear_coefficient_alias():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    model.set_alias({"a": {"Intercept": "a0"}})
    model.build()
    assert "a0" in model.backend.model.named_vars
    assert "a_Intercept" not in model.backend.model.named_vars


def test_vector_parent_is_rejected():
    formula = bmb.Formula("y ~ rate * x", nlpars=("rate",))

    with pytest.raises(ValueError, match="scalar parent parameter"):
        bmb.Model(formula, linear_data(), family="categorical")


def test_non_proportion_multicolumn_response_is_rejected():
    data = linear_data()
    data["status"] = "none"
    formula = bmb.Formula("censored(y, status) ~ rate * x", nlpars=("rate",))

    with pytest.raises(ValueError, match="one observed response or a proportion response"):
        bmb.Model(formula, data)


def test_dropna_aligns_all_model_inputs():
    data = linear_data(8)
    data.index = [4, 4, 2, 2, 9, 9, 1, 1]
    data["group"] = ["g", "h"] * 4
    data["unused"] = np.nan
    for row, column in enumerate(["y", "x", "z", "group"]):
        data.iloc[row, data.columns.get_loc(column)] = np.nan
    original = data.copy(deep=True)
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + z", "b ~ 1 + (1 | group)", nlpars=("a", "b"))
    model = bmb.Model(formula, data, dropna=True)
    model.build()

    expected = data.iloc[4:].copy()
    expected["group"] = expected["group"].astype("category")
    pd.testing.assert_frame_equal(model.data, expected)
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[4:])
    np.testing.assert_array_equal(model.backend.model["mu__x_data"].get_value(), data.x.iloc[4:])
    np.testing.assert_array_equal(model.nonlinear_predictors["a"].terms["z"].data, data.z.iloc[4:])
    assert_ip_dlogp(model)


@pytest.mark.parametrize("column", ["successes", "attempts", "x"])
def test_dropna_aligns_nonlinear_proportion_inputs(column):
    data = pd.DataFrame(
        {
            "successes": [6.0, 13.0, 18.0],
            "attempts": [59.0, 60.0, 62.0],
            "x": [0.1, 0.2, 0.3],
        }
    )
    data.loc[0, column] = np.nan
    formula = bmb.Formula("prop(successes, attempts) ~ normal_cdf(a + b * x)", nlpars=("a", "b"))

    model = bmb.Model(formula, data, family="binomial", link="identity", dropna=True)
    model.build()

    pd.testing.assert_frame_equal(model.data, data.iloc[1:])
    np.testing.assert_array_equal(model.response_term.data, data.loc[1:, ["successes", "attempts"]])
    np.testing.assert_array_equal(model.backend.model["p__x_data"].get_value(), data.x.iloc[1:])


@pytest.mark.parametrize("column", ["y", "x", "z"])
def test_missing_model_inputs_raise_without_dropna(column):
    data = linear_data()
    data.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        bmb.Model(exponential_formula(), data)


def test_dropna_rejects_no_complete_observations():
    data = linear_data()
    data["x"] = np.nan
    with pytest.raises(ValueError, match="complete observation"):
        bmb.Model(exponential_formula(), data, dropna=True)


def test_dependency_check_ignores_string_literals():
    data = linear_data(4)
    data["category"] = ["a", "b", "a", "b"]
    formula = bmb.Formula("y ~ a * x", "a ~ C(category, Treatment(reference='a'))", nlpars=("a",))
    model = bmb.Model(formula, data)
    model.build()
    assert_ip_dlogp(model)


def test_nonlinear_predictor_can_depend_on_another_predictor():
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + b", nlpars=("a", "b"))
    model = bmb.Model(formula, linear_data())
    model.build()
    draws = xr.Dataset(
        {
            "b_Intercept": (("chain", "draw"), [[0.25, -0.5]]),
            "sigma": (("chain", "draw"), [[1.0, 1.0]]),
        }
    )

    with model.backend.model:
        actual = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)

    x = xr.DataArray(linear_data().x.to_numpy(), dims="__obs__")
    expected = 1 + draws.b_Intercept + draws.b_Intercept * x
    np.testing.assert_allclose(actual.mu, expected)


@pytest.mark.parametrize(("rhs", "error"), [("I(b ** 2)", "function 'I'"), ("(1 | b)", "BitOr")])
def test_parameter_dependent_formula_requires_nonlinear_expression_syntax(rhs, error):
    formula = bmb.Formula("y ~ a + b * x", f"a ~ {rhs}", nlpars=("a", "b"))
    with pytest.raises(ValueError, match=error):
        bmb.Model(formula, linear_data())


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("column", ["x", "z"])
def test_prediction_rejects_incomplete_inputs(column):
    model = bmb.Model(
        exponential_formula(), linear_data(), priors=exponential_priors(), dropna=True
    )
    idata = model.fit(draws=2, chains=1)
    data = linear_data(3).drop(columns="y")
    data.loc[1, column] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=data, inplace=False)


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
    formula = bmb.Formula(f"y ~ {expression}", nlpars=("a",))
    with pytest.raises(ValueError, match="Nonlinear|nonlinear|Unsupported"):
        bmb.Model(formula, linear_data())


@pytest.mark.parametrize("name", ["mu", "sigma", "exp", "sin", "normal_cdf", "x"])
def test_reserved_parameter_names_are_rejected(name):
    formula = bmb.Formula(f"y ~ {name}", nlpars=(name,))
    with pytest.raises(ValueError, match="names must not"):
        bmb.Model(formula, linear_data())
