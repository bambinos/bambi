import logging
import pathlib

import pytest
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from scipy import stats
from scipy.special import expit, ndtr  # pylint: disable=no-name-in-module
from scipy.stats import norm

from bambi.backend.pymc.transform import transforms_registry
from bambi.parameters import (
    Conditional,
    ConditionalCoefficient,
    ConditionalParameter,
    Marginal,
    MarginalCoefficient,
    MarginalParameter,
)
from bambi.terms import CommonTerm, GroupSpecificTerm
from bambi.backend.pymc.transform import transforms_registry
from bambi.defaults import get_builtin_family
from bambi.backend.pymc.parameters import remove_group_specific_contributions
from bambi.backend.pymc.terms.response import _untruncate_response
from bambi.backend.pymc.utils import _compute_logccdf, make_competing_risks_logp
from formulae import design_matrices
from pytensor.sparse import StructuredDot

from helpers import assert_ip_dlogp, graph_contains_op


@pytest.fixture(scope="module")
def init_data():
    """Data used to test initialization method"""
    return pd.read_csv(pathlib.Path(__file__).parent / "data" / "obs.csv")


def test_term_init(data_diabetes):
    design = design_matrices("BMI", data_diabetes)
    term = design.common.terms["BMI"]
    term = CommonTerm(term, prior=None)
    assert term.name == "BMI"
    assert not term.categorical
    assert term.levels is None
    assert term.data.shape == (442,)


def test_distribute_group_specific_effect_over(data_diabetes):
    # 163 unique levels of BMI in data_diabetes
    # With intercept
    model = bmb.Model("BP ~ (C(age_grp)|BMI)", data_diabetes)

    # Treatment encoding because of the intercept
    levels = sorted(list(data_diabetes["age_grp"].unique()))[1:]
    levels = [str(level) for level in levels]
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert "C(age_grp)|BMI" in parent_parameter.terms
    assert "1|BMI" in parent_parameter.terms
    assert parent_parameter.terms["C(age_grp)|BMI"].expr.levels == levels

    # This is equal to the sub-matrix of Z that corresponds to this term.
    # 442 is the number of observations. 163 the number of groups.
    # 2 is the number of levels of the categorical variable 'C(age_grp)' after removing
    # the reference level. Then the number of columns is 326 = 163 * 2.
    assert parent_parameter.terms["C(age_grp)|BMI"].data.shape == (442, 326)

    # Without intercept. Reference level is not removed.
    model = bmb.Model("BP ~ (0 + C(age_grp)|BMI)", data_diabetes)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert "C(age_grp)|BMI" in parent_parameter.terms
    assert not "1|BMI" in parent_parameter.terms
    assert parent_parameter.terms["C(age_grp)|BMI"].data.shape == (442, 489)


def test_model_init_bad_data():
    with pytest.raises(ValueError):
        bmb.Model("y ~ x", {"x": 1})


def test_unbuilt_model(data_diabetes):
    model = bmb.Model("Y ~ AGE", data=data_diabetes)
    with pytest.raises(ValueError):
        model._check_built()


def test_model_categorical_argument():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.integers(2, size=100),
            "z": rng.integers(2, size=100),
        }
    )
    model = bmb.Model("y ~ 0 + x", data, categorical="x")
    assert model.parameters[model.family.likelihood.parent].terms["x"].categorical

    model = bmb.Model("y ~ 0 + x*z", data, categorical=["x", "z"])
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["x"].categorical
    assert parent_parameter.terms["z"].categorical
    assert parent_parameter.terms["x:z"].categorical


def test_additive_and_group_specific_terms_share_predictor_data():
    data = pd.DataFrame(
        {
            "y": np.linspace(0, 1, 24),
            "g": np.tile(["u", "v"], 12),
            "h": np.tile(["a", "b", "c"], 8),
        }
    )

    model = bmb.Model("y ~ g + (g|h)", data)
    model.build()

    pymc_model = model.backend.model
    assert pymc_model.named_vars_to_dims["g_data"] == ("__obs__", "g_dim_reduced")
    assert "g_2_data" not in pymc_model.named_vars
    assert_ip_dlogp(model)


def test_model_no_response():
    with pytest.raises(ValueError):
        bmb.Model("x", pd.DataFrame({"x": [1]}))


def test_model_term_names_property(data_diabetes):
    model = bmb.Model("BMI ~ age_grp + BP + S1", data_diabetes)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.intercept_term.name == "Intercept"
    assert set(parent_parameter.common_terms) == {"age_grp", "BP", "S1"}


def test_model_term_names_property_interaction(data_crossed):
    data_crossed["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = bmb.Model("Y ~ threecats*fourcats", data_crossed)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.intercept_term.name == "Intercept"
    assert set(parent_parameter.common_terms) == {
        "threecats",
        "fourcats",
        "threecats:fourcats",
    }


def test_model_terms_levels_interaction(data_crossed):
    data_crossed["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = bmb.Model("Y ~ threecats*fourcats", data_crossed)

    assert model.parameters[model.family.likelihood.parent].terms["threecats:fourcats"].levels == [
        "b, b",
        "b, c",
        "b, d",
        "c, b",
        "c, c",
        "c, d",
    ]


def test_model_terms_levels():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=50),
            "x": rng.normal(size=50),
            "z": np.repeat([f"Group {x}" for x in ["1", "2", "3", "1", "2"]], 10),
            "time": list(range(1, 11)) * 5,
            "subject": np.repeat([f"Subject {x}" for x in range(1, 6)], 10),
        }
    )
    model = bmb.Model("y ~ x + z + time + (time|subject)", data)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["z"].levels == ["Group 2", "Group 3"]
    assert parent_parameter.terms["1|subject"].groups == [f"Subject {x}" for x in range(1, 6)]
    assert parent_parameter.terms["time|subject"].groups == [f"Subject {x}" for x in range(1, 6)]


def test_model_term_classes():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=50),
            "x": rng.normal(size=50),
            "s": ["s1"] * 25 + ["s2"] * 25,
            "g": rng.choice(["a", "b", "c"], size=50),
        }
    )

    model = bmb.Model("y ~ x*g + (x|s)", data)

    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert isinstance(parent_parameter.terms["x"], CommonTerm)
    assert isinstance(parent_parameter.terms["g"], CommonTerm)
    assert isinstance(parent_parameter.terms["x:g"], CommonTerm)
    assert isinstance(parent_parameter.terms["1|s"], GroupSpecificTerm)
    assert isinstance(parent_parameter.terms["x|s"], GroupSpecificTerm)

    # Also check 'categorical' attribute is right
    assert parent_parameter.terms["g"].categorical


def test_one_shot_formula_fit(data_diabetes, mock_pymc_sample):
    model = bmb.Model("S3 ~ S1 + S2", data_diabetes)
    model.build()
    assert_ip_dlogp(model)
    model.fit(chains=2)
    named_vars = set(model.backend.model.named_vars)
    targets = {"S3", "S1", "Intercept"}
    assert len(named_vars & targets) == 3


def test_categorical_term(mock_pymc_sample):
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=6),
            "x1": rng.normal(size=6),
            "x2": [1, 1, 0, 0, 1, 1],
            "g1": ["a"] * 3 + ["b"] * 3,
            "g2": ["x", "x", "z", "z", "y", "y"],
        }
    )
    model = bmb.Model("y ~ x1 + x2 + g1 + (g1|g2) + (x2|g2)", data)
    model.build()
    assert_ip_dlogp(model)
    fitted = model.fit(chains=2)
    df = az.summary(fitted)
    names = {
        "Intercept",
        "Intercept_centered",
        "x1",
        "x2",
        "g1[b]",
        "1|g2_sigma",
        "g1|g2_sigma[b]",
        "x2|g2_sigma",
        "sigma",
        "1|g2[x]",
        "1|g2[y]",
        "1|g2[z]",
        "g1|g2[x, b]",
        "g1|g2[y, b]",
        "g1|g2[z, b]",
        "x2|g2[x]",
        "x2|g2[y]",
        "x2|g2[z]",
    }
    assert set(df.index) == names


def test_omit_offsets_false(data_random_n100, mock_pymc_sample):
    model = bmb.Model("continuous1 ~ continuous2 + (continuous2|binary_cat)", data_random_n100)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2, omit_offsets=False)
    offsets = set(var for var in idata.posterior.data_vars if var.endswith("_offset"))
    assert offsets == {"1|binary_cat_offset", "continuous2|binary_cat_offset"}


def test_omit_offsets_true(data_random_n100, mock_pymc_sample):
    model = bmb.Model("continuous1 ~ continuous2 + (continuous2|binary_cat)", data_random_n100)
    idata = model.fit(chains=2, omit_offsets=True)
    model.predict(idata)
    model.predict(idata, kind="response")
    assert_ip_dlogp(model)
    offsets = [var for var in idata.posterior.var() if var.endswith("_offset")]
    assert not offsets


def test_hyperprior_on_common_effect(data_random_n100):
    slope = bmb.Prior("Normal", mu=0, sd=bmb.Prior("HalfCauchy", beta=2))

    priors = {"continuous2": slope}
    with pytest.raises(ValueError):
        bmb.Model(
            "continuous1 ~ continuous2 + (continuous2|binary_cat)", data_random_n100, priors=priors
        )

    priors = {"common": slope}
    with pytest.raises(ValueError):
        bmb.Model(
            "continuous1 ~ continuous2 + (continuous2|binary_cat)", data_random_n100, priors=priors
        )


@pytest.mark.parametrize(
    "family",
    [
        "asymmetriclaplace",
        "exgaussian",
        "gaussian",
        "loglogistic",
        "lognormal",
        "negativebinomial",
        "bernoulli",
        "poisson",
        "gamma",
        "vonmises",
        "wald",
    ],
)
def test_automatic_priors(family):
    """Test that automatic priors work correctly"""
    obs = pd.DataFrame([0], columns=["x"])
    bmb.Model("x ~ 0", obs, family=family)


def test_links(data_random_n100):
    FAMILIES = {
        "asymmetriclaplace": ["identity", "log", "inverse"],
        "bernoulli": ["identity", "logit", "probit", "cloglog"],
        "beta": ["logit", "probit", "cloglog"],
        "exgaussian": ["identity", "log", "inverse"],
        "gamma": ["identity", "inverse", "log"],
        "gaussian": ["identity", "log", "inverse"],
        "loglogistic": ["identity", "log", "inverse"],
        "lognormal": ["identity", "log", "inverse"],
        "negativebinomial": ["identity", "log", "cloglog"],
        "poisson": ["identity", "log"],
        "vonmises": ["identity"],
        "wald": ["inverse", "inverse_squared", "identity", "log"],
    }
    for family, links in FAMILIES.items():
        for link in links:
            if family == "bernoulli":
                formula = "binary_num ~ continuous2"
            else:
                formula = "count2 ~ continuous2"
            bmb.Model(formula, data_random_n100, family=family, link=link)


def test_bad_links(data_random_n100):
    """Passes names of links that are not suitable for the family."""
    FAMILIES = {
        "bernoulli": ["inverse", "inverse_squared", "log"],
        "beta": ["inverse", "inverse_squared", "log"],
        "exgaussian": ["logit", "probit", "cloglog"],
        "gamma": ["logit", "probit", "cloglog"],
        "gaussian": ["logit", "probit", "cloglog"],
        "loglogistic": ["logit", "probit", "cloglog"],
        "lognormal": ["logit", "probit", "cloglog"],
        "negativebinomial": ["logit", "probit", "inverse", "inverse_squared"],
        "poisson": ["logit", "probit", "cloglog", "inverse", "inverse_squared"],
        "vonmises": ["logit", "probit", "cloglog"],
        "wald": ["logit", "probit", "cloglog"],
    }

    for family, links in FAMILIES.items():
        for link in links:
            with pytest.raises(ValueError):
                if family == "bernoulli":
                    formula = "binary_num ~ continuous2"
                else:
                    formula = "count2 ~ continuous2"
                bmb.Model(formula, data_random_n100, family=family, link=link)


def test_constant_terms():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=10),
            "x": rng.choice([1], size=10),
            "z": rng.choice(["A"], size=10),
        }
    )

    with pytest.raises(ValueError):
        bmb.Model("y ~ 0 + x", data)

    with pytest.raises(ValueError):
        bmb.Model("y ~ 0 + z", data)


def test_1d_group_specific(data_random_n100):
    # Since there's 1|g, there's only one column for x|g
    # We need to ensure x|g is of shape (100,) and not of shape (100, 1)
    # We do so by checking the mean is (100, ) because shape of x|g still returns (100, 1)
    # The difference is that we do .squeeze() on it after creation.
    model = bmb.Model("continuous1 ~ (binary_cat|categorical1)", data_random_n100)
    model.build()
    assert model.backend.model["mu"].shape.eval() == (100,)


def test_data_is_copied():
    adults = bmb.load_data("adults")

    model_1 = bmb.Model("age ~ sex * race", adults)
    model_2 = bmb.Model("age ~ sex * race", adults, categorical=["age", "sex"])

    for model in [model_1, model_2]:
        assert id(adults) != id(model.data)
        assert all(model.data.dtypes[:3] == "category")

    # NOTE: https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#dedicated-string-data-type-by-default
    assert all(dtype in ("object", "str") for dtype in adults.dtypes[:3])


def test_response_is_censored():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "status": ["none", "right", "none", "left", "none"],
        }
    )
    dm = bmb.Model("censored(x, status) ~ 1", df)
    assert dm.response_term.is_censored is True


@pytest.mark.parametrize(
    ("formula", "family", "response_name"),
    [
        ("censored(y, censoring) ~ 1", None, "y"),
        ("truncated(y, lower, upper) ~ 1", None, "y"),
        ("constrained(y, -1, 5) ~ 1", None, "y"),
        ("weighted(y, weights) ~ 1", None, "y"),
        ("counts(y1, y2, n=n) ~ 1", "multinomial", "y1_y2"),
        ("prop(y, n) ~ 1", "binomial", "y"),
        ("risks(time, event_status, cause) ~ 1", "weibull", "time"),
    ],
)
def test_transformed_response_uses_observed_variable_name(formula, family, response_name):
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "censoring": ["none", "right", "left"],
            "lower": [0.0, 0.0, 0.0],
            "upper": [4.0, 4.0, 4.0],
            "weights": [1.0, 1.0, 1.0],
            "y1": [1, 2, 1],
            "y2": [3, 2, 3],
            "n": [4, 4, 4],
            "time": [1.0, 2.0, 3.0],
            "event_status": ["event", "right", "event"],
            "cause": ["cause_a", "none", "cause_b"],
        }
    )
    kwargs = {} if family is None else {"family": family}
    model = bmb.Model(formula, data, **kwargs)

    assert model.response_term.name == response_name
    model.build()
    assert response_name in model.backend.model.named_vars


def test_transformed_response_accepts_its_full_name_as_an_alias():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "status": ["none", "right", "none"]})
    model = bmb.Model("censored(y, status) ~ 1", data)
    model.set_alias({"censored(y, status)": "outcome"})

    assert model.response_term.label == "outcome"


@pytest.mark.parametrize("model_cure", [False, True])
@pytest.mark.parametrize("model_shape", [False, True])
@pytest.mark.parametrize(
    "family, parent, shape",
    [
        ("cure_weibull", "mu", "alpha"),
        ("cure_exponential", "mu", None),
        ("cure_gamma", "mu", "alpha"),
        ("cure_lognormal", "mu", "sigma"),
        ("cure_loglogistic", "mu", "alpha"),
        ("cure_weibull_ph", "lam", "alpha"),
    ],
)
def test_cure_model_construction(model_cure, model_shape, family, parent, shape):
    rng = np.random.default_rng(1234)
    x = rng.normal(size=100)
    cure = 1 / (1 + np.exp(-(-0.5 + 0.3 * x)))
    event_time = np.exp(0.3 * x) * rng.weibull(1.5, size=100)
    event_time[rng.uniform(size=100) < cure] = np.inf
    follow_up = rng.uniform(2, 8, size=100)
    data = pd.DataFrame(
        {
            "time": np.minimum(event_time, follow_up),
            "status": np.where(event_time > follow_up, "right", "none"),
            "x": x,
        }
    )
    additionals = []
    if model_cure:
        additionals.append("cure ~ x")
    if model_shape and shape:
        additionals.append(f"{shape} ~ x")

    model = bmb.Model(bmb.Formula("censored(time, status) ~ x", *additionals), data, family=family)
    model.build()
    assert model.family.link["cure"].name == "logit"
    assert set(model.conditional_parameters) == {parent} | ({"cure"} if model_cure else set()) | (
        {shape} if model_shape and shape else set()
    )
    point = model.backend.model.initial_point()
    assert np.isfinite(model.backend.model.compile_logp()(point))
    assert np.isfinite(model.backend.model.compile_dlogp()(point)).all()


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.5])
def test_weibull_ph_density_and_hazard_ratio(alpha):
    family = get_builtin_family("weibull_ph")
    transform = transforms_registry.get_parameter_transform(family)
    lam = np.exp(np.array([-0.7, 0.0, 0.7]))
    parameters = transform(
        {"lam": pt.as_tensor_variable(lam), "alpha": pt.as_tensor_variable(np.float64(alpha))}
    )
    time = np.array([0.5, 1.0, 3.0])
    distribution = pm.Weibull.dist(**parameters)
    log_survival = -lam * time**alpha
    log_hazard = np.log(alpha) + np.log(lam) + (alpha - 1) * np.log(time)
    np.testing.assert_allclose(pm.logp(distribution, time).eval(), log_hazard + log_survival)
    np.testing.assert_allclose(
        pm.logcdf(distribution, time).eval(), np.log(-np.expm1(log_survival)), atol=1e-12
    )
    # The hazard ratio is exp(0.7) at every time, regardless of the shared shape.
    hazards = stats.weibull_min.pdf(time[:, None], c=alpha, scale=parameters["beta"].eval())
    hazards /= stats.weibull_min.sf(time[:, None], c=alpha, scale=parameters["beta"].eval())
    np.testing.assert_allclose(hazards[:, 2] / hazards[:, 1], np.exp(0.7))
    if alpha == 1:
        np.testing.assert_allclose(
            pm.logp(distribution, time).eval(), pm.logp(pm.Exponential.dist(lam=lam), time).eval()
        )


@pytest.mark.parametrize(
    "family", ["exponential", "weibull", "weibull_ph", "lognormal", "loglogistic", "gamma", "wald"]
)
def test_competing_risks_response_data(family):
    data = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": ["right", "event", "event", "right"],
            "cause": ["none", "cause_b", "cause_a", "none"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    kwargs = {"link": "log"} if family == "gamma" else {}
    model = bmb.Model("risks(time, status, cause) ~ x", data, family=family, **kwargs)

    assert model.response_term.is_competing_risks is True
    assert model.response_term.levels == ["cause_a", "cause_b"]
    model.build()
    assert "time_data" in model.backend.model.named_vars
    assert "status_data" in model.backend.model.named_vars
    assert "cause_data" in model.backend.model.named_vars
    np.testing.assert_array_equal(model.backend.model["status_data"].get_value(), [1, 0, 0, 1])
    np.testing.assert_array_equal(model.backend.model["cause_data"].get_value(), [0, 2, 1, 0])
    parent = model.family.likelihood.parent
    assert tuple(model.backend.model[parent].shape.eval()) == (len(data), 2)
    assert model.backend.model.named_vars_to_dims[parent] == ("__obs__", "cause_dim")
    assert list(model.backend.model.coords["cause_dim"]) == ["cause_a", "cause_b"]
    prediction_data, _, _ = model.backend._build_new_data(
        pd.DataFrame({"x": [4.0, 5.0]}), "prediction", "response_params"
    )
    np.testing.assert_array_equal(prediction_data["status_data"], [1, 1])
    np.testing.assert_array_equal(prediction_data["cause_data"], [0, 0])

    conditional_data, _, _ = model.backend._build_new_data(
        data.iloc[[1, 2]], "prediction", "response_conditional"
    )
    np.testing.assert_array_equal(conditional_data["time_data"], [2.0, 3.0])
    np.testing.assert_array_equal(conditional_data["status_data"], [0, 0])
    np.testing.assert_array_equal(conditional_data["cause_data"], [2, 1])

    log_likelihood_data, _, _ = model.backend._build_new_data(data.iloc[[1]], "log_likelihood")
    # `cause_b` remains code 2 even though `cause_a` is absent from this data frame.
    np.testing.assert_array_equal(log_likelihood_data["cause_data"], [2])
    assert np.isfinite(model.backend.model.compile_logp()(model.backend.model.initial_point()))


def test_competing_risks_uses_cause_variable_for_coordinate_name():
    data = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "event_status": ["right", "event"],
            "event_type": ["none", "cause_a"],
        }
    )
    model = bmb.Model("risks(time, event_status, event_type) ~ 1", data, family="weibull")
    model.build()

    assert model.backend.model.named_vars_to_dims["mu"] == ("__obs__", "event_type_dim")
    assert list(model.backend.model.coords["event_type_dim"]) == ["cause_a"]


@pytest.mark.parametrize(
    ("dist", "parameter_names", "parameters"),
    [
        (pm.Exponential, ["lam"], (np.array([[1.0, 1.0]]),)),
        (pm.Weibull, ["alpha", "beta"], (np.array([[3.0, 0.7]]), np.array([[20.0, 10.0]]))),
        (pm.LogNormal, ["mu", "sigma"], (np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))),
        (pm.Gamma, ["mu", "sigma"], (np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))),
        (pm.Wald, ["mu", "lam"], (np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]))),
    ],
)
def test_competing_risks_logp_is_stable_in_the_upper_tail(dist, parameter_names, parameters):
    logp = make_competing_risks_logp(dist, parameter_names)(
        np.array([1000.0]),
        *parameters,
        status=np.array([1]),
        cause=np.array([1]),
    )
    assert np.isfinite(pytensor.function([], logp, mode="FAST_COMPILE")()).all()


@pytest.mark.parametrize(
    ("dist", "parameter_names", "parameters", "reference_logsf"),
    [
        (
            pm.Exponential,
            ["lam"],
            (0.7,),
            lambda value: stats.expon.logsf(value, scale=1 / 0.7),
        ),
        (
            pm.Weibull,
            ["alpha", "beta"],
            (0.7, 3.0),
            lambda value: stats.weibull_min.logsf(value, c=0.7, scale=3.0),
        ),
        (
            pm.LogNormal,
            ["mu", "sigma"],
            (0.2, 0.7),
            lambda value: stats.lognorm.logsf(value, s=0.7, scale=np.exp(0.2)),
        ),
        (
            pm.Gamma,
            ["mu", "sigma"],
            (2.0, 1.0),
            lambda value: stats.gamma.logsf(value, a=4.0, scale=0.5),
        ),
        (
            pm.Wald,
            ["mu", "lam"],
            (2.0, 3.0),
            lambda value: stats.invgauss.logsf(value, mu=2 / 3, scale=3.0),
        ),
    ],
)
def test_competing_risks_logccdf_matches_scipy(dist, parameter_names, parameters, reference_logsf):
    value = np.array([0.1, 1.0, 10.0, 100.0])
    parameter_tensors = tuple(np.array([[parameter]]) for parameter in parameters)
    base_dist = dist.dist(**dict(zip(parameter_names, parameter_tensors, strict=True)))
    logccdf = _compute_logccdf(dist, base_dist, pt.shape_padright(value), parameter_tensors)
    actual = pytensor.function([], logccdf, mode="FAST_COMPILE")()[:, 0]

    np.testing.assert_allclose(actual, reference_logsf(value), rtol=1e-6, atol=1e-6)


def test_response_is_truncated():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    dm = bmb.Model("truncated(x, 5.5) ~ 1", df)
    assert dm.response_term.is_truncated is True


def test_counts_response_data():
    data = pd.DataFrame(
        {
            "y1": [1, 2, 3],
            "y2": [3, 4, 3],
            "n": [4, 6, 6],
            "x": [0.0, 1.0, 2.0],
        }
    )

    fixed_model = bmb.Model("counts(y1, y2) ~ x", data, family="multinomial")
    assert fixed_model.response_term.is_counts is True
    fixed_model.build()
    assert "n_data" not in fixed_model.backend.model.named_vars

    with pytest.warns(UserWarning, match="first training total"):
        prediction_data, _, _ = fixed_model.backend._build_new_data(data[["x"]], "prediction")
    assert np.array_equal(prediction_data["y1_y2_data"].sum(axis=1), np.full(len(data), 4))

    log_likelihood_data, _, _ = fixed_model.backend._build_new_data(data, "log_likelihood")
    assert np.array_equal(log_likelihood_data["y1_y2_data"], data[["y1", "y2"]])

    variable_model = bmb.Model("counts(y1, y2, n=n) ~ x", data, family="multinomial")
    variable_model.build()
    assert "n_data" in variable_model.backend.model.named_vars

    prediction_data, _, _ = variable_model.backend._build_new_data(data[["x", "n"]], "prediction")
    assert np.array_equal(prediction_data["n_data"], data["n"])

    log_likelihood_data, _, _ = variable_model.backend._build_new_data(data, "log_likelihood")
    assert np.array_equal(log_likelihood_data["n_data"], data["n"])


def test_custom_likelihood_function(mock_pymc_sample):
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [1, 1, 2, 2, 3]})

    def CustomGaussian(*args, **kwargs):
        return pm.Normal(*args, **kwargs)

    sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    likelihood = bmb.Likelihood(
        "CustomGaussian", params=["mu", "sigma"], parent="mu", dist=CustomGaussian
    )
    family = bmb.Family("custom_gaussian", likelihood, "identity")
    model = bmb.Model("y ~ x", df, family=family, priors={"sigma": sigma_prior})
    model.build()
    assert_ip_dlogp(model)
    model.fit(chains=2)
    assert (
        model.backend.model.observed_RVs[0].str_repr()
        == "y ~ Normal(f(Intercept_centered, x), sigma)"
    )


def test_extra_namespace():
    """Tests the formula can access an additional namespace"""
    data = bmb.load_data("carclaims")
    extra_namespace = {"levels": data["veh_body"].unique()}
    formula = "numclaims ~ 0 + C(veh_body, levels=levels)"
    model = bmb.Model(formula, data, family="poisson", link="log", extra_namespace=extra_namespace)
    term = model.parameters[model.family.likelihood.parent].terms["C(veh_body, levels=levels)"]
    assert set(np.asarray(term.levels)) == set(data["veh_body"].unique())


def test_drop_na(data_crossed, caplog):
    data_crossed_missing = data_crossed.copy()
    data_crossed_missing.loc[0, "Y"] = np.nan
    data_crossed_missing.loc[1, "continuous"] = np.nan
    data_crossed_missing.loc[2, "threecats"] = np.nan

    with caplog.at_level(logging.INFO):
        bmb.Model("Y ~ continuous + threecats", data_crossed_missing, dropna=True)
        assert "Automatically removing 3/120 rows from the dataset." in caplog.text

    with pytest.raises(ValueError, match="'data' contains 3 incomplete rows"):
        bmb.Model("Y ~ continuous + threecats", data_crossed_missing)


def test_plot_priors(data_crossed):
    model = bmb.Model("Y ~ 0 + threecats", data_crossed)
    with pytest.raises(ValueError, match="Model is not built yet"):
        model.plot_priors()
    model.build()
    model.plot_priors()


def test_model_graph(data_crossed):
    model = bmb.Model("Y ~ 0 + threecats", data_crossed)
    with pytest.raises(ValueError, match="Model is not built yet"):
        model.graph()
    model.build()
    model.graph()


def test_potentials():
    data = pd.DataFrame(np.repeat((0, 1), (18, 20)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}

    potentials = [
        (("Intercept", "Intercept"), lambda x, y: bmb.math.switch(x < 0.45, y, -np.inf)),
        ("Intercept", lambda x: bmb.math.switch(x > 0.55, 0, -np.inf)),
    ]

    model = bmb.Model(
        "w ~ 1",
        data,
        family="bernoulli",
        link="identity",
        priors=priors,
        potentials=potentials,
    )
    model.build()
    assert len(model.backend.model.potentials) == 2

    pot0 = model.backend.model.potentials[0].get_parents()[0]
    pot1 = model.backend.model.potentials[1].get_parents()[0]
    assert pot0.__str__() == "Switch(Lt.0, Intercept, -inf)"
    assert pot1.__str__() == "Switch(Gt.0, 0, -inf)"


def test_potentials_resolve_aliases():
    data = pd.DataFrame(np.repeat((0, 1), (18, 20)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    potentials = [("alpha", lambda x: bmb.math.switch(x > 0.55, 0, -np.inf))]

    model = bmb.Model(
        "w ~ 1",
        data,
        family="bernoulli",
        link="identity",
        priors=priors,
        potentials=potentials,
    )
    model.set_alias({"Intercept": "alpha"})
    model.build()

    assert len(model.backend.model.potentials) == 1


def test_potentials_missing_variable():
    data = pd.DataFrame(np.repeat((0, 1), (18, 20)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    potentials = [("not_a_variable", lambda x: x)]

    model = bmb.Model(
        "w ~ 1",
        data,
        family="bernoulli",
        link="identity",
        priors=priors,
        potentials=potentials,
    )

    with pytest.raises(ValueError, match="not_a_variable"):
        model.build()


def test_potentials_non_callable_constraint():
    data = pd.DataFrame(np.repeat((0, 1), (18, 20)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    potentials = [("Intercept", 1)]

    model = bmb.Model(
        "w ~ 1",
        data,
        family="bernoulli",
        link="identity",
        priors=priors,
        potentials=potentials,
    )

    with pytest.raises(TypeError, match="must be callable"):
        model.build()


def test_compute_log_likelihood(data_random_n100, mock_pymc_sample):
    data = data_random_n100.iloc[:10].copy()
    model = bmb.Model("continuous1 ~ continuous2", data)
    idata = model.fit(draws=4, chains=2)
    assert "continuous1_data" in model.backend.model.named_vars

    result = model.compute_log_likelihood(idata, inplace=False)

    assert "log_likelihood" not in idata
    assert result.log_likelihood["continuous1"].shape == (2, 4, 10)
    assert result.log_likelihood.attrs["modeling_interface"] == "bambi"

    same_data_result = model.compute_log_likelihood(idata, data=data, inplace=False)
    assert (
        same_data_result.log_likelihood["continuous1"] == result.log_likelihood["continuous1"]
    ).all()

    new_data = data.iloc[:3].copy()
    new_result = model.compute_log_likelihood(idata, data=new_data, inplace=False)
    assert new_result.log_likelihood["continuous1"].shape == (2, 4, 3)

    model.compute_log_likelihood(idata)
    assert idata.log_likelihood["continuous1"].shape == (2, 4, 10)


def test_non_sampler_progressbars(data_random_n100, monkeypatch):
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    model.build()
    progressbars = {"predict": [], "log_likelihood": []}

    def capture_predict(**kwargs):
        progressbars["predict"].append(kwargs["progressbar"])

    def capture_log_likelihood(**kwargs):
        progressbars["log_likelihood"].append(kwargs["progressbar"])

    monkeypatch.setattr(model.backend, "predict", capture_predict)
    monkeypatch.setattr(model.backend, "compute_log_likelihood", capture_log_likelihood)

    model.predict(idata=None)
    model.predict(idata=None, progressbar=True)
    model.compute_log_likelihood(idata=None)
    model.compute_log_likelihood(idata=None, progressbar=True)

    assert progressbars == {"predict": [False, True], "log_likelihood": [False, True]}


def test_non_sampler_operations_suppress_pymc_logs(data_random_n100, mock_pymc_sample, caplog):
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    idata = model.fit(draws=4, chains=2)

    caplog.clear()
    caplog.set_level(logging.INFO, logger="pymc")
    model.prior_predictive(draws=4)
    model.predict(idata, kind="response")
    model.compute_log_likelihood(idata)

    assert not any(
        record.name == "pymc.sampling.forward" and record.getMessage().startswith("Sampling:")
        for record in caplog.records
    )


def test_compute_log_likelihood_transformed_response(data_beetle, mock_pymc_sample):
    model = bmb.Model("prop(y, n) ~ x", data_beetle, family="binomial")
    idata = model.fit(draws=4, chains=2)
    assert {"y_data", "n_data"}.issubset(model.backend.model.named_vars)

    model.compute_log_likelihood(idata)
    assert idata.log_likelihood["y"].shape == (2, 4, len(data_beetle))

    same_data_result = model.compute_log_likelihood(idata, data=data_beetle, inplace=False)
    assert (same_data_result.log_likelihood["y"] == idata.log_likelihood["y"]).all()

    result = model.compute_log_likelihood(idata, data=data_beetle.head(3), inplace=False)
    assert result.log_likelihood["y"].shape == (2, 4, 3)


def test_predict_transformed_response_side_data(data_beetle, mock_pymc_sample):
    model = bmb.Model("prop(y, n) ~ x", data_beetle, family="binomial")
    idata = model.fit(draws=4, chains=2)

    assert {"y_data", "n_data"}.issubset(model.backend.model.named_vars)

    result = model.predict(idata, kind="response", data=data_beetle.head(3), inplace=False)
    samples = result.predictions["y"]
    assert samples.shape == (2, 4, 3)
    assert (samples <= data_beetle["n"].to_numpy()[:3][None, None, :]).all()

    model = bmb.Model("p(y, 62) ~ x", data_beetle, family="binomial")
    idata = model.fit(draws=4, chains=2)
    assert "y_data" in model.backend.model.named_vars

    result = model.predict(idata, kind="response", data=data_beetle.head(3), inplace=False)
    samples = result.predictions["y"]
    assert samples.shape == (2, 4, 3)
    assert (samples <= 62).all()


def test_predict_truncated_response_scalar_bounds(mock_pymc_sample):
    data = pd.DataFrame({"x": np.linspace(-1, 1, 8), "y": np.linspace(-0.5, 0.5, 8)})
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "x": bmb.Prior("Normal", mu=0, sigma=1),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }
    model = bmb.Model("truncated(y, -5, 5) ~ x", data, priors=priors)
    idata = model.fit(draws=4, chains=2)

    assert "y_data" in model.backend.model.named_vars

    result = model.predict(idata, kind="response_conditional", data=data.head(3), inplace=False)
    samples = result.predictions["y"]
    assert samples.shape == (2, 4, 3)
    assert (samples > -5).all()
    assert (samples < 5).all()


def test_out_of_sample_censored_response_predictions(mock_pymc_sample):
    data = pd.DataFrame(
        {
            "predictor": [-1.0, 0.0, 1.0],
            "y": [0.0, 0.5, 1.0],
            "status": ["left", "none", "right"],
        }
    )
    model = bmb.Model("censored(y, status) ~ predictor", data)
    idata = model.fit(draws=4, chains=2)

    response = model.response_term.label
    original_model = model.backend.model
    original_response = original_model[response]
    original_y_data = original_model["y_data"].get_value().copy()
    original_status_data = original_model["status_data"].get_value().copy()

    in_sample = model.predict(idata, kind="response", inplace=False, random_seed=1234)
    in_sample_samples = in_sample.posterior_predictive[response].to_numpy()
    assert in_sample_samples.shape == (2, 4, len(data))

    in_sample_conditional = model.predict(
        idata, kind="response_conditional", inplace=False, random_seed=1234
    )
    in_sample_conditional_samples = in_sample_conditional.posterior_predictive[response].to_numpy()
    assert (in_sample_conditional_samples[..., 0] <= data["y"].iloc[0]).all()
    assert (in_sample_conditional_samples[..., 2] >= data["y"].iloc[2]).all()

    result = model.predict(idata, data=data, kind="response", inplace=False, random_seed=1234)
    samples = result.predictions[response].to_numpy()

    assert samples.shape == (2, 4, len(data))

    conditional = model.predict(
        idata, data=data, kind="response_conditional", inplace=False, random_seed=1234
    )
    conditional_samples = conditional.predictions[response].to_numpy()
    assert (conditional_samples[..., 0] <= data["y"].iloc[0]).all()
    assert (conditional_samples[..., 2] >= data["y"].iloc[2]).all()
    assert model.backend.model is original_model
    assert model.backend.model[response] is original_response
    np.testing.assert_array_equal(model.backend.model["y_data"].get_value(), original_y_data)
    np.testing.assert_array_equal(
        model.backend.model["status_data"].get_value(), original_status_data
    )

    likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
    assert likelihood.log_likelihood[response].shape == (2, 4, len(data))

    missing_status = data.drop(columns="status")
    with pytest.raises(ValueError, match="Censored response log-likelihood requires variables"):
        model.compute_log_likelihood(idata, data=missing_status, inplace=False)


def test_out_of_sample_truncated_response_predictions(mock_pymc_sample):
    data = pd.DataFrame(
        {
            "predictor": [-1.0, 0.0, 1.0],
            "y": [-0.5, 0.0, 0.5],
            "lower": [-1.0, -1.5, -2.0],
            "upper": [1.0, 1.5, 2.0],
        }
    )
    model = bmb.Model("truncated(y, lower, upper) ~ predictor", data)
    idata = model.fit(draws=100, chains=2)

    response = model.response_term.label
    original_model = model.backend.model
    original_response = original_model[response]
    original_y_data = original_model["y_data"].get_value().copy()
    original_lower_data = original_model["lower_data"].get_value().copy()
    original_upper_data = original_model["upper_data"].get_value().copy()

    result = model.predict(idata, data=data, kind="response", inplace=False, random_seed=1234)
    samples = result.predictions[response].to_numpy()

    outside_bounds = (samples <= data["lower"].to_numpy()[None, None, :]) | (
        samples >= data["upper"].to_numpy()[None, None, :]
    )
    assert outside_bounds.any()

    conditional = model.predict(
        idata, data=data, kind="response_conditional", inplace=False, random_seed=1234
    )
    conditional_samples = conditional.predictions[response].to_numpy()
    assert (conditional_samples > data["lower"].to_numpy()[None, None, :]).all()
    assert (conditional_samples < data["upper"].to_numpy()[None, None, :]).all()

    assert model.backend.model is original_model
    assert model.backend.model[response] is original_response
    np.testing.assert_array_equal(model.backend.model["y_data"].get_value(), original_y_data)
    np.testing.assert_array_equal(
        model.backend.model["lower_data"].get_value(), original_lower_data
    )
    np.testing.assert_array_equal(
        model.backend.model["upper_data"].get_value(), original_upper_data
    )

    likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
    assert likelihood.log_likelihood[response].shape == (2, 100, len(data))

    missing_bounds = data.drop(columns="upper")
    result = model.predict(idata, data=missing_bounds, kind="response", inplace=False)
    assert result.predictions[response].shape == (2, 100, len(data))

    with pytest.raises(ValueError, match="Use kind='response' for unconditional predictions"):
        model.predict(idata, data=missing_bounds, kind="response_conditional", inplace=False)

    with pytest.raises(
        ValueError, match="Truncated response log-likelihood requires bound variables"
    ):
        model.compute_log_likelihood(idata, data=missing_bounds, inplace=False)


@pytest.mark.parametrize(
    ("family", "response"),
    [(None, [0.1, 0.2, 0.3]), ("gamma", [0.1, 0.2, 0.3]), ("poisson", [1, 2, 3])],
)
def test_untruncate_response_rebuilds_base_distribution(family, response):
    data = pd.DataFrame(
        {
            "predictor": [0.0, 1.0, 2.0],
            "y": response,
            "lower": [0.0, 0.0, 0.0],
            "upper": [3.0, 3.0, 3.0],
        }
    )
    kwargs = {} if family is None else {"family": family}
    model = bmb.Model("truncated(y, lower, upper) ~ predictor", data, **kwargs)
    model.build()

    response_name = model.response_term.label
    original = model.backend.model
    latent = _untruncate_response(response_name, original)

    assert latent is not original
    assert latent[response_name] in latent.observed_RVs
    assert "truncated" in str(original[response_name].owner.op).lower()
    assert "truncated" not in str(latent[response_name].owner.op).lower()


@pytest.mark.skip(reason="this example no longer trigger the fallback to adapt_diag")
def test_init_fallback(init_data, caplog):
    model = bmb.Model("od ~ temp + (1|source) + 0", init_data)
    with caplog.at_level(logging.INFO):
        model.fit(draws=100, init="auto")
        assert "Initializing NUTS using jitter+adapt_diag..." in caplog.text
        assert "The default initialization" in caplog.text
        assert "Initializing NUTS using adapt_diag..." in caplog.text


def test_2d_response_no_shape(mock_pymc_sample):
    """
    This tests whether a model where there's a single linear predictor and a response with
    response.ndim > 1 works well, without Bambi causing any shape problems.
    See https://github.com/bambinos/bambi/pull/629
    Updated https://github.com/bambinos/bambi/pull/632
    """

    def fn(name, p, observed, n, **kwargs):
        # Binomial responses expose successes and trials as separate inputs.
        # It's the users' responsibility to take only the observation dimension.
        kwargs["dims"] = kwargs.get("dims")[0]
        return pm.Binomial(name, p=p, n=n, observed=observed, **kwargs)

    likelihood = bmb.Likelihood("CustomBinomial", params=["p"], parent="p", dist=fn)
    link = bmb.Link("logit")
    family = bmb.Family("custom-binomial", likelihood, link)

    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    model = bmb.Model("prop(y, n) ~ x", data, family=family)
    model.build()
    assert_ip_dlogp(model)
    model.fit(chains=2)


def test_sparse_dot_univariate(mock_pymc_sample, monkeypatch):
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=6),
            "x1": rng.normal(size=6),
            "x2": [1, 1, 0, 0, 1, 1],
            "g1": ["a"] * 3 + ["b"] * 3,
            "g2": ["x", "x", "z", "z", "y", "y"],
        }
    )

    formula = "y ~ x1 + x2 + g1 + (g1|g2) + (x2|g2)"
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", False)
    model_dense = bmb.Model(formula, data)
    model_dense.build()

    monkeypatch.setattr(bmb.config, "SPARSE_DOT", True)
    model_sparse = bmb.Model(formula, data)
    model_sparse.build()

    assert graph_contains_op(model_sparse.backend.model["mu"], StructuredDot)

    logp_dense = model_dense.backend.model.compile_logp()
    dlogp_dense = model_dense.backend.model.compile_dlogp()
    ip_dense = model_dense.backend.model.initial_point()

    logp_sparse = model_sparse.backend.model.compile_logp()
    dlogp_sparse = model_sparse.backend.model.compile_dlogp()
    ip_sparse = model_sparse.backend.model.initial_point()

    # Keys of initial point are equal
    assert set(ip_dense) == set(ip_sparse)

    # Initial point values are equal
    for key in ip_dense:
        assert np.allclose(ip_dense[key], ip_sparse[key])

    # Initial logps and dlogps are equal
    assert np.allclose(logp_dense(ip_dense), logp_sparse(ip_sparse))
    assert np.allclose(dlogp_dense(ip_dense), dlogp_sparse(ip_sparse))

    idata_sparse = model_sparse.fit(chains=2)
    # NOTE: names for dense are tested elsewhere
    names = {
        "Intercept",
        "Intercept_centered",
        "x1",
        "x2",
        "g1[b]",
        "1|g2_sigma",
        "g1|g2_sigma[b]",
        "x2|g2_sigma",
        "sigma",
        "1|g2[x]",
        "1|g2[y]",
        "1|g2[z]",
        "g1|g2[x, b]",
        "g1|g2[y, b]",
        "g1|g2[z, b]",
        "x2|g2[x]",
        "x2|g2[y]",
        "x2|g2[z]",
    }
    assert set(az.summary(idata_sparse).index) == names


def test_sparse_dot_multivariate(data_inhaler, mock_pymc_sample, monkeypatch):
    formula = "rating ~ 1 + period + treat + (1 + treat|subject)"

    monkeypatch.setattr(bmb.config, "SPARSE_DOT", False)
    model_dense = bmb.Model(formula, data_inhaler, family="categorical")
    model_dense.build()

    monkeypatch.setattr(bmb.config, "SPARSE_DOT", True)
    model_sparse = bmb.Model(formula, data_inhaler, family="categorical")
    model_sparse.build()

    assert graph_contains_op(model_sparse.backend.model["p"], StructuredDot)

    logp_dense = model_dense.backend.model.compile_logp()
    dlogp_dense = model_dense.backend.model.compile_dlogp()
    ip_dense = model_dense.backend.model.initial_point()

    logp_sparse = model_sparse.backend.model.compile_logp()
    dlogp_sparse = model_sparse.backend.model.compile_dlogp()
    ip_sparse = model_sparse.backend.model.initial_point()

    # Keys of initial point are equal
    assert set(ip_dense) == set(ip_sparse)

    # Initial point values are equal
    for key in ip_dense:
        assert np.allclose(ip_dense[key], ip_sparse[key])

    # Initial logps and dlogps are equal
    assert np.allclose(logp_dense(ip_dense), logp_sparse(ip_sparse))
    assert np.allclose(dlogp_dense(ip_dense), dlogp_sparse(ip_sparse))

    idata_dense = model_dense.fit(chains=2)
    idata_sparse = model_sparse.fit(chains=2)
    assert set(az.summary(idata_dense).index) == set(az.summary(idata_sparse).index)


def test_sparse_dot_out_of_sample_prediction(mock_pymc_sample, monkeypatch):
    data = pd.DataFrame(
        {
            "y": [0.1, 0.3, -0.2, 0.5, 0.7, -0.4],
            "x": [1, 2, 3, 4, 5, 6],
            "group": ["a", "a", "b", "b", "c", "c"],
        }
    )
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", True)
    model = bmb.Model("y ~ x + (1 + x|group)", data)
    idata = model.fit(draws=4, chains=2)

    assert set(model.backend.model.named_vars) >= {
        "mu__group_specific_data",
        "mu__group_specific_indices",
        "mu__group_specific_indptr",
        "mu__group_specific_ncols",
    }
    assert "1|group_data" not in model.backend.model.named_vars
    assert "x|group_data" not in model.backend.model.named_vars

    result = model.predict(idata, data=data.head(3), kind="response", inplace=False)

    assert result.predictions["mu"].shape == (2, 4, 3)
    assert set(result.predictions_constant_data) >= {
        "mu__group_specific_data",
        "mu__group_specific_indices",
        "mu__group_specific_indptr",
        "mu__group_specific_ncols",
    }


def test_sparse_dot_out_of_sample_log_likelihood(mock_pymc_sample, monkeypatch):
    data = pd.DataFrame(
        {
            "y": [0.1, 0.3, -0.2, 0.5, 0.7, -0.4],
            "x": [1, 2, 3, 4, 5, 6],
            "group": ["a", "a", "b", "b", "c", "c"],
        }
    )
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", True)
    model = bmb.Model("y ~ x + (1 + x|group)", data)
    idata = model.fit(draws=4, chains=2)

    result = model.compute_log_likelihood(idata, data=data.head(3), inplace=False)

    assert result.log_likelihood["y"].shape == (2, 4, 3)


@pytest.mark.parametrize("noncentered", [True, False])
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_without_group_specific_effect(
    mock_pymc_sample, monkeypatch, noncentered, sparse_dot
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = pd.DataFrame(
        {
            "y": [0.1, -0.1, 0.2, -0.2],
            "group": ["a", "b", "a", "b"],
        }
    )
    model = bmb.Model("y ~ 1 + (1|group)", data, noncentered=noncentered)
    idata = model.fit(draws=4, chains=2)
    coefficients = idata.posterior["1|group"].copy(deep=True)
    reduced_model = remove_group_specific_contributions(
        model.backend._group_specific_state,
        model.backend.model,
    )

    assert "1|group" in model.backend.model.named_vars
    assert "1|group" not in reduced_model.named_vars
    if sparse_dot:
        assert "1|group_data" not in reduced_model.named_vars
    else:
        assert "group__idx" not in reduced_model.named_vars

    included = model.predict(idata, data=data, inplace=False)
    excluded = model.predict(idata, data=data, include_group_specific=False, inplace=False)
    excluded_in_sample = model.predict(idata, include_group_specific=False, inplace=False)

    model.predict(idata, data=data, include_group_specific=False)
    model.predict(idata, include_group_specific=False)

    assert not np.allclose(
        included.predictions["mu"].values, included.predictions["mu"].values[..., :1]
    )
    np.testing.assert_allclose(
        excluded.predictions["mu"].values - excluded.predictions["mu"].values[..., :1], 0
    )
    np.testing.assert_allclose(
        excluded_in_sample.posterior["mu"].values
        - excluded_in_sample.posterior["mu"].values[..., :1],
        0,
    )
    assert idata.posterior["1|group"].identical(coefficients)


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_prune_discards_group_specific_predictors(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = pd.DataFrame(
        {
            "y": [0.1, -0.1, 0.2, -0.2],
            "x": [1, 2, 3, 4],
            "group": ["a", "b", "a", "b"],
        }
    )
    model = bmb.Model("y ~ 1 + (1 + x|group)", data)
    model.build()

    assert not any(name.endswith("__selected") for name in model.backend.model.named_vars)
    assert not hasattr(model.backend.model, "__bambi_metadata__")
    assert "mu" in model.backend._group_specific_state.parameters

    reduced_model = remove_group_specific_contributions(
        model.backend._group_specific_state,
        model.backend.model,
    )

    if sparse_dot:
        assert "1|group_data" not in reduced_model.named_vars
        assert "x|group_data" not in reduced_model.named_vars
    else:
        assert "group__idx" not in reduced_model.named_vars
        assert "x_data" not in reduced_model.named_vars
    assert "1|group" not in reduced_model.named_vars
    assert "x|group" not in reduced_model.named_vars


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ 1 + (1|group) + (1|site)",
        "y ~ x + (1 + x|group)",
        "y ~ C(condition) + (C(condition)|group)",
        "y ~ 1 + hsgp(x, c=2, m=5) + (1|group)",
    ],
)
def test_predict_without_group_specific_effect_complex(mock_pymc_sample, monkeypatch, formula):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", False)
    data = pd.DataFrame(
        {
            "y": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            "condition": ["a", "b", "a", "b", "a", "b"],
            "group": ["a", "b", "a", "b", "a", "b"],
            "site": ["u", "u", "v", "v", "u", "v"],
        }
    )
    prediction_data = pd.DataFrame(
        {
            "y": [0.0] * 4,
            "x": [0.0] * 4,
            "condition": ["a"] * 4,
            "group": ["a", "b", "a", "b"],
            "site": ["u", "u", "v", "v"],
        }
    )
    model = bmb.Model(formula, data)
    idata = model.fit(draws=4, chains=2)

    included = model.predict(idata, data=prediction_data, inplace=False)
    excluded = model.predict(
        idata, data=prediction_data, include_group_specific=False, inplace=False
    )

    assert not np.allclose(
        included.predictions["mu"].values, included.predictions["mu"].values[..., :1]
    )
    np.testing.assert_allclose(
        excluded.predictions["mu"].values - excluded.predictions["mu"].values[..., :1], 0
    )


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_without_group_specific_effect_multivariate(
    mock_pymc_sample, monkeypatch, sparse_dot
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = pd.DataFrame(
        {
            "rating": pd.Categorical(["low", "medium", "high", "low", "medium", "high"]),
            "group": ["a", "b", "a", "b", "a", "b"],
        }
    )
    prediction_data = pd.DataFrame(
        {
            "rating": pd.Categorical(["low"] * 4, categories=data["rating"].cat.categories),
            "group": ["a", "b", "a", "b"],
        }
    )
    model = bmb.Model("rating ~ 1 + (1|group)", data, family="categorical")
    idata = model.fit(draws=4, chains=2)

    included = model.predict(idata, data=prediction_data, inplace=False)
    excluded = model.predict(
        idata, data=prediction_data, include_group_specific=False, inplace=False
    )

    assert not np.allclose(
        included.predictions["p"].values, included.predictions["p"].values[..., :1, :]
    )
    np.testing.assert_allclose(
        excluded.predictions["p"].values - excluded.predictions["p"].values[..., :1, :],
        0,
        atol=1e-15,
    )


# Nonlinear backend construction


@pytest.fixture
def nonlinear_ownership_data():
    return pd.DataFrame({"y": [0.2, 0.3, 0.4], "x": [1.0, 2.0, 3.0], "z": [0.0, 0.5, 1.0]})


@pytest.mark.parametrize("parameter", ["a", "sigma"])
@pytest.mark.parametrize("covariate", ["z", "`time value`"])
def test_nonlinear_model_preserves_additive_interactions(
    nonlinear_ownership_data, parameter, covariate
):
    data = nonlinear_ownership_data.assign(**{"time value": nonlinear_ownership_data.z})
    model = bmb.Model(
        bmb.Formula("y ~ a * x", f"{parameter} ~ x:{covariate}", nlpars=("a",)),
        data,
        center_predictors=False,
    )
    model.build()
    term_name = "x:z" if covariate == "z" else "x:time value"
    draws = xr.Dataset(
        {
            "a_Intercept" if parameter == "a" else "a": (("chain", "draw"), [[0.5]]),
            f"{parameter}_{term_name}": (("chain", "draw"), [[0.2]]),
        }
    )
    if parameter == "sigma":
        draws["sigma_Intercept"] = (("chain", "draw"), [[0.5]])
    else:
        draws["sigma"] = (("chain", "draw"), [[1.0]])
    new_data = data.iloc[:2].drop(columns="y")
    result = model.predict(
        xr.DataTree.from_dict({"posterior": draws}), data=new_data, inplace=False
    )
    predictor = 0.5 + 0.2 * new_data.x * new_data.z
    name = "mu" if parameter == "a" else "sigma"
    expected = predictor * new_data.x if parameter == "a" else np.exp(predictor)
    np.testing.assert_allclose(result.predictions[name], expected.to_numpy()[None, None, :])


@pytest.mark.parametrize("parameter", ["sigma", "b"])
@pytest.mark.parametrize(
    "expression, coefficient, expected",
    [("1 / a", 2.0, 0.5), ("-a", -2.0, 2.0), ("a / 2", 2.0, 1.0), ("a ** 0.5", 4.0, 2.0)],
)
def test_additional_nonlinear_arithmetic(
    nonlinear_ownership_data, parameter, expression, coefficient, expected
):
    data = nonlinear_ownership_data
    formula = bmb.Formula(
        "y ~ a * x" if parameter == "sigma" else "y ~ b * x",
        f"{parameter} ~ {expression}",
        nlpars=("a",) if parameter == "sigma" else ("a", "b"),
    )
    model = bmb.Model(formula, data)
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[coefficient]])})
    if parameter == "b":
        draws["sigma"] = (("chain", "draw"), [[1.0]])
    result = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    if parameter == "sigma":
        np.testing.assert_allclose(result.posterior.sigma, expected)
    else:
        np.testing.assert_allclose(result.posterior.mu, expected * data.x.to_numpy()[None, None, :])


def test_nonlinear_coefficients_are_owned_by_expression_parameters(nonlinear_ownership_data):
    data = nonlinear_ownership_data
    formula = bmb.Formula("y ~ a * x", "sigma ~ sqrt(a ** 2 + 0.1)", nlpars=("a",))

    model = bmb.Model(formula, data)

    mu = model.parameters["mu"]
    sigma = model.parameters["sigma"]
    assert isinstance(mu, ConditionalParameter)
    assert isinstance(sigma, ConditionalParameter)
    assert mu.is_nonlinear
    assert sigma.is_nonlinear
    assert model.parameter_graph.nodes["mu"] is mu
    assert model.parameter_graph.nodes["sigma"] is sigma
    assert set(mu.nonlinear_coefficients) == {"a"}
    assert set(sigma.nonlinear_coefficients) == {"a"}
    assert mu.nonlinear_coefficients["a"] is sigma.nonlinear_coefficients["a"]
    assert not hasattr(model, "_nonlinear_predictors")
    assert not hasattr(model, "nonlinear_predictors")
    assert not hasattr(model, "additive_parameters")


def test_nonlinear_auxiliary_owns_its_coefficient(nonlinear_ownership_data):
    formula = bmb.Formula(
        "y ~ b * x",
        "sigma ~ exp(sigma_b * x)",
        nlpars=("b", "sigma_b"),
    )

    model = bmb.Model(formula, nonlinear_ownership_data)
    model.build()

    mu = model.parameters["mu"]
    sigma = model.parameters["sigma"]
    assert set(mu.nonlinear_coefficients) == {"b"}
    assert set(sigma.nonlinear_coefficients) == {"sigma_b"}
    assert "b" in model.backend.model.named_vars
    assert "sigma_b" in model.backend.model.named_vars


def test_intermediate_expression_owns_its_direct_coefficients(nonlinear_ownership_data):
    data = nonlinear_ownership_data
    formula = bmb.Formula(
        "y ~ eta * x",
        "eta ~ a + b * z",
        nlpars=("eta", "b", "a"),
    )

    model = bmb.Model(formula, data)

    mu = model.parameters["mu"]
    eta = model.parameter_graph.nodes["eta"]
    assert eta not in model.parameters.values()
    assert mu.nonlinear_coefficients["eta"] is eta
    assert set(eta.nonlinear_coefficients) == {"a", "b"}
    assert model.parameter_graph.order.index("eta") < model.parameter_graph.order.index("mu")


def test_additive_and_marginal_likelihood_parameters_keep_their_roles(
    nonlinear_ownership_data,
):
    data = nonlinear_ownership_data
    conditional = bmb.Model(bmb.Formula("y ~ a * x", "sigma ~ z", nlpars=("a",)), data)
    marginal = bmb.Model(bmb.Formula("y ~ a * x", nlpars=("a",)), data)

    sigma = conditional.parameters["sigma"]
    assert isinstance(sigma, ConditionalParameter)
    assert not sigma.is_nonlinear
    assert set(sigma.terms) == {"Intercept", "z"}
    assert isinstance(marginal.parameters["sigma"], MarginalParameter)


@pytest.mark.parametrize("predictor", ["x", "x:z"])
@pytest.mark.parametrize("dependent_scale", [False, True])
def test_additive_parent_with_nonlinear_auxiliary(
    nonlinear_ownership_data, predictor, dependent_scale
):
    data = nonlinear_ownership_data
    sigma_formula = "sigma ~ sqrt(mu ** 2 + a ** 2)" if dependent_scale else "sigma ~ exp(a)"
    model = bmb.Model(
        bmb.Formula(f"y ~ {predictor}", sigma_formula, nlpars=("a",)),
        data,
        center_predictors=False,
    )
    model.build()
    assert not model.parameters["mu"].is_nonlinear
    assert model.parameters["sigma"].is_nonlinear
    draws = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), [[1.0]]),
            predictor: (("chain", "draw"), [[0.2]]),
            "a": (("chain", "draw"), [[0.3]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, data.iloc[:2].assign(x=[0.5, 1.5])):
        observations = data if new_data is None else new_data
        covariate = observations.x if predictor == "x" else observations.x * observations.z
        mu = 1.0 + 0.2 * covariate.to_numpy()
        sigma = np.sqrt(mu**2 + 0.3**2) if dependent_scale else np.exp(0.3)
        result = model.predict(
            idata, data=None if new_data is None else new_data.drop(columns="y"), inplace=False
        )
        group = result.posterior if new_data is None else result.predictions
        np.testing.assert_allclose(group.mu, mu[None, None, :])
        np.testing.assert_allclose(group.sigma, np.broadcast_to(sigma, group.sigma.shape))
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        expected = stats.norm.logpdf(observations.y, loc=mu, scale=sigma)
        np.testing.assert_allclose(likelihood.log_likelihood.y, expected[None, None, :])


def test_nonlinear_auxiliary_aligns_additive_parent_data():
    data = pd.DataFrame(
        {"y": [0.2, 0.4, 0.6, 0.8], "x": [np.nan, 1.0, 2.0, 3.0], "z": [0.0, np.nan, 0.5, 1.0]}
    )
    formula = bmb.Formula("y ~ x", "sigma ~ exp(a * z)", nlpars=("a",))
    model = bmb.Model(formula, data, dropna=True)
    model.build()
    pd.testing.assert_frame_equal(model.data, data.iloc[2:])
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[2:])
    assert_ip_dlogp(model)
    with pytest.raises(ValueError, match="incomplete rows"):
        bmb.Model(formula, data)
    draws = xr.Dataset({name: (("chain", "draw"), [[0.2]]) for name in ["Intercept", "x", "a"]})
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(
            xr.DataTree.from_dict({"posterior": draws}),
            data=pd.DataFrame({"x": [np.nan, 1.0], "z": [0.0, 0.5]}),
            inplace=False,
        )


def test_one_edge_parameter_dependency_matches_direct_calculation():
    data = pd.DataFrame({"y": [0.2, 0.3, 0.4], "x": [1.0, 2.0, 3.0]})
    formula = bmb.Formula("y ~ a * x", "sigma ~ a ** 2 + 0.1", nlpars=("a",))
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[0.5, -0.25]])})

    with model.backend.model:
        actual = pm.compute_deterministics(draws, var_names=["sigma"], progressbar=False)

    expected = (draws.a**2 + 0.1).values[..., None]
    np.testing.assert_allclose(actual.sigma, np.broadcast_to(expected, actual.sigma.shape))


@pytest.mark.usefixtures("mock_pymc_sample")
def test_intermediate_parameter_is_filtered_from_prior_and_posterior():
    data = pd.DataFrame({"y": [0.2, 0.3, 0.4], "x": [1.0, 2.0, 3.0]})
    formula = bmb.Formula("y ~ a * x", "a ~ b + 1", nlpars=("a", "b"))
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()

    prior = model.backend.prior_predictive(draws=2, prior_only=True, random_seed=123)
    idata = model.fit(draws=2, chains=1, include_response_params=True, random_seed=123)

    assert "a" not in prior.prior
    assert "a" not in idata.posterior
    assert {"b", "mu"} <= set(idata.posterior.data_vars)


@pytest.fixture
def nonlinear_parameter_dag_model():
    data = pd.DataFrame(
        {
            "distance": [0.0, 0.5, 1.0, 1.5],
            "attempts": [20, 40, 80, 160],
            "success_rate": [0.42, 0.38, 0.33, 0.29],
        }
    )
    formula = bmb.Formula(
        "success_rate ~ p_angle * p_distance",
        "sigma ~ sqrt(mu * (1 - mu) / attempts + sigma_y ** 2)",
        "p_distance ~ 1 + distance",
        nlpars=("p_angle", "p_distance", "sigma_y"),
    )
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()
    return model


@pytest.fixture
def nonlinear_parameter_dag_draws():
    return xr.Dataset(
        {
            "p_angle": (("chain", "draw"), [[0.8, 0.7]]),
            "p_distance_Intercept": (("chain", "draw"), [[0.55, 0.65]]),
            "p_distance_distance": (("chain", "draw"), [[-0.08, -0.12]]),
            "sigma_y": (("chain", "draw"), [[0.03, 0.05]]),
        }
    )


def golf_parameter_values(draws, data):
    distance = xr.DataArray(data.distance.to_numpy(), dims="__obs__")
    attempts = xr.DataArray(data.attempts.to_numpy(), dims="__obs__")
    p_distance = draws.p_distance_Intercept + draws.p_distance_distance * distance
    mu = draws.p_angle * p_distance
    sigma = np.sqrt(mu * (1 - mu) / attempts + draws.sigma_y**2)
    return mu, sigma


def test_nonlinear_parameter_dag_matches_golf_calculation(
    nonlinear_parameter_dag_model, nonlinear_parameter_dag_draws
):
    model = nonlinear_parameter_dag_model
    with model.backend.model:
        actual = pm.compute_deterministics(
            nonlinear_parameter_dag_draws, var_names=["mu", "sigma"], progressbar=False
        )
    expected_mu, expected_sigma = golf_parameter_values(nonlinear_parameter_dag_draws, model.data)

    np.testing.assert_allclose(actual.mu, expected_mu)
    np.testing.assert_allclose(actual.sigma, expected_sigma)


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_nonlinear_parameter_dag_prediction(
    nonlinear_parameter_dag_model, nonlinear_parameter_dag_draws, out_of_sample
):
    model = nonlinear_parameter_dag_model
    idata = xr.DataTree.from_dict({"posterior": nonlinear_parameter_dag_draws})
    data = (
        pd.DataFrame({"distance": [0.25, 1.25], "attempts": [30, 120]})
        if out_of_sample
        else model.data
    )

    result = model.predict(idata, data=data if out_of_sample else None, inplace=False)
    group = result.predictions if out_of_sample else result.posterior
    expected_mu, expected_sigma = golf_parameter_values(nonlinear_parameter_dag_draws, data)

    np.testing.assert_allclose(group.mu, expected_mu)
    np.testing.assert_allclose(group.sigma, expected_sigma)


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_nonlinear_parameter_dag_log_likelihood(
    nonlinear_parameter_dag_model, nonlinear_parameter_dag_draws, out_of_sample
):
    model = nonlinear_parameter_dag_model
    idata = xr.DataTree.from_dict({"posterior": nonlinear_parameter_dag_draws})
    data = (
        pd.DataFrame(
            {
                "distance": [0.25, 1.25],
                "attempts": [30, 120],
                "success_rate": [0.39, 0.30],
            }
        )
        if out_of_sample
        else model.data
    )

    result = model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    mu, sigma = golf_parameter_values(nonlinear_parameter_dag_draws, data)
    expected = norm.logpdf(data.success_rate.to_numpy(), loc=mu, scale=sigma)

    np.testing.assert_allclose(result.log_likelihood.success_rate, expected)


@pytest.fixture
def nonlinear_exponential_model():
    data = pd.DataFrame(
        {"x": [0.0, 0.5, 1.5, 3.0], "z": [-1.0, 0.0, 0.5, 2.0], "y": [2.0, 1.5, 1.0, 0.8]}
    )
    formula = bmb.Formula("y ~ a + b * exp(-k * x)", "a ~ 1 + z", nlpars=("a", "b", "k"))
    model = bmb.Model(
        formula,
        data,
        priors={
            "a": {
                "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
                "z": bmb.Prior("Normal", mu=0, sigma=1),
            },
            "b": bmb.Prior("Normal", mu=1, sigma=2),
            "k": bmb.Prior("LogNormal", mu=0, sigma=0.5),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        },
        center_predictors=False,
    )
    model.build()
    return model


def test_exponential_log_density_matches_direct_pymc(nonlinear_exponential_model):
    data = nonlinear_exponential_model.data
    with pm.Model(coords={"__obs__": np.arange(len(data))}) as reference:
        x = pm.Data("x", data["x"], dims="__obs__")
        z = pm.Data("z", data["z"], dims="__obs__")
        a_intercept = pm.Normal("a_Intercept", mu=0, sigma=2)
        a_z = pm.Normal("a_z", mu=0, sigma=1)
        b = pm.Normal("b", mu=1, sigma=2)
        k = pm.LogNormal("k", mu=0, sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = a_intercept + a_z * z + b * pm.math.exp(-k * x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=data["y"], dims="__obs__")

    actual_logp = nonlinear_exponential_model.backend.model.compile_logp()
    expected_logp = reference.compile_logp()
    for a_intercept, a_z, b, k, sigma in [(0.4, 0.2, 1.5, 0.8, 0.3), (-0.2, 0.5, 2, 1.2, 0.7)]:
        point = {
            "a_Intercept": np.array(a_intercept),
            "a_z": np.array(a_z),
            "b": np.array(b, dtype=float),
            "k_log__": np.log(k),
            "sigma_log__": np.log(sigma),
        }
        np.testing.assert_allclose(actual_logp(point), expected_logp(point))


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_exponential_log_likelihood_matches_normal(nonlinear_exponential_model, out_of_sample):
    posterior = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.4, -0.2]]),
            "a_z": (("chain", "draw"), [[0.2, 0.5]]),
            "b": (("chain", "draw"), [[1.5, 2.0]]),
            "k": (("chain", "draw"), [[0.8, 1.2]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    data = (
        pd.DataFrame({"x": [0.2, 2.5], "z": [1.5, -0.5], "y": [2.2, 0.3]})
        if out_of_sample
        else nonlinear_exponential_model.data
    )
    result = nonlinear_exponential_model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    x = xr.DataArray(data["x"].to_numpy(), dims="__obs__")
    z = xr.DataArray(data["z"].to_numpy(), dims="__obs__")
    mu = posterior["a_Intercept"] + posterior["a_z"] * z
    mu += posterior["b"] * np.exp(-posterior["k"] * x)
    expected = norm.logpdf(data["y"].to_numpy(), loc=mu, scale=posterior["sigma"].values[..., None])

    np.testing.assert_allclose(result.log_likelihood["y"], expected)
    assert "log_likelihood" not in idata


def test_zero_predictor_broadcasts_for_new_observations():
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.1, 1.2, 2.1]})
    model = bmb.Model(bmb.Formula("y ~ 0 + nl(a + x)", "a ~ 0", nlpars=("a",)), data)
    model.build()
    idata = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"sigma": (("chain", "draw"), [[0.2]])})}
    )

    fitted = model.predict(idata, inplace=False)
    predicted = model.predict(idata, data=pd.DataFrame({"x": [3.0, 4.0]}), inplace=False)

    np.testing.assert_allclose(fitted.posterior["mu"], [[[0.0, 1.0, 2.0]]])
    np.testing.assert_allclose(predicted.predictions["mu"], [[[3.0, 4.0]]])


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_multiple_nonlinear_summands_share_parameter(out_of_sample):
    data = pd.DataFrame({"x": [0.0, 0.5, 1.5, 3.0], "y": [2.0, 1.5, 1.0, 0.8]})
    formula = bmb.Formula(
        "y ~ a * exp(-k * x) + b * exp(-2 * k * x)",
        "a ~ 1",
        "b ~ 1",
        "k ~ 1",
        nlpars=("a", "b", "k"),
    )
    model = bmb.Model(formula, data)
    model.build()
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), [[0.4, 1.2]]),
            "b": (("chain", "draw"), [[1.5, 2.0]]),
            "k": (("chain", "draw"), [[0.8, 1.2]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = pd.DataFrame({"x": [0.2, 2.5]}) if out_of_sample else data
    result = model.predict(idata, data=prediction_data if out_of_sample else None, inplace=False)
    x = xr.DataArray(prediction_data["x"].to_numpy(), dims="__obs__")
    a = posterior["a"]
    b = posterior["b"]
    k = posterior["k"]
    expected = a * np.exp(-k * x) + b * np.exp(-2 * k * x)
    actual = result.predictions["mu"] if out_of_sample else result.posterior["mu"]

    assert actual.dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_nonlinear_parameter_offset_prediction(out_of_sample):
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.5, 3.0],
            "z": [-1.0, 0.0, 0.5, 2.0],
            "exposure": [0.2, 0.5, 1.0, 1.5],
            "y": [0.1, 1.5, 3.0, 5.8],
        }
    )
    formula = bmb.Formula("y ~ exp(a) * x", "a ~ 1 + z + offset(exposure)", nlpars=("a",))
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()
    posterior = xr.Dataset(
        {
            "a_Intercept": (("chain", "draw"), [[0.4, -0.2]]),
            "a_z": (("chain", "draw"), [[0.2, 0.5]]),
            "sigma": (("chain", "draw"), [[0.3, 0.7]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = (
        pd.DataFrame({"x": [0.2, 2.5], "z": [1.5, -0.5], "exposure": [0.7, 2.0]})
        if out_of_sample
        else data
    )
    result = model.predict(idata, data=prediction_data if out_of_sample else None, inplace=False)
    x = xr.DataArray(prediction_data["x"].to_numpy(), dims="__obs__")
    z = xr.DataArray(prediction_data["z"].to_numpy(), dims="__obs__")
    exposure = xr.DataArray(prediction_data["exposure"].to_numpy(), dims="__obs__")
    a = posterior["a_Intercept"] + posterior["a_z"] * z + exposure
    expected = np.exp(a) * x
    actual = result.predictions["mu"] if out_of_sample else result.posterior["mu"]

    assert actual.dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(actual, expected)


@pytest.fixture(name="golf_model")
def make_golf_model():
    data = pd.DataFrame(
        {
            "distance": [2.0, 3.0, 4.0, 5.0],
            "attempts": [1443, 694, 455, 353],
            "successes": [1346, 577, 337, 208],
            "ball_radius": np.repeat((1.68 / 2) / 12, 4),
            "hole_radius": np.repeat((4.25 / 2) / 12, 4),
        }
    )
    formula = bmb.Formula(
        "prop(successes, attempts) ~ "
        "2 * normal_cdf(asin((hole_radius - ball_radius) / distance) / sigma_angle) - 1",
        nlpars=("sigma_angle",),
    )
    model = bmb.Model(
        formula,
        data,
        family="binomial",
        link="identity",
        priors={
            "sigma_angle": bmb.Prior("HalfNormal", sigma=0.5),
        },
    )
    model.build()
    return model


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_golf_probability_and_log_likelihood(golf_model, out_of_sample):
    posterior = xr.Dataset({"sigma_angle": (("chain", "draw"), [[0.02, 0.03]])})
    idata = xr.DataTree.from_dict({"posterior": posterior})
    data = golf_model.data
    if out_of_sample:
        data = pd.DataFrame(
            {
                "distance": [6.0, 10.0],
                "attempts": [272, 200],
                "successes": [149, 67],
                "ball_radius": np.repeat((1.68 / 2) / 12, 2),
                "hole_radius": np.repeat((4.25 / 2) / 12, 2),
            }
        )

    result = golf_model.predict(idata, data=data if out_of_sample else None, inplace=False)
    group = result.predictions if out_of_sample else result.posterior
    threshold = np.arcsin((data["hole_radius"] - data["ball_radius"]) / data["distance"])
    probability = (
        2 * norm.cdf(threshold.to_numpy()[None, :] / posterior["sigma_angle"].to_numpy()[..., None])
        - 1
    )

    np.testing.assert_allclose(group["p"], probability)
    result = golf_model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    expected = pm.logp(
        pm.Binomial.dist(n=data["attempts"].to_numpy(), p=probability),
        data["successes"].to_numpy(),
    ).eval()
    np.testing.assert_allclose(result.log_likelihood["successes"], expected, rtol=1e-6)


def test_nonlinear_binomial_literal_trials():
    data = pd.DataFrame({"successes": [6, 13, 18], "x": [0.1, 0.2, 0.3]})
    model = bmb.Model(
        bmb.Formula("p(successes, 62) ~ normal_cdf(a + b * x)", nlpars=("a", "b")),
        data,
        family="binomial",
        link="identity",
    )
    model.build()

    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), [[-0.4, 0.3]]),
            "b": (("chain", "draw"), [[0.8, -0.2]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})

    prediction = model.predict(idata, inplace=False)
    x = xr.DataArray(data.x.to_numpy(), dims="__obs__")
    expected_probability = ndtr(posterior.a + posterior.b * x)
    np.testing.assert_allclose(prediction.posterior.p, expected_probability)

    likelihood = model.compute_log_likelihood(idata, inplace=False)
    expected_likelihood = pm.logp(
        pm.Binomial.dist(n=62, p=expected_probability.values), data.successes.to_numpy()
    ).eval()
    np.testing.assert_allclose(likelihood.log_likelihood.successes, expected_likelihood)


def test_nonlinear_beta_binomial_proportion_prediction_and_likelihood():
    data = pd.DataFrame({"successes": [6, 13, 18], "attempts": [59, 60, 62], "x": [0.1, 0.2, 0.3]})
    model = bmb.Model(
        bmb.Formula("prop(successes, attempts) ~ a + b * x", nlpars=("a", "b")),
        data,
        family="beta_binomial",
        priors={"kappa": 10.0},
    )
    model.build()
    posterior = xr.Dataset(
        {
            "a": (("chain", "draw"), [[-0.4, 0.3]]),
            "b": (("chain", "draw"), [[0.8, -0.2]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})

    prediction = model.predict(idata, inplace=False)
    x = xr.DataArray(data.x.to_numpy(), dims="__obs__")
    expected_mu = expit(posterior.a + posterior.b * x)
    np.testing.assert_allclose(prediction.posterior.mu, expected_mu)

    likelihood = model.compute_log_likelihood(idata, inplace=False)
    expected_likelihood = pm.logp(
        pm.BetaBinomial.dist(
            n=data.attempts.to_numpy(),
            alpha=expected_mu.values * 10,
            beta=(1 - expected_mu.values) * 10,
        ),
        data.successes.to_numpy(),
    ).eval()
    np.testing.assert_allclose(likelihood.log_likelihood.successes, expected_likelihood)


# Nonlinear links and predictor transforms


@pytest.mark.parametrize(
    "family, link, inverse_link, distribution, parent, auxiliary",
    [
        ("poisson", "log", np.exp, pm.Poisson, "mu", {}),
        ("bernoulli", "logit", expit, pm.Bernoulli, "p", {}),
        ("bernoulli", "probit", ndtr, pm.Bernoulli, "p", {}),
        ("bernoulli", "cloglog", lambda x: -np.expm1(-np.exp(x)), pm.Bernoulli, "p", {}),
        ("gaussian", "identity", lambda x: x, pm.Normal, "mu", {"sigma": 1.0}),
        (
            "poisson",
            bmb.Link("scaled_log", inverse_link=lambda x: pm.math.exp(x / 2)),
            lambda x: np.exp(x / 2),
            pm.Poisson,
            "mu",
            {},
        ),
    ],
)
def test_parent_link_matches_pymc_and_prediction(
    family, link, inverse_link, distribution, parent, auxiliary
):
    data = pd.DataFrame({"y": [0, 1, 0, 1], "x": [-1.0, -0.3, 0.2, 0.7]})
    model = bmb.Model(
        bmb.Formula("y ~ a + b * x ** 2", nlpars=("a", "b")),
        data,
        family=family,
        link={parent: link},
        priors=auxiliary,
    )
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[-0.4, 0.3]]),
            "b": (("chain", "draw"), [[0.8, -0.2]]),
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    with model.backend.model:
        actual = pm.compute_deterministics(draws, progressbar=False)
    assert "a" not in actual
    assert model.backend.model.named_vars_to_dims["a"] == ()
    for new_data in (None, pd.DataFrame({"x": [0.1, 1.4], "y": [1, 0]})):
        prediction_data = data if new_data is None else new_data
        eta = np.array([-0.4, 0.3])[:, None] + np.array([0.8, -0.2])[:, None] * (
            prediction_data.x.to_numpy() ** 2
        )
        expected = inverse_link(eta)
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        np.testing.assert_allclose(group[parent].values, expected[None])
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        direct = pm.logp(
            distribution.dist(**{parent: expected}, **auxiliary), prediction_data.y.to_numpy()
        ).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.y.values, direct[None])


def test_link_preserves_likelihood_parameter_transform():
    data = pd.DataFrame({"y": [0.2, 0.7], "x": [-0.5, 0.5]})
    model = bmb.Model(
        bmb.Formula("y ~ a * x", nlpars=("a",)),
        data,
        family="beta",
        priors={"kappa": 4.0},
    )
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[1.2]])})
    result = model.compute_log_likelihood(
        xr.DataTree.from_dict({"posterior": draws}), inplace=False
    )
    mu = expit(1.2 * data.x.to_numpy())
    expected = pm.logp(pm.Beta.dist(alpha=mu * 4, beta=(1 - mu) * 4), data.y).eval()
    np.testing.assert_allclose(result.log_likelihood.y.values, [[expected]])


def test_scalar_predictor_transform_receives_auxiliary_parameters(monkeypatch):
    data = pd.DataFrame({"y": [0.0, 1.0], "x": [-0.5, 0.5]})
    model = bmb.Model(bmb.Formula("y ~ a * x", nlpars=("a",)), data, priors={"sigma": 2.0})
    monkeypatch.setitem(
        transforms_registry.additive_predictors,
        (type(model.family), "mu"),
        lambda value, parameters, inverse_link: inverse_link(value + parameters["sigma"]),
    )
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[1.2]])})
    result = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    np.testing.assert_allclose(result.posterior.mu.values, [[[1.4, 2.6]]])


@pytest.mark.parametrize("transform", ["inverse_link", "predictor"])
def test_scalar_transform_result_broadcasts_for_prediction(monkeypatch, transform):
    data = pd.DataFrame({"y": [0.0, 1.0], "x": [-0.5, 0.5]})
    link = {"mu": bmb.Link("constant", inverse_link=lambda value: 1.0)}
    model = bmb.Model(
        bmb.Formula("y ~ a * x", nlpars=("a",)),
        data,
        priors={"sigma": 1.0},
        link=link if transform == "inverse_link" else None,
    )
    if transform == "predictor":
        monkeypatch.setitem(
            transforms_registry.additive_predictors,
            (type(model.family), "mu"),
            lambda value, parameters, inverse_link: 1.0,
        )
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[1.2]])})
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, pd.DataFrame({"x": [-1.0, 0.0, 1.0]})):
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        size = len(data) if new_data is None else len(new_data)
        np.testing.assert_array_equal(group.mu.values, np.ones((1, 1, size)))


@pytest.mark.parametrize("family", ["categorical", "cumulative", "sratio"])
def test_vector_parent_links_remain_rejected(family):
    with pytest.raises(ValueError, match="scalar parent parameter"):
        bmb.Model(
            bmb.Formula("y ~ a * x", nlpars=("a",)),
            pd.DataFrame({"y": [0, 1, 2], "x": [0, 1, 2]}),
            family=family,
        )


@pytest.mark.parametrize("family, inverse_link", [("gaussian", lambda x: x), ("poisson", np.exp)])
@pytest.mark.parametrize("predictor", ["a ~ 0", "a ~ 1"])
def test_intercept_only_prediction_accepts_row_only_data(family, inverse_link, predictor):
    model = bmb.Model(
        bmb.Formula("y ~ a + b", predictor, nlpars=("a", "b")),
        pd.DataFrame({"y": [0, 1]}),
        family=family,
        priors={"sigma": 1.0} if family == "gaussian" else None,
    )
    model.build()
    value = 0.5 if predictor == "a ~ 1" else 0.0
    draws = xr.Dataset({"b": (("chain", "draw"), [[0.2]])})
    if predictor == "a ~ 1":
        draws["a"] = (("chain", "draw"), [[value]])
    idata = xr.DataTree.from_dict({"posterior": draws})
    result = model.predict(idata, data=pd.DataFrame(index=range(3)), inplace=False)
    np.testing.assert_allclose(
        result.predictions.mu.values, np.full((1, 1, 3), inverse_link(value + 0.2))
    )
    with pytest.raises(ValueError, match="does not contain any complete observation"):
        model.predict(idata, data=pd.DataFrame(), inplace=False)


def test_invalid_family_link_remains_rejected():
    with pytest.raises(ValueError, match="cannot be used"):
        bmb.Model(
            bmb.Formula("y ~ a", nlpars=("a",)),
            pd.DataFrame({"y": [0, 1]}),
            family="poisson",
            link="logit",
        )


def test_bare_nonlinear_coefficient_broadcasts_without_data_columns():
    model = bmb.Model(
        bmb.Formula("y ~ a", nlpars=("a",)),
        pd.DataFrame({"y": [0, 1]}),
        priors={"sigma": 1.0},
    )
    model.build()
    draws = xr.Dataset({"a": (("chain", "draw"), [[0.5]])})
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in (None, pd.DataFrame(index=range(3))):
        result = model.predict(idata, data=new_data, inplace=False)
        group = result.posterior if new_data is None else result.predictions
        size = 2 if new_data is None else 3
        np.testing.assert_allclose(group.mu.values, np.full((1, 1, size), 0.5))
        assert ("a" in group) is (new_data is None)


def test_dependency_graph_contains_all_quantity_roles(nonlinear_ownership_data):
    model = bmb.Model(
        bmb.Formula("y ~ a + c * x", "a ~ b + c * z + sigma", "c ~ 1 + z", nlpars=("a", "b", "c")),
        nonlinear_ownership_data,
        center_predictors=False,
    )
    nodes = model.parameter_graph.nodes
    assert set(nodes) == {"mu", "sigma", "a", "b", "c"}
    assert set(model.parameters) == {"mu", "sigma"}
    assert isinstance(nodes["mu"], ConditionalParameter)
    assert isinstance(nodes["sigma"], MarginalParameter)
    assert isinstance(nodes["a"], ConditionalCoefficient)
    assert isinstance(nodes["b"], MarginalCoefficient)
    assert isinstance(nodes["c"], ConditionalCoefficient)
    assert isinstance(nodes["a"], Conditional)
    assert isinstance(nodes["b"], Marginal)
    assert nodes["mu"].nonlinear_coefficients["a"] is nodes["a"]
    assert nodes["mu"].nonlinear_coefficients["c"] is nodes["a"].nonlinear_coefficients["c"]
    for name in ("a", "b", "c"):
        assert not hasattr(nodes[name], "is_parent")
        assert not hasattr(nodes[name], "link")
    order = model.parameter_graph.order
    for name, dependencies in model.parameter_graph.dependencies.items():
        assert all(order.index(dependency) < order.index(name) for dependency in dependencies)
    model.build()
    draws = xr.Dataset(
        {
            name: (("chain", "draw"), [[value]])
            for name, value in {"b": 0.4, "c_Intercept": 0.2, "c_z": 0.3, "sigma": 0.5}.items()
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    for data in (None, nonlinear_ownership_data.iloc[:2].assign(x=0.7, z=0.2)):
        result = model.predict(idata, data=data, inplace=False)
        observations = model.data if data is None else data
        posterior = result.posterior if data is None else result.predictions
        c = 0.2 + 0.3 * observations.z
        expected = 0.4 + c * observations.z + 0.5 + c * observations.x
        np.testing.assert_allclose(posterior.mu, expected.to_numpy()[None, None, :])


@pytest.mark.parametrize(
    "rhs, expected_terms, expected_covariates",
    [
        ("nl(exp(b * x) + x) + z", {"Intercept", "z"}, {"x", "z"}),
        (
            "z - exp(b * x) + (1 + x | group)",
            {"Intercept", "z", "1|group", "x|group"},
            {"x", "z", "group"},
        ),
        ("0 + nl(b * x) + x:z", {"x:z"}, {"x", "z"}),
        ("nl(b * x) - 1", set(), {"x"}),
    ],
)
def test_mixed_predictor_terms_and_ownership(rhs, expected_terms, expected_covariates):
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [1.0, 0.0, 3.0, 2.0],
            "group": ["a", "a", "b", "b"],
        }
    )
    model = bmb.Model(bmb.Formula(f"y ~ {rhs}", nlpars=("b",)), data)
    parent = model.parameters["mu"]
    assert set(parent.terms) == expected_terms
    assert set(parent.nonlinear_coefficients) == {"b"}
    assert parent.data_names == ("x",)
    assert set(model.get_covariates()) == expected_covariates
    model.build()
    assert "nl" not in model.backend.model.named_vars


def test_mixed_predictor_missing_rows_are_aligned():
    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, np.nan, 2.0, 3.0], "z": [1.0, 0.0, np.nan, 2.0]}
    )
    formula = bmb.Formula("y ~ nl(b * x) + z", nlpars=("b",))
    with pytest.raises(ValueError, match="2 incomplete rows"):
        bmb.Model(formula, data)
    model = bmb.Model(formula, data, dropna=True)
    assert list(model.data.index) == [0, 3]
    assert len(model.parameters["mu"].design.common.design_matrix) == 2
    model.build()
