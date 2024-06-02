import logging

from functools import reduce
from operator import add
from os.path import dirname, join

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from formulae import design_matrices

import bambi as bmb
from bambi.terms import CommonTerm, GroupSpecificTerm


@pytest.fixture(scope="module")
def data_numeric_xy():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.normal(size=100),
        }
    )
    return data


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "diabetes.txt"), sep="\t")
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


@pytest.fixture(scope="module")
def crossed_data():
    """
    Group specific effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    Common effects:
    A continuous predictor, a numeric dummy, and a three-level category
    (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both
    dummies)
    Sites crossed with threecats
    """
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "crossed_random.csv"))
    return data


@pytest.fixture(scope="module")
def init_data():
    """
    Data used to test initialization method
    """
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "obs.csv"))
    return data


def test_term_init(diabetes_data):
    design = design_matrices("BMI", diabetes_data)
    term = design.common.terms["BMI"]
    term = CommonTerm(term, prior=None)
    assert term.name == "BMI"
    assert not term.categorical
    assert term.levels is None
    assert term.data.shape == (442,)


def test_distribute_group_specific_effect_over(diabetes_data):
    # 163 unique levels of BMI in diabetes_data
    # With intercept
    model = bmb.Model("BP ~ (C(age_grp)|BMI)", diabetes_data)

    # Treatment encoding because of the intercept
    levels = sorted(list(diabetes_data["age_grp"].unique()))[1:]
    levels = [str(level) for level in levels]
    parent_component = model.components[model.family.likelihood.parent]
    assert "C(age_grp)|BMI" in parent_component.terms
    assert "1|BMI" in parent_component.terms
    assert parent_component.terms["C(age_grp)|BMI"].coords["C(age_grp)__expr_dim"] == levels

    # This is equal to the sub-matrix of Z that corresponds to this term.
    # 442 is the number of observations. 163 the number of groups.
    # 2 is the number of levels of the categorical variable 'C(age_grp)' after removing
    # the reference level. Then the number of columns is 326 = 163 * 2.
    assert parent_component.terms["C(age_grp)|BMI"].data.shape == (442, 326)

    # Without intercept. Reference level is not removed.
    model = bmb.Model("BP ~ (0 + C(age_grp)|BMI)", diabetes_data)
    parent_component = model.components[model.family.likelihood.parent]
    assert "C(age_grp)|BMI" in parent_component.terms
    assert not "1|BMI" in parent_component.terms
    assert parent_component.terms["C(age_grp)|BMI"].data.shape == (442, 489)


def test_model_init_bad_data():
    with pytest.raises(ValueError):
        bmb.Model("y ~ x", {"x": 1})


def test_unbuilt_model(diabetes_data):
    model = bmb.Model("Y ~ AGE", data=diabetes_data)
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
    assert model.components[model.family.likelihood.parent].terms["x"].categorical

    model = bmb.Model("y ~ 0 + x*z", data, categorical=["x", "z"])
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["x"].categorical
    assert parent_component.terms["z"].categorical
    assert parent_component.terms["x:z"].categorical


def test_model_no_response():
    with pytest.raises(ValueError):
        bmb.Model("x", pd.DataFrame({"x": [1]}))


def test_model_term_names_property(diabetes_data):
    model = bmb.Model("BMI ~ age_grp + BP + S1", diabetes_data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.intercept_term.name == "Intercept"
    assert set(parent_component.common_terms) == {"age_grp", "BP", "S1"}


def test_model_term_names_property_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = bmb.Model("Y ~ threecats*fourcats", crossed_data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.intercept_term.name == "Intercept"
    assert set(parent_component.common_terms) == {
        "threecats",
        "fourcats",
        "threecats:fourcats",
    }


def test_model_terms_levels_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = bmb.Model("Y ~ threecats*fourcats", crossed_data)

    assert model.components[model.family.likelihood.parent].terms["threecats:fourcats"].levels == [
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
            "z": reduce(add, [[f"Group {x}"] * 10 for x in ["1", "2", "3", "1", "2"]]),
            "time": list(range(1, 11)) * 5,
            "subject": reduce(add, [[f"Subject {x}"] * 10 for x in range(1, 6)]),
        }
    )
    model = bmb.Model("y ~ x + z + time + (time|subject)", data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["z"].levels == ["Group 2", "Group 3"]
    assert parent_component.terms["1|subject"].groups == [
        f"Subject {x}" for x in range(1, 6)
    ]
    assert parent_component.terms["time|subject"].groups == [
        f"Subject {x}" for x in range(1, 6)
    ]


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

    parent_component = model.components[model.family.likelihood.parent]
    assert isinstance(parent_component.terms["x"], CommonTerm)
    assert isinstance(parent_component.terms["g"], CommonTerm)
    assert isinstance(parent_component.terms["x:g"], CommonTerm)
    assert isinstance(parent_component.terms["1|s"], GroupSpecificTerm)
    assert isinstance(parent_component.terms["x|s"], GroupSpecificTerm)

    # Also check 'categorical' attribute is right
    assert parent_component.terms["g"].categorical


def test_one_shot_formula_fit(diabetes_data):
    model = bmb.Model("S3 ~ S1 + S2", diabetes_data)
    model.fit(tune=100, draws=100)
    named_vars = model.backend.model.named_vars
    targets = ["S3", "S1", "Intercept"]
    assert len(set(named_vars.keys()) & set(targets)) == 3


def test_categorical_term():
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
    fitted = model.fit(tune=100, draws=100)
    df = az.summary(fitted)
    names = {
        "Intercept",
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
        "g1|g2[b, x]",
        "g1|g2[b, y]",
        "g1|g2[b, z]",
        "x2|g2[x]",
        "x2|g2[y]",
        "x2|g2[z]",
    }
    assert set(df.index) == names


def test_omit_offsets_false():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x1": rng.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = bmb.Model("y ~ x1 + (x1|g1)", data)
    fitted = model.fit(tune=100, draws=100, omit_offsets=False)
    offsets = [var for var in fitted.posterior.var() if var.endswith("_offset")]
    assert offsets == ["1|g1_offset", "x1|g1_offset"]


def test_omit_offsets_true():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x1": rng.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = bmb.Model("y ~ x1 + (x1|g1)", data)
    fitted = model.fit(tune=100, draws=100, omit_offsets=True)
    offsets = [var for var in fitted.posterior.var() if var.endswith("_offset")]
    assert not offsets


def test_hyperprior_on_common_effect():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x1": rng.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    slope = bmb.Prior("Normal", mu=0, sd=bmb.Prior("HalfCauchy", beta=2))

    priors = {"x1": slope}
    with pytest.raises(ValueError):
        bmb.Model("y ~ x1 + (x1|g1)", data, priors=priors)

    priors = {"common": slope}
    with pytest.raises(ValueError):
        bmb.Model("y ~ x1 + (x1|g1)", data, priors=priors)


@pytest.mark.parametrize(
    "family",
    [
        "asymmetriclaplace",
        "gaussian",
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


def test_links():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "g": rng.choice([0, 1], size=100),
            "y": rng.integers(3, 10, size=100),
            "x": rng.integers(3, 10, size=100),
        }
    )

    FAMILIES = {
        "asymmetriclaplace": ["identity", "log", "inverse"],
        "bernoulli": ["identity", "logit", "probit", "cloglog"],
        "beta": ["logit", "probit", "cloglog"],
        "gamma": ["identity", "inverse", "log"],
        "gaussian": ["identity", "log", "inverse"],
        "negativebinomial": ["identity", "log", "cloglog"],
        "poisson": ["identity", "log"],
        "vonmises": ["identity", "tan_2"],
        "wald": ["inverse", "inverse_squared", "identity", "log"],
    }
    for family, links in FAMILIES.items():
        for link in links:
            if family == "bernoulli":
                bmb.Model("g ~ x", data, family=family, link=link)
            else:
                bmb.Model("y ~ x", data, family=family, link=link)


def test_bad_links():
    """Passes names of links that are not suitable for the family."""
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "g": rng.choice([0, 1], size=100),
            "y": rng.integers(3, 10, size=100),
            "x": rng.integers(3, 10, size=100),
        }
    )
    FAMILIES = {
        "bernoulli": ["inverse", "inverse_squared", "log"],
        "beta": ["inverse", "inverse_squared", "log"],
        "gamma": ["logit", "probit", "cloglog"],
        "gaussian": ["logit", "probit", "cloglog"],
        "negativebinomial": ["logit", "probit", "inverse", "inverse_squared"],
        "poisson": ["logit", "probit", "cloglog", "inverse", "inverse_squared"],
        "vonmises": ["logit", "probit", "cloglog"],
        "wald": ["logit", "probit", "cloglog"],
    }

    for family, links in FAMILIES.items():
        for link in links:
            with pytest.raises(ValueError):
                if family == "bernoulli":
                    formula = "g ~ x"
                else:
                    formula = "y ~ x"
                bmb.Model(formula, data, family=family, link=link)


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


def test_1d_group_specific():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=40),
            "x": rng.choice(["A", "B"], size=40),
            "g": ["A", "B", "C", "D"] * 10,
        }
    )
    # Since there's 1|g, there's only one column for x|g
    # We need to ensure x|g is of shape (40,) and not of shape (40, 1)
    # We do so by checking the mean is (40, ) because shape of x|g still returns (40, 1)
    # The difference is that we do .squeeze() on it after creation.
    model = bmb.Model("y ~ (x|g)", data)
    model.build()
    assert model.backend.components["mu"].output.shape.eval() == (40,)


def test_data_is_copied():
    adults = bmb.load_data("adults")

    model_1 = bmb.Model("age ~ sex * race", adults)
    model_2 = bmb.Model("age ~ sex * race", adults, categorical=["age", "sex"])

    for model in [model_1, model_2]:
        assert id(adults) != id(model.data)
        assert all(model.data.dtypes[:3] == "category")

    assert all(adults.dtypes[:3] == "object")


def test_response_is_censored():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "status": ["none", "right", "interval", "left", "none"],
        }
    )
    dm = bmb.Model("censored(x, status) ~ 1", df)
    assert dm.response_component.term.is_censored is True


def test_response_is_truncated():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    dm = bmb.Model("truncated(x, 5.5) ~ 1", df)
    assert dm.response_component.term.is_truncated is True


def test_custom_likelihood_function():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [1, 1, 2, 2, 3]})

    def CustomGaussian(*args, **kwargs):
        return pm.Normal(*args, **kwargs)

    sigma_prior = bmb.Prior("HalfNormal", sigma=1)
    likelihood = bmb.Likelihood(
        "CustomGaussian", params=["mu", "sigma"], parent="mu", dist=CustomGaussian
    )
    family = bmb.Family("custom_gaussian", likelihood, "identity")
    model = bmb.Model("y ~ x", df, family=family, priors={"sigma": sigma_prior})
    _ = model.fit(tune=100, draws=100)
    assert model.backend.model.observed_RVs[0].str_repr() == "y ~ Normal(mu, sigma)"


def test_extra_namespace():
    """Tests the formula can access an additional namespace"""
    data = bmb.load_data("carclaims")
    extra_namespace = {"levels": data["veh_body"].unique()}
    formula = "numclaims ~ 0 + C(veh_body, levels=levels)"
    model = bmb.Model(formula, data, family="poisson", link="log", extra_namespace=extra_namespace)
    term = model.components[model.family.likelihood.parent].terms["C(veh_body, levels=levels)"]
    assert (np.asarray(term.levels) == data["veh_body"].unique()).all()


def test_drop_na(crossed_data, caplog):
    crossed_data_missing = crossed_data.copy()
    crossed_data_missing.loc[0, "Y"] = np.nan
    crossed_data_missing.loc[1, "continuous"] = np.nan
    crossed_data_missing.loc[2, "threecats"] = np.nan

    with caplog.at_level(logging.INFO):
        bmb.Model("Y ~ continuous + threecats", crossed_data_missing, dropna=True)
        assert "Automatically removing 3/120 rows from the dataset." in caplog.text

    with pytest.raises(ValueError, match="'data' contains 3 incomplete rows"):
        bmb.Model("Y ~ continuous + threecats", crossed_data_missing)


def test_plot_priors(crossed_data):
    model = bmb.Model("Y ~ 0 + threecats", crossed_data)
    with pytest.raises(ValueError, match="Model is not built yet"):
        model.plot_priors()
    model.build()
    model.plot_priors()


def test_model_graph(crossed_data):
    model = bmb.Model("Y ~ 0 + threecats", crossed_data)
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


@pytest.mark.skip(reason="this example no longer trigger the fallback to adapt_diag")
def test_init_fallback(init_data, caplog):
    model = bmb.Model("od ~ temp + (1|source) + 0", init_data)
    with caplog.at_level(logging.INFO):
        model.fit(draws=100, init="auto")
        assert "Initializing NUTS using jitter+adapt_diag..." in caplog.text
        assert "The default initialization" in caplog.text
        assert "Initializing NUTS using adapt_diag..." in caplog.text


def test_2d_response_no_shape():
    """
    This tests whether a model where there's a single linear predictor and a response with
    response.ndim > 1 works well, without Bambi causing any shape problems.
    See https://github.com/bambinos/bambi/pull/629
    Updated https://github.com/bambinos/bambi/pull/632
    """

    def fn(name, p, observed, **kwargs):
        y = observed[:, 0].flatten()
        n = observed[:, 1].flatten()
        # It's the users' responsibility to take only the first dim
        kwargs["dims"] = kwargs.get("dims")[0]
        return pm.Binomial(name, p=p, n=n, observed=y, **kwargs)

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
    model.fit(draws=10, tune=10)
