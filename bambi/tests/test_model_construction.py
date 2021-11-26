from functools import reduce
from operator import add
from os.path import dirname, join

import arviz as az
import numpy as np
import pandas as pd
import pytest

from formulae import design_matrices

from bambi.models import Model
from bambi.terms import Term, GroupSpecificTerm
from bambi.priors import Prior


@pytest.fixture(scope="module")
def data_numeric_xy():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
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
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "crossed_random.csv"))
    return data


def test_term_init(diabetes_data):
    design = design_matrices("BMI", diabetes_data)
    term_info = design.common.terms_info["BMI"]
    term = Term("BMI", term_info, diabetes_data["BMI"])
    # Test that all defaults are properly initialized
    assert term.name == "BMI"
    assert not term.categorical
    assert not term.group_specific
    assert term.levels is not None
    assert term.data.shape == (442,)


def test_distribute_group_specific_effect_over(diabetes_data):
    # 163 unique levels of BMI in diabetes_data
    # With intercept
    model = Model("BP ~ (C(age_grp)|BMI)", diabetes_data)
    model.build()

    # Treatment encoding because of the intercept
    lvls = sorted(list(diabetes_data["age_grp"].unique()))[1:]

    assert "C(age_grp)|BMI" in model.terms
    assert "1|BMI" in model.terms
    assert model.terms["C(age_grp)|BMI"].pymc_coords["C(age_grp)_coord_group_expr"] == lvls

    # This is equal to the sub-matrix of Z that corresponds to this term.
    # 442 is the number of observations. 163 the number of groups.
    # 2 is the number of levels of the categorical variable 'C(age_grp)' after removing
    # the reference level. Then the number of columns is 326 = 163 * 2.
    assert model.terms["C(age_grp)|BMI"].data.shape == (442, 326)

    # Without intercept. Reference level is not removed.
    model = Model("BP ~ (0 + C(age_grp)|BMI)", diabetes_data)
    model.build()

    assert "C(age_grp)|BMI" in model.terms
    assert not "1|BMI" in model.terms
    assert model.terms["C(age_grp)|BMI"].data.shape == (442, 489)


def test_model_init_from_filename():
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    filename = join(data_dir, "diabetes.txt")
    model = Model("BP ~ BMI", filename)
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.shape == (442, 11)
    assert "BMI" in model.data.columns


def test_model_init_bad_data():
    with pytest.raises(ValueError):
        Model("y ~ x", {"x": 1})


def test_model_categorical_argument():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x": np.random.randint(2, size=100),
            "z": np.random.randint(2, size=100),
        }
    )
    model = Model("y ~ 0 + x", data, categorical="x")
    assert model.terms["x"].categorical

    model = Model("y ~ 0 + x*z", data, categorical=["x", "z"])
    assert model.terms["x"].categorical
    assert model.terms["z"].categorical
    assert model.terms["x:z"].categorical


def test_model_no_response():
    with pytest.raises(ValueError):
        Model("x", pd.DataFrame({"x": [1]}))


def test_model_taylor_value(data_numeric_xy):
    Model("y ~ x", data=data_numeric_xy, taylor=5)


def test_model_alternative_scaler(data_numeric_xy):
    Model("y ~ x", data=data_numeric_xy, automatic_priors="mle")


def test_model_term_names_property(diabetes_data):
    model = Model("BMI ~ age_grp + BP + S1", diabetes_data)
    assert model.term_names == ["Intercept", "age_grp", "BP", "S1"]


def test_model_term_names_property_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model("Y ~ threecats*fourcats", crossed_data)
    assert model.term_names == ["Intercept", "threecats", "fourcats", "threecats:fourcats"]


def test_model_terms_levels_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model("Y ~ threecats*fourcats", crossed_data)

    assert model.terms["threecats:fourcats"].levels == [
        "threecats[b]:fourcats[b]",
        "threecats[b]:fourcats[c]",
        "threecats[b]:fourcats[d]",
        "threecats[c]:fourcats[b]",
        "threecats[c]:fourcats[c]",
        "threecats[c]:fourcats[d]",
    ]


def test_model_terms_levels():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=50),
            "x": np.random.normal(size=50),
            "z": reduce(add, [[f"Group {x}"] * 10 for x in ["1", "2", "3", "1", "2"]]),
            "time": list(range(1, 11)) * 5,
            "subject": reduce(add, [[f"Subject {x}"] * 10 for x in range(1, 6)]),
        }
    )
    model = Model("y ~ x + z + time + (time|subject)", data)
    assert model.terms["z"].levels == ["z[Group 2]", "z[Group 3]"]
    assert model.terms["1|subject"].groups == [f"Subject {x}" for x in range(1, 6)]
    assert model.terms["time|subject"].groups == [f"Subject {x}" for x in range(1, 6)]


def test_model_term_classes():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=50),
            "x": np.random.normal(size=50),
            "s": ["s1"] * 25 + ["s2"] * 25,
            "g": np.random.choice(["a", "b", "c"], size=50),
        }
    )

    model = Model("y ~ x*g + (x|s)", data)

    assert isinstance(model.terms["x"], Term)
    assert isinstance(model.terms["g"], Term)
    assert isinstance(model.terms["x:g"], Term)
    assert isinstance(model.terms["1|s"], GroupSpecificTerm)
    assert isinstance(model.terms["x|s"], GroupSpecificTerm)

    # Also check 'categorical' attribute is right
    assert model.terms["g"].categorical


def test_one_shot_formula_fit(diabetes_data):
    model = Model("S3 ~ S1 + S2", diabetes_data)
    model.fit(draws=50)
    named_vars = model.backend.model.named_vars
    targets = ["S3", "S1", "Intercept"]
    assert len(set(named_vars.keys()) & set(targets)) == 3


def test_categorical_term():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=6),
            "x1": np.random.normal(size=6),
            "x2": [1, 1, 0, 0, 1, 1],
            "g1": ["a"] * 3 + ["b"] * 3,
            "g2": ["x", "x", "z", "z", "y", "y"],
        }
    )
    model = Model("y ~ x1 + x2 + g1 + (g1|g2) + (x2|g2)", data)
    fitted = model.fit(draws=10)
    df = az.summary(fitted)
    names = [
        "Intercept",
        "x1",
        "x2",
        "g1[b]",
        "1|g2_sigma",
        "1|g2[x]",
        "1|g2[y]",
        "1|g2[z]",
        "g1|g2_sigma[b]",
        "g1|g2[b, x]",
        "g1|g2[b, y]",
        "g1|g2[b, z]",
        "x2|g2_sigma",
        "x2|g2[x]",
        "x2|g2[y]",
        "x2|g2[z]",
        "y_sigma",
    ]
    assert list(df.index) == names


def test_omit_offsets_false():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model("y ~ x1 + (x1|g1)", data)
    fitted = model.fit(omit_offsets=False)
    offsets = [var for var in fitted.posterior.var() if var.endswith("_offset")]
    assert offsets == ["1|g1_offset", "x1|g1_offset"]


def test_omit_offsets_true():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model("y ~ x1 + (x1|g1)", data)
    fitted = model.fit(omit_offsets=True)
    offsets = [var for var in fitted.posterior.var() if var.endswith("_offset")]
    assert not offsets


def test_hyperprior_on_common_effect():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    slope = Prior("Normal", mu=0, sd=Prior("HalfCauchy", beta=2))

    priors = {"x1": slope}
    with pytest.raises(ValueError):
        Model("y ~ x1 + (x1|g1)", data, priors=priors)

    priors = {"common": slope}
    with pytest.raises(ValueError):
        Model("y ~ x1 + (x1|g1)", data, priors=priors)


def test_empty_formula_assertion():
    data = pd.DataFrame({"y": [1]})
    # ValueError when attempt to fit a model without having passed a formula
    with pytest.raises(ValueError):
        Model(data=data)


def test_sparse_fails():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=4),
            "x1": np.random.normal(size=4),
            "x2": np.random.normal(size=4),
            "x3": np.random.normal(size=4),
            "x4": np.random.normal(size=4),
        }
    )
    with pytest.raises(ValueError, match="Design matrix for common effects is not full-rank"):
        Model("y ~ x1 + x2 + x3 + x4", data, automatic_priors="mle")

    data = pd.DataFrame(
        {
            "y": np.random.normal(size=4),
            "g1": ["a", "b", "c", "d"],
            "g2": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(ValueError, match="Design matrix for common effects is not full-rank"):
        Model("y ~ g1 + g2", data, automatic_priors="mle")


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "negativebinomial",
        "bernoulli",
        "poisson",
        "gamma",
        "wald",
    ],
)
def test_automatic_priors(family):
    """Test that automatic priors work correctly"""
    obs = pd.DataFrame([0], columns=["x"])
    Model("x ~ 0", obs, family=family)


def test_links():
    data = pd.DataFrame(
        {
            "g": np.random.choice([0, 1], size=100),
            "y": np.random.randint(3, 10, size=100),
            "x": np.random.randint(3, 10, size=100),
        }
    )

    FAMILIES = {
        "bernoulli": ["identity", "logit", "probit", "cloglog"],
        "beta": ["identity", "logit", "probit", "cloglog"],
        "gamma": ["identity", "inverse", "log"],
        "gaussian": ["identity", "log", "inverse"],
        "negativebinomial": ["identity", "log", "cloglog"],
        "poisson": ["identity", "log"],
        "wald": ["inverse", "inverse_squared", "identity", "log"],
    }
    for family, links in FAMILIES.items():
        for link in links:
            if family == "bernoulli":
                Model("g ~ x", data, family=family, link=link)
            else:
                Model("y ~ x", data, family=family, link=link)


def test_bad_links():
    """Passes names of links that are not suitable for the family."""
    data = pd.DataFrame(
        {
            "g": np.random.choice([0, 1], size=100),
            "y": np.random.randint(3, 10, size=100),
            "x": np.random.randint(3, 10, size=100),
        }
    )
    FAMILIES = {
        "bernoulli": ["inverse", "inverse_squared", "log"],
        "beta": ["inverse", "inverse_squared", "log"],
        "gamma": ["logit", "probit", "cloglog"],
        "gaussian": ["logit", "probit", "cloglog"],
        "negativebinomial": ["logit", "probit", "inverse", "inverse_squared"],
        "poisson": ["logit", "probit", "cloglog", "inverse", "inverse_squared"],
        "wald": ["logit", "probit", "cloglog"],
    }

    for family, links in FAMILIES.items():
        for link in links:
            with pytest.raises(ValueError):
                if family == "bernoulli":
                    formula = "g ~ x"
                else:
                    formula = "y ~ x"
                Model(formula, data, family=family, link=link)


def test_constant_terms():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=10),
            "x": np.random.choice([1], size=10),
            "z": np.random.choice(["A"], size=10),
        }
    )

    with pytest.raises(ValueError):
        Model("y ~ 0 + x", data)

    with pytest.raises(ValueError):
        Model("y ~ 0 + z", data)
