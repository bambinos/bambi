from os.path import dirname, join
import pytest

import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import special

from bambi.families import Family, Likelihood, Link
from bambi.models import Model
from bambi.priors import Prior


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "diabetes.txt"), sep="\t")
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


def test_prior_class():
    prior = Prior("CheeseWhiz", holes=0, taste=-10)
    assert prior.name == "CheeseWhiz"
    assert isinstance(prior.args, dict)
    assert prior.args["taste"] == -10
    prior.update(taste=-100, return_to_store=1)
    assert prior.args["return_to_store"] == 1


def test_likelihood_class():
    # A recognized likelihood
    sigma = Prior("HalfNormal", sigma=100)
    likelihood = Likelihood("Normal", parent="mu", sigma=sigma)

    for name in ["name", "priors", "parent"]:
        assert hasattr(likelihood, name)

    # A likelihood with unrecognized name
    # The class is not going to complain. Whether "Magic" works in PyMC3 is up to the user.
    likelihood = Likelihood("Magic", parent="Wizard", sigma=sigma)
    for name in ["name", "priors", "parent"]:
        assert hasattr(likelihood, name)


def test_likelihood_bad_parent():
    with pytest.raises(ValueError):
        Likelihood("Normal", parent="Mu", sigma=Prior("HalfNormal", sigma=100))

    with pytest.raises(ValueError):
        Likelihood("Bernoulli", parent="mu")


def test_likelihood_parent_inferred():
    sigma = Prior("HalfNormal", sigma=100)
    lh1 = Likelihood("Normal", parent="mu", sigma=sigma)
    lh2 = Likelihood("Normal", sigma=sigma)
    assert lh1.parent == lh2.parent


def test_likelihood_bad_priors():
    sigma = Prior("HalfNormal", sigma=100)
    # Required prior is missing
    with pytest.raises(ValueError):
        Likelihood("Normal", parent="mu")

    # Prior is not a prior
    with pytest.raises(ValueError):
        Likelihood("Normal", parent="mu", sigma="HalfNormal")

    # Passing unnecessary priors
    with pytest.raises(ValueError):
        Likelihood("Bernoulli", sigma=sigma)

    # Passed priors, but not the one needed
    with pytest.raises(ValueError):
        Likelihood("Gamma", sigma=sigma)


def test_family_class():
    cheese = Prior("CheeseWhiz", holes=0, taste=-10)
    likelihood = Likelihood("Cheese", parent="holes", cheese=cheese)
    family = Family("cheese", likelihood=likelihood, link="logit")

    for name in ["name", "likelihood", "link"]:
        assert hasattr(family, name)


def test_auto_scale(diabetes_data):

    # By default, should scale everything except custom Prior() objects
    priors = {"S1": 0.3, "BP": Prior("Cauchy", alpha=1, beta=17.5)}
    model = Model("BMI ~ S1 + S2 + BP", diabetes_data, priors=priors)
    p1 = model.terms["S1"].prior
    p2 = model.terms["S2"].prior
    p3 = model.terms["BP"].prior
    assert p1.name == p2.name == "Normal"
    assert 0 < p1.args["sigma"] < 1
    assert p2.args["sigma"] > p1.args["sigma"]
    assert p3.name == "Cauchy"
    assert p3.args["beta"] == 17.5

    # With auto_scale off, custom priors are considered, but not custom scaling.
    # Prior has no effect, and prior for BP has effect.
    priors = {"S1": 0.3, "BP": Prior("Cauchy", alpha=1, beta=17.5)}
    model = Model("BMI ~ S1 + S2 + BP", diabetes_data, priors=priors, auto_scale=False)
    p1_off = model.terms["S1"].prior
    p2_off = model.terms["S2"].prior
    p3_off = model.terms["BP"].prior
    assert p1_off.name == "Normal"
    assert p2_off.name == "Flat"
    assert p1_off.args["sigma"] == 1
    assert "sigma" not in p2_off.args
    assert p3_off.name == "Cauchy"


def test_prior_str():
    # Tests __str__ method
    prior1 = Prior("Normal", mu=0, sigma=1)
    prior2 = Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=1))
    assert str(prior1) == "Normal(mu: 0, sigma: 1)"
    assert str(prior2) == "Normal(mu: 0, sigma: HalfNormal(sigma: 1))"
    assert str(prior1) == repr(prior1)


def test_prior_eq():
    # Tests __eq__ method
    prior1 = Prior("Normal", mu=0, sigma=1)
    prior2 = Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=1))
    assert prior1 == prior1
    assert prior2 == prior2
    assert prior1 != prior2
    assert prior1 != "Prior"


def test_family_link_unsupported():
    cheese = Prior("CheeseWhiz", holes=0, taste=-10)
    likelihood = Likelihood("Cheese", parent="holes", cheese=cheese)
    family = Family("cheese", likelihood=likelihood, link="cloglog")
    with pytest.raises(ValueError):
        family.link = "Empty"


def test_custom_link():
    likelihood = Likelihood("Bernoulli", parent="p")
    link = Link(
        "my_logit", link=special.expit, linkinv=special.logit, linkinv_backend=pm.math.sigmoid
    )
    family = Family("bernoulli", likelihood, link)

    data = pd.DataFrame(
        {
            "y": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "x1": np.random.uniform(size=10),
            "x2": np.random.uniform(size=10),
        }
    )
    model = Model("y ~ x1 + x2", data, family=family)


def test_family_bad_type():
    data = pd.DataFrame({"x": [1], "y": [1]})

    with pytest.raises(ValueError):
        Model("y ~ x", data, family=0)

    with pytest.raises(ValueError):
        Model("y ~ x", data, family=set("gaussian"))

    with pytest.raises(ValueError):
        Model("y ~ x", data, family={"family": "gaussian"})


def test_set_priors():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
            "g": np.random.choice(["A", "B"], size=100),
        }
    )
    model = Model("y ~ x + (1|g)", data)
    prior = Prior("Uniform", lower=0, upper=50)

    # Common
    model.set_priors(common=prior)
    assert model.terms["x"].prior == prior

    # Group-specific
    model.set_priors(group_specific=prior)
    assert model.terms["1|g"].prior == prior

    # By name
    model = Model("y ~ x + (1|g)", data)
    model.set_priors(priors={"Intercept": prior})
    model.set_priors(priors={"x": prior})
    model.set_priors(priors={"1|g": prior})
    assert model.terms["Intercept"].prior == prior
    assert model.terms["x"].prior == prior
    assert model.terms["1|g"].prior == prior


def test_set_prior_with_tuple():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
            "z": np.random.normal(size=100),
        }
    )
    prior = Prior("Uniform", lower=0, upper=50)
    model = Model("y ~ x + z", data)
    model.set_priors(priors={("x", "z"): prior})

    # Prior is set to auto_scale=False when it set in the model
    prior.auto_scale = False
    assert model.terms["x"].prior == prior
    assert model.terms["z"].prior == prior


def test_set_prior_unexisting_term():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x": np.random.normal(size=100),
        }
    )
    prior = Prior("Uniform", lower=0, upper=50)
    model = Model("y ~ x", data)
    with pytest.raises(ValueError):
        model.set_priors(priors={("x", "z"): prior})


def test_response_prior():
    data = pd.DataFrame({"y": np.random.randint(3, 10, size=50), "x": np.random.normal(size=50)})

    priors = {"sigma": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, priors=priors)
    priors["sigma"].auto_scale = False  # the one in the model is set to False
    assert model.family.likelihood.priors["sigma"] == priors["sigma"]

    priors = {"alpha": Prior("Uniform", lower=1, upper=20)}
    model = Model("y ~ x", data, family="negativebinomial", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.family.likelihood.priors["alpha"] == priors["alpha"]

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.family.likelihood.priors["alpha"] == priors["alpha"]

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.family.likelihood.priors["alpha"] == priors["alpha"]


def test_set_response_prior():
    data = pd.DataFrame({"y": np.random.randint(3, 10, size=50), "x": np.random.normal(size=50)})

    priors = {"sigma": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data)
    model.set_priors(priors)
    assert model.family.likelihood.priors["sigma"] == Prior("Uniform", False, lower=0, upper=50)

    priors = {"alpha": Prior("Uniform", lower=1, upper=20)}
    model = Model("y ~ x", data, family="negativebinomial")
    model.set_priors(priors)
    assert model.family.likelihood.priors["alpha"] == Prior("Uniform", False, lower=1, upper=20)

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma")
    model.set_priors(priors)
    assert model.family.likelihood.priors["alpha"] == Prior("Uniform", False, lower=0, upper=50)


def test_response_prior_fail():
    data = pd.DataFrame(
        {"y": np.random.randint(3, 10, size=50), "sigma": np.random.normal(size=50)}
    )

    priors = {"sigma": Prior("Uniform", lower=0, upper=50)}
    with pytest.raises(ValueError):
        Model("y ~ sigma", data, priors=priors)

    data.rename(columns={"sigma": "alpha"}, inplace=True)
    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    with pytest.raises(ValueError):
        Model("y ~ alpha", data, family="negativebinomial", priors=priors)

    with pytest.raises(ValueError):
        Model("y ~ alpha", data, family="gamma", priors=priors)


def test_prior_shape():
    data = pd.DataFrame(
        {
            "score": np.random.normal(size=100),
            "q": np.random.choice(["1", "2", "3", "4", "5"], size=100),
            "s": np.random.choice(["a", "b", "c"], size=100),
            "g": np.random.choice(["A", "B", "C"], size=100),
        }
    )

    model = Model("score ~ 0 + q", data)
    assert model.terms["q"].prior.args["mu"].shape == (5,)
    assert model.terms["q"].prior.args["sigma"].shape == (5,)

    model = Model("score ~ q", data)
    assert model.terms["q"].prior.args["mu"].shape == (4,)
    assert model.terms["q"].prior.args["sigma"].shape == (4,)

    model = Model("score ~ 0 + q:s", data)
    assert model.terms["q:s"].prior.args["mu"].shape == (15,)
    assert model.terms["q:s"].prior.args["sigma"].shape == (15,)

    # "s" is automatically added to ensure full rank matrix
    model = Model("score ~ q:s", data)
    assert model.terms["Intercept"].prior.args["mu"].shape == ()
    assert model.terms["Intercept"].prior.args["sigma"].shape == ()

    assert model.terms["s"].prior.args["mu"].shape == (2,)
    assert model.terms["s"].prior.args["sigma"].shape == (2,)

    assert model.terms["q:s"].prior.args["mu"].shape == (12,)
    assert model.terms["q:s"].prior.args["sigma"].shape == (12,)


def test_set_priors_but_intercept():
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "z": np.random.normal(size=10),
            "x1": np.random.uniform(size=10),
            "x2": np.random.uniform(size=10),
            "g": ["A"] * 5 + ["B"] * 5,
        }
    )

    priors = {
        "x1": Prior("TruncatedNormal", sigma=1, mu=0, lower=0),
        "x2": Prior("TruncatedNormal", sigma=1, mu=0, upper=0),
    }

    Model("y ~ x1 + x2", df, family="bernoulli", priors=priors)

    priors = {
        "x1": Prior("StudentT", mu=0, nu=4, lam=1),
        "x2": Prior("StudentT", mu=0, nu=8, lam=2),
    }

    Model("z ~ x1 + x2 + (1|g)", df, priors=priors)
