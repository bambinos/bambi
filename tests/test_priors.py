from os.path import dirname, join
import pytest

import numpy as np
import pymc as pm
import pandas as pd
from scipy import special

import bambi as bmb


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "diabetes.txt"), sep="\t")
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


def test_prior_class():
    prior = bmb.Prior("CheeseWhiz", holes=0, taste=-10)
    assert prior.name == "CheeseWhiz"
    assert isinstance(prior.args, dict)
    assert prior.args["taste"] == -10
    prior.update(taste=-100, return_to_store=1)
    assert prior.args["return_to_store"] == 1


def test_likelihood_class():
    # bmb.Likelihood with recognized name
    likelihood = bmb.Likelihood("Normal", ["mu", "sigma"], "mu")
    for name in ["name", "params", "parent", "dist"]:
        assert hasattr(likelihood, name)

    # A likelihood with unrecognized name
    # The class is not going to complain. Whether "Magic" works in PyMC is up to the user.
    likelihood = bmb.Likelihood("Magic", ["Wizard", "Witcher"], "Wizard")
    for name in ["name", "params", "parent", "dist"]:
        assert hasattr(likelihood, name)


def test_likelihood_bad_parent():
    with pytest.raises(
        ValueError, match="'Mu' is not a valid parameter for the likelihood 'Normal'"
    ):
        bmb.Likelihood("Normal", params=["mu", "sigma"], parent="Mu")

    with pytest.raises(
        ValueError, match="'Mu' is not a valid parameter for the likelihood 'Normal'"
    ):
        bmb.Likelihood("Normal", parent="Mu")

    with pytest.raises(
        ValueError, match="'mu' is not a valid parameter for the likelihood 'Bernoulli'"
    ):
        bmb.Likelihood("Bernoulli", params=["p"], parent="mu")

    with pytest.raises(
        ValueError, match="'mu' is not a valid parameter for the likelihood 'Bernoulli'"
    ):
        bmb.Likelihood("Bernoulli", parent="mu")


def test_likelihood_parent_inferred():
    lh1 = bmb.Likelihood("Normal", parent="mu")
    lh2 = bmb.Likelihood("Normal")
    assert lh1.parent == lh2.parent


def test_family_class():
    likelihood = bmb.Likelihood("Cheese", params=["holes", "milk"], parent="holes")
    family = bmb.Family("cheese", likelihood=likelihood, link="logit")

    for name in ["name", "likelihood", "link"]:
        assert hasattr(family, name)


def test_family_bad_priors():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.normal(size=100),
            "g": rng.choice(["A", "B"], size=100),
        }
    )
    likelihood = bmb.Likelihood("Normal", params=["mu", "sigma"], parent="mu")
    family = bmb.Family("MyNormal", likelihood, "identity")
    # Required prior is missing
    with pytest.raises(ValueError, match="The component 'sigma' needs a prior."):
        bmb.Model("y ~ x", data, family=family)

    # bmb.Prior is not a prior
    with pytest.raises(ValueError, match="'Whatever' is not a valid prior."):
        bmb.Model("y ~ x", data, family=family, priors={"sigma": "Whatever"})


def test_auto_scale(diabetes_data):
    # By default, should scale everything except custom bmb.Prior() objects
    priors = {"BP": bmb.Prior("Cauchy", alpha=1, beta=17.5)}
    model = bmb.Model("BMI ~ S1 + S2 + BP", diabetes_data, priors=priors)
    p1 = model.response_component.terms["S1"].prior
    p2 = model.response_component.terms["S2"].prior
    p3 = model.response_component.terms["BP"].prior
    assert p1.name == p2.name == "Normal"
    assert 0 < p1.args["sigma"] < 1
    assert p2.args["sigma"] > p1.args["sigma"]
    assert p3.name == "Cauchy"
    assert p3.args["beta"] == 17.5

    # With auto_scale off, custom priors are considered.
    priors = {"BP": bmb.Prior("Cauchy", alpha=1, beta=17.5)}
    model = bmb.Model("BMI ~ S1 + S2 + BP", diabetes_data, priors=priors, auto_scale=False)
    p2_off = model.response_component.terms["S2"].prior
    p3_off = model.response_component.terms["BP"].prior
    assert p2_off.name == "Flat"
    assert "sigma" not in p2_off.args
    assert p3_off.name == "Cauchy"


def test_prior_str():
    # Tests __str__ method
    prior1 = bmb.Prior("Normal", mu=0, sigma=1)
    prior2 = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
    assert str(prior1) == "Normal(mu: 0.0, sigma: 1.0)"
    assert str(prior2) == "Normal(mu: 0.0, sigma: HalfNormal(sigma: 1.0))"
    assert str(prior1) == repr(prior1)


def test_prior_eq():
    # Tests __eq__ method
    prior1 = bmb.Prior("Normal", mu=0, sigma=1)
    prior2 = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
    assert prior1 == prior1
    assert prior2 == prior2
    assert prior1 != prior2
    assert prior1 != "bmb.Prior"


def test_family_link_unsupported():
    prior = bmb.Prior("CheeseWhiz", holes=0, taste=-10)
    likelihood = bmb.Likelihood("Cheese", parent="holes", params=["holes", "milk"])
    family = bmb.Family("cheese", likelihood=likelihood, link="cloglog")
    family.set_default_priors({"milk": prior})
    with pytest.raises(
        ValueError, match="Link 'Empty' cannot be used for 'holes' with family 'cheese'"
    ):
        family.link = "Empty"


def test_custom_link():
    rng = np.random.default_rng(121195)
    likelihood = bmb.Likelihood("Bernoulli", parent="p")
    link = bmb.Link(
        "my_logit", link=special.expit, linkinv=special.logit, linkinv_backend=pm.math.sigmoid
    )
    family = bmb.Family("bernoulli", likelihood, link)

    data = pd.DataFrame(
        {
            "y": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "x1": rng.uniform(size=10),
            "x2": rng.uniform(size=10),
        }
    )
    model = bmb.Model("y ~ x1 + x2", data, family=family)
    model.build()


def test_family_bad_type():
    data = pd.DataFrame({"x": [1], "y": [1]})

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family=0)

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family=set("gaussian"))

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family={"family": "gaussian"})


def test_set_priors():
    # NOTE I'm not sure if this test is OK. 'prior' and 'gp_prior' still point to the same
    #      object and that's why the `.auto_scale` attribute is updated in both..
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.normal(size=100),
            "g": rng.choice(["A", "B"], size=100),
        }
    )
    model = bmb.Model("y ~ x + (1|g)", data)
    prior = bmb.Prior("Uniform", lower=0, upper=50)
    gp_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Normal", mu=0, sigma=1))

    # Common
    model.set_priors(common=prior)
    assert model.response_component.terms["x"].prior == prior

    # Group-specific
    with pytest.raises(ValueError, match="must have hyperpriors"):
        model.set_priors(group_specific=prior)

    model.set_priors(group_specific=gp_prior)
    assert model.response_component.terms["1|g"].prior == gp_prior

    # By name
    model = bmb.Model("y ~ x + (1|g)", data)
    model.set_priors(priors={"Intercept": prior})
    model.set_priors(priors={"x": prior})
    model.set_priors(priors={"1|g": gp_prior})
    assert model.response_component.terms["Intercept"].prior == prior
    assert model.response_component.terms["x"].prior == prior
    assert model.response_component.terms["1|g"].prior == gp_prior


def test_set_prior_unexisting_term():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=100),
            "x": rng.normal(size=100),
        }
    )
    prior = bmb.Prior("Uniform", lower=0, upper=50)
    model = bmb.Model("y ~ x", data)
    with pytest.raises(KeyError):
        model.set_priors(priors={"z": prior})


def test_response_prior():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame({"y": rng.integers(3, 10, size=50), "x": rng.normal(size=50)})

    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("y ~ x", data, priors=priors)
    priors["sigma"].auto_scale = False  # the one in the model is set to False
    assert model.constant_components["sigma"].prior == priors["sigma"]

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model("y ~ x", data, family="negativebinomial", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("y ~ x", data, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("y ~ x", data, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]


def test_set_response_prior():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame({"y": rng.integers(3, 10, size=50), "x": rng.normal(size=50)})

    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("y ~ x", data)
    model.set_priors(priors)
    assert model.constant_components["sigma"].prior == bmb.Prior("Uniform", False, lower=0, upper=50)

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model("y ~ x", data, family="negativebinomial")
    model.set_priors(priors)
    assert model.constant_components["alpha"].prior == bmb.Prior("Uniform", False, lower=1, upper=20)

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("y ~ x", data, family="gamma")
    model.set_priors(priors)
    assert model.constant_components["alpha"].prior == bmb.Prior("Uniform", False, lower=0, upper=50)


def test_prior_shape():
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "score": rng.normal(size=100),
            "q": rng.choice(["1", "2", "3", "4", "5"], size=100),
            "s": rng.choice(["a", "b", "c"], size=100),
            "g": rng.choice(["A", "B", "C"], size=100),
        }
    )

    model = bmb.Model("score ~ 0 + q", data)
    assert model.response_component.terms["q"].prior.args["mu"].shape == (5,)
    assert model.response_component.terms["q"].prior.args["sigma"].shape == (5,)

    model = bmb.Model("score ~ q", data)
    assert model.response_component.terms["q"].prior.args["mu"].shape == (4,)
    assert model.response_component.terms["q"].prior.args["sigma"].shape == (4,)

    model = bmb.Model("score ~ 0 + q:s", data)
    assert model.response_component.terms["q:s"].prior.args["mu"].shape == (15,)
    assert model.response_component.terms["q:s"].prior.args["sigma"].shape == (15,)

    # "s" is automatically added to ensure full rank matrix
    model = bmb.Model("score ~ q:s", data)
    assert model.response_component.terms["Intercept"].prior.args["mu"].shape == ()
    assert model.response_component.terms["Intercept"].prior.args["sigma"].shape == ()

    assert model.response_component.terms["s"].prior.args["mu"].shape == (2,)
    assert model.response_component.terms["s"].prior.args["sigma"].shape == (2,)

    assert model.response_component.terms["q:s"].prior.args["mu"].shape == (12,)
    assert model.response_component.terms["q:s"].prior.args["sigma"].shape == (12,)


def test_set_priors_but_intercept():
    rng = np.random.default_rng(121195)
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "z": rng.normal(size=10),
            "x1": rng.uniform(size=10),
            "x2": rng.uniform(size=10),
            "g": ["A"] * 5 + ["B"] * 5,
        }
    )

    priors = {
        "x1": bmb.Prior("TruncatedNormal", sigma=1, mu=0, lower=0),
        "x2": bmb.Prior("TruncatedNormal", sigma=1, mu=0, upper=0),
    }

    bmb.Model("y ~ x1 + x2", df, family="bernoulli", priors=priors)

    priors = {
        "x1": bmb.Prior("StudentT", mu=0, nu=4, lam=1),
        "x2": bmb.Prior("StudentT", mu=0, nu=8, lam=2),
    }

    bmb.Model("z ~ x1 + x2 + (1|g)", df, priors=priors)


def test_custom_prior():
    def CustomPrior(name, *args, dims=None, **kwargs):
        return pm.Normal(name, *args, dims=dims, **kwargs)

    data = bmb.load_data("my_data")

    priors = {"x": bmb.Prior("CustomPrior", mu=0, sigma=5, dist=CustomPrior)}
    model = bmb.Model("y ~ x", data, priors=priors)
    model.build()
    assert model.backend.model.free_RVs[-1].str_repr() == "x ~ N(0, 5)"
