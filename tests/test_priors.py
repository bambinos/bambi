import pytest

import bambi as bmb
import numpy as np
import pymc as pm
import pandas as pd
from scipy import special


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


def test_family_bad_priors(data_random_n100):
    likelihood = bmb.Likelihood("Normal", params=["mu", "sigma"], parent="mu")
    family = bmb.Family("MyNormal", likelihood, "identity")
    # Required prior is missing
    with pytest.raises(ValueError, match="The component 'sigma' needs a prior."):
        bmb.Model("continuous1 ~ continuous2", data_random_n100, family=family)

    # bmb.Prior is not a prior
    with pytest.raises(ValueError, match="'Whatever' is not a valid prior."):
        bmb.Model(
            "continuous1 ~ continuous2",
            data_random_n100,
            family=family,
            priors={"sigma": "Whatever"},
        )


def test_auto_scale(data_diabetes):
    # By default, should scale everything except custom bmb.Prior() objects
    priors = {"BP": bmb.Prior("Cauchy", alpha=1, beta=17.5)}
    model = bmb.Model("BMI ~ S1 + S2 + BP", data_diabetes, priors=priors)
    parent_component = model.components[model.family.likelihood.parent]
    p1 = parent_component.terms["S1"].prior
    p2 = parent_component.terms["S2"].prior
    p3 = parent_component.terms["BP"].prior
    assert p1.name == p2.name == "Normal"
    assert 0 < p1.args["sigma"] < 1
    assert p2.args["sigma"] > p1.args["sigma"]
    assert p3.name == "Cauchy"
    assert p3.args["beta"] == 17.5

    # With auto_scale off, custom priors are considered.
    priors = {"BP": bmb.Prior("Cauchy", alpha=1, beta=17.5)}
    model = bmb.Model("BMI ~ S1 + S2 + BP", data_diabetes, priors=priors, auto_scale=False)
    parent_component = model.components[model.family.likelihood.parent]
    p2_off = parent_component.terms["S2"].prior
    p3_off = parent_component.terms["BP"].prior
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


def test_custom_link(data_random_n100):
    likelihood = bmb.Likelihood("Bernoulli", parent="p")
    link = bmb.Link(
        "my_logit", link=special.expit, linkinv=special.logit, linkinv_backend=pm.math.sigmoid
    )
    family = bmb.Family("bernoulli", likelihood, link)
    model = bmb.Model("binary_num ~ continuous1 + continuous2", data_random_n100, family=family)
    model.build()


def test_family_bad_type():
    data = pd.DataFrame({"x": [1], "y": [1]})

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family=0)

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family=set("gaussian"))

    with pytest.raises(ValueError):
        bmb.Model("y ~ x", data, family={"family": "gaussian"})


def test_set_priors(data_random_n100):
    # NOTE I'm not sure if this test is OK. 'prior' and 'gp_prior' still point to the same
    #      object and that's why the `.auto_scale` attribute is updated in both..
    model = bmb.Model("continuous1 ~ continuous2 + (1|categorical1)", data_random_n100)
    prior = bmb.Prior("Uniform", lower=0, upper=50)
    gp_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Normal", mu=0, sigma=1))

    # Common
    model.set_priors(common=prior)
    assert model.components[model.family.likelihood.parent].terms["continuous2"].prior == prior

    # Group-specific
    with pytest.raises(ValueError, match="must have hyperpriors"):
        model.set_priors(group_specific=prior)

    model.set_priors(group_specific=gp_prior)
    assert (
        model.components[model.family.likelihood.parent].terms["1|categorical1"].prior == gp_prior
    )

    # By name
    model = bmb.Model("continuous1 ~ continuous2 + (1|categorical1)", data_random_n100)
    model.set_priors(priors={"Intercept": prior})
    model.set_priors(priors={"continuous2": prior})
    model.set_priors(priors={"1|categorical1": gp_prior})
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["Intercept"].prior == prior
    assert parent_component.terms["continuous2"].prior == prior
    assert parent_component.terms["1|categorical1"].prior == gp_prior


def test_set_prior_unexisting_term(data_random_n100):
    prior = bmb.Prior("Uniform", lower=0, upper=50)
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    with pytest.raises(KeyError):
        model.set_priors(priors={"z": prior})


def test_response_prior(data_random_n100):
    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, priors=priors)
    priors["sigma"].auto_scale = False  # the one in the model is set to False
    assert model.constant_components["sigma"].prior == priors["sigma"]

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model(
        "count2 ~ continuous1", data_random_n100, family="negativebinomial", priors=priors
    )
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.constant_components["alpha"].prior == priors["alpha"]


def test_set_response_prior(data_random_n100):
    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100)
    model.set_priors(priors)
    assert model.constant_components["sigma"].prior == bmb.Prior(
        "Uniform", False, lower=0, upper=50
    )

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="negativebinomial")
    model.set_priors(priors)
    assert model.constant_components["alpha"].prior == bmb.Prior(
        "Uniform", False, lower=1, upper=20
    )

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma")
    model.set_priors(priors)
    assert model.constant_components["alpha"].prior == bmb.Prior(
        "Uniform", False, lower=0, upper=50
    )


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
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["q"].prior.args["mu"].shape == (5,)
    assert parent_component.terms["q"].prior.args["sigma"].shape == (5,)

    model = bmb.Model("score ~ q", data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["q"].prior.args["mu"].shape == (4,)
    assert parent_component.terms["q"].prior.args["sigma"].shape == (4,)

    model = bmb.Model("score ~ 0 + q:s", data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["q:s"].prior.args["mu"].shape == (15,)
    assert parent_component.terms["q:s"].prior.args["sigma"].shape == (15,)

    # "s" is automatically added to ensure full rank matrix
    model = bmb.Model("score ~ q:s", data)
    parent_component = model.components[model.family.likelihood.parent]
    assert parent_component.terms["Intercept"].prior.args["mu"].shape == ()
    assert parent_component.terms["Intercept"].prior.args["sigma"].shape == ()

    assert parent_component.terms["s"].prior.args["mu"].shape == (2,)
    assert parent_component.terms["s"].prior.args["sigma"].shape == (2,)

    assert parent_component.terms["q:s"].prior.args["mu"].shape == (12,)
    assert parent_component.terms["q:s"].prior.args["sigma"].shape == (12,)


def test_set_priors_but_intercept(data_random_n100):
    priors = {
        "continuous1": bmb.Prior("TruncatedNormal", sigma=1, mu=0, lower=0),
        "continuous2": bmb.Prior("TruncatedNormal", sigma=1, mu=0, upper=0),
    }
    bmb.Model(
        "binary_num ~ continuous1 + continuous2",
        data_random_n100,
        family="bernoulli",
        priors=priors,
    )

    priors = {
        "continuous2": bmb.Prior("StudentT", mu=0, nu=4, lam=1),
        "continuous3": bmb.Prior("StudentT", mu=0, nu=8, lam=2),
    }
    bmb.Model(
        "continuous1 ~ continuous2 + continuous3 + (1|categorical1)",
        data_random_n100,
        priors=priors,
    )


def test_custom_prior(data_random_n100):
    def CustomPrior(name, *args, dims=None, **kwargs):
        return pm.Normal(name, *args, dims=dims, **kwargs)

    priors = {"continuous2": bmb.Prior("CustomPrior", mu=0, sigma=5, dist=CustomPrior)}
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100, priors=priors)
    model.build()
    assert model.backend.model.free_RVs[-1].str_repr() == "continuous2 ~ Normal(0, 5)"
