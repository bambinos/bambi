import pytest

import bambi as bmb
import numpy as np
import pymc as pm
import pandas as pd
import pytensor.tensor as pt


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
    with pytest.raises(ValueError, match="The parameter 'sigma' needs a prior."):
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
    parent_parameter = model.parameters[model.family.likelihood.parent]
    p1 = parent_parameter.terms["S1"].prior
    p2 = parent_parameter.terms["S2"].prior
    p3 = parent_parameter.terms["BP"].prior
    assert p1.name == p2.name == "Normal"
    assert 0 < p1.args["sigma"] < 1
    assert p2.args["sigma"] > p1.args["sigma"]
    assert p3.name == "Cauchy"
    assert p3.args["beta"] == 17.5

    # With auto_scale off, custom priors are considered.
    priors = {"BP": bmb.Prior("Cauchy", alpha=1, beta=17.5)}
    model = bmb.Model("BMI ~ S1 + S2 + BP", data_diabetes, priors=priors, auto_scale=False)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    p2_off = parent_parameter.terms["S2"].prior
    p3_off = parent_parameter.terms["BP"].prior
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


def test_prior_noncentered_field():
    # Default is None
    prior_default = bmb.Prior("Normal", mu=0, sigma=1)
    assert prior_default.noncentered is None

    # Explicit True / False are stored as-is
    prior_true = bmb.Prior("Normal", mu=0, sigma=1, noncentered=True)
    prior_false = bmb.Prior("Normal", mu=0, sigma=1, noncentered=False)
    assert prior_true.noncentered is True
    assert prior_false.noncentered is False

    # Equality reflects the new field
    assert prior_true == bmb.Prior("Normal", mu=0, sigma=1, noncentered=True)
    assert prior_true != prior_false
    assert prior_default != prior_true
    assert prior_default != prior_false

    # __str__ / __repr__ stay unchanged when noncentered is None,
    # and append the field only when explicitly set
    assert str(prior_default) == "Normal(mu: 0.0, sigma: 1.0)"
    assert str(prior_true) == "Normal(mu: 0.0, sigma: 1.0, noncentered: True)"
    assert str(prior_false) == "Normal(mu: 0.0, sigma: 1.0, noncentered: False)"
    assert str(prior_true) == repr(prior_true)


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
        "my_logit",
        inverse_link=lambda x: pt.zeros_like(x) + 0.25,
    )
    family = bmb.Family("bernoulli", likelihood, link)
    model = bmb.Model("binary_num ~ continuous1 + continuous2", data_random_n100, family=family)
    model.build()
    assert np.allclose(model.backend.model["p"].eval(), 0.25)


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
    assert model.parameters[model.family.likelihood.parent].terms["continuous2"].prior == prior

    # Group-specific
    with pytest.raises(ValueError, match="must have hyperpriors"):
        model.set_priors(group_specific=prior)

    model.set_priors(group_specific=gp_prior)
    assert (
        model.parameters[model.family.likelihood.parent].terms["1|categorical1"].prior == gp_prior
    )

    # By name
    model = bmb.Model("continuous1 ~ continuous2 + (1|categorical1)", data_random_n100)
    model.set_priors(priors={"Intercept": prior})
    model.set_priors(priors={"continuous2": prior})
    model.set_priors(priors={"1|categorical1": gp_prior})
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["Intercept"].prior == prior
    assert parent_parameter.terms["continuous2"].prior == prior
    assert parent_parameter.terms["1|categorical1"].prior == gp_prior


def test_response_prior(data_random_n100):
    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, priors=priors)
    priors["sigma"].auto_scale = False  # the one in the model is set to False
    assert model.marginal_parameters["sigma"].prior == priors["sigma"]

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model(
        "count2 ~ continuous1", data_random_n100, family="negativebinomial", priors=priors
    )
    priors["alpha"].auto_scale = False
    assert model.marginal_parameters["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.marginal_parameters["alpha"].prior == priors["alpha"]

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma", priors=priors)
    priors["alpha"].auto_scale = False
    assert model.marginal_parameters["alpha"].prior == priors["alpha"]


def test_set_response_prior(data_random_n100):
    priors = {"sigma": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100)
    model.set_priors(priors)
    assert model.marginal_parameters["sigma"].prior == bmb.Prior(
        "Uniform", False, lower=0, upper=50
    )

    priors = {"alpha": bmb.Prior("Uniform", lower=1, upper=20)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="negativebinomial")
    model.set_priors(priors)
    assert model.marginal_parameters["alpha"].prior == bmb.Prior(
        "Uniform", False, lower=1, upper=20
    )

    priors = {"alpha": bmb.Prior("Uniform", lower=0, upper=50)}
    model = bmb.Model("count2 ~ continuous1", data_random_n100, family="gamma")
    model.set_priors(priors)
    assert model.marginal_parameters["alpha"].prior == bmb.Prior(
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
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["q"].prior.args["mu"].shape == (5,)
    assert parent_parameter.terms["q"].prior.args["sigma"].shape == (5,)

    model = bmb.Model("score ~ q", data)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["q"].prior.args["mu"].shape == (4,)
    assert parent_parameter.terms["q"].prior.args["sigma"].shape == (4,)

    model = bmb.Model("score ~ 0 + q:s", data)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["q:s"].prior.args["mu"].shape == (15,)
    assert parent_parameter.terms["q:s"].prior.args["sigma"].shape == (15,)

    # "s" is automatically added to ensure full rank matrix
    model = bmb.Model("score ~ q:s", data)
    parent_parameter = model.parameters[model.family.likelihood.parent]
    assert parent_parameter.terms["Intercept"].prior.args["mu"].shape == ()
    assert parent_parameter.terms["Intercept"].prior.args["sigma"].shape == ()

    assert parent_parameter.terms["s"].prior.args["mu"].shape == (2,)
    assert parent_parameter.terms["s"].prior.args["sigma"].shape == (2,)

    assert parent_parameter.terms["q:s"].prior.args["mu"].shape == (12,)
    assert parent_parameter.terms["q:s"].prior.args["sigma"].shape == (12,)


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


def test_unused_prior_in_model_warns(data_random_n100):
    # Issue #815: this used to be silently ignored, so the model was fit with the automatic prior.
    prior = bmb.Prior("Normal", mu=0, sigma=1)
    with pytest.warns(UserWarning, match=r"Unused name\(s\) in `priors`: \['TYPO'\]"):
        bmb.Model("continuous1 ~ continuous2", data_random_n100, priors={"TYPO": prior})


def test_unused_prior_error_mode(data_random_n100, monkeypatch):
    monkeypatch.setattr(bmb.config, "UNUSED_PRIORS", "error")
    prior = bmb.Prior("Normal", mu=0, sigma=1)
    with pytest.raises(ValueError, match=r"Unused name\(s\) in `priors`: \['TYPO'\]") as info:
        bmb.Model("continuous1 ~ continuous2", data_random_n100, priors={"TYPO": prior})

    message = str(info.value)
    assert "continuous2" in message
    assert "Intercept" in message
    assert "UNUSED_PRIORS" in message


def test_unused_prior_ignore_mode(data_random_n100, monkeypatch, recwarn):
    monkeypatch.setattr(bmb.config, "UNUSED_PRIORS", "ignore")
    prior = bmb.Prior("Normal", mu=0, sigma=1)
    bmb.Model("continuous1 ~ continuous2", data_random_n100, priors={"TYPO": prior})
    assert not [w for w in recwarn.list if "Unused name" in str(w.message)]


def test_bare_prior_with_named_parent(data_random_n100, monkeypatch):
    monkeypatch.setattr(bmb.config, "UNUSED_PRIORS", "error")
    intercept_prior = bmb.Prior("Normal", mu=0, sigma=1)
    slope_prior = bmb.Prior("Normal", mu=0, sigma=2)
    bare_slope_prior = bmb.Prior("Normal", mu=0, sigma=3)
    model = bmb.Model(
        "continuous1 ~ continuous2",
        data_random_n100,
        priors={
            "mu": {"continuous2": slope_prior},
            "Intercept": intercept_prior,
            "continuous2": bare_slope_prior,
        },
    )

    intercept_prior.auto_scale = False
    slope_prior.auto_scale = False
    assert model.components["mu"].terms["Intercept"].prior == intercept_prior
    assert model.components["mu"].terms["continuous2"].prior == slope_prior


def test_set_bare_prior_with_named_parent(data_random_n100):
    intercept_prior = bmb.Prior("Normal", mu=0, sigma=1)
    slope_prior = bmb.Prior("Normal", mu=0, sigma=2)
    bare_slope_prior = bmb.Prior("Normal", mu=0, sigma=3)
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    model.set_priors(
        priors={
            "mu": {"continuous2": slope_prior},
            "Intercept": intercept_prior,
            "continuous2": bare_slope_prior,
        }
    )

    intercept_prior.auto_scale = False
    slope_prior.auto_scale = False
    assert model.components["mu"].terms["Intercept"].prior == intercept_prior
    assert model.components["mu"].terms["continuous2"].prior == slope_prior


def test_unused_prior_nested_component(data_random_n100, monkeypatch):
    monkeypatch.setattr(bmb.config, "UNUSED_PRIORS", "error")
    prior = bmb.Prior("Normal", mu=0, sigma=1)
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")
    with pytest.raises(ValueError, match=r"sigma\.TYPO"):
        bmb.Model(formula, data_random_n100, priors={"sigma": {"TYPO": prior}})


def test_model_applies_bare_and_component_priors(data_random_n100, monkeypatch):
    monkeypatch.setattr(bmb.config, "UNUSED_PRIORS", "error")
    prior = bmb.Prior("Normal", mu=0, sigma=7.5)
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")
    model = bmb.Model(
        formula,
        data_random_n100,
        priors={"continuous2": prior, "sigma": {"continuous2": prior}},
    )
    prior.auto_scale = False  # the one in the model is set to False
    assert model.components["mu"].terms["continuous2"].prior == prior
    assert model.components["sigma"].terms["continuous2"].prior == prior


def test_set_priors_applies_bare_terms_with_several_components(data_random_n100):
    # `_set_priors` used to dispatch strictly by component name here, so a bare term name was
    # silently dropped while `Model(priors=...)` applied it.
    prior = bmb.Prior("Normal", mu=0, sigma=7.5)
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")
    model = bmb.Model(formula, data_random_n100)
    model.set_priors(priors={"continuous2": prior})
    prior.auto_scale = False
    assert model.components["mu"].terms["continuous2"].prior == prior


@pytest.mark.parametrize(
    "priors, component",
    [
        ({"common": "PLACEHOLDER"}, "mu"),  # applies to the intercept too, as at construction
        ({"sigma": {"common": "PLACEHOLDER"}}, "sigma"),  # same rule inside a named component
    ],
)
def test_set_priors_common_key_matches_model(data_random_n100, priors, component):
    # The "common" key must behave identically at construction and via set_priors -- including
    # the intercept, which the `common` *argument* deliberately does not touch.
    prior = bmb.Prior("Normal", mu=0, sigma=7.5)
    priors = {k: (prior if v == "PLACEHOLDER" else {"common": prior}) for k, v in priors.items()}
    formula = bmb.Formula("continuous1 ~ continuous2", "sigma ~ continuous2")

    via_init = bmb.Model(formula, data_random_n100, priors=priors)
    via_set = bmb.Model(formula, data_random_n100)
    via_set.set_priors(priors=priors)

    for name in ("Intercept", "continuous2"):
        expected = via_init.components[component].terms[name].prior
        assert via_set.components[component].terms[name].prior == expected


def test_set_priors_dict_wins_over_arguments(data_random_n100):
    # Entries in `priors` take precedence over the `common`/`group_specific` arguments, just as
    # term-specific entries always have.
    keyed = bmb.Prior("Normal", mu=0, sigma=7.5)
    argued = bmb.Prior("Normal", mu=0, sigma=1.5)
    model = bmb.Model("continuous1 ~ continuous2", data_random_n100)
    model.set_priors(priors={"common": keyed}, common=argued)
    keyed.auto_scale = False
    assert model.components["mu"].terms["continuous2"].prior == keyed


def test_group_specific_key_in_model(data_random_n100):
    gs_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1))
    model = bmb.Model(
        "continuous1 ~ continuous2 + (1|binary_cat)",
        data_random_n100,
        priors={"group_specific": gs_prior},
    )
    gs_prior.auto_scale = False
    assert model.components["mu"].terms["1|binary_cat"].prior == gs_prior
