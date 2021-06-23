import json
from multiprocessing.sharedctypes import Value
from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from bambi.models import Model
from bambi.priors import Family, Prior, PriorFactory

from statsmodels.tools.sm_exceptions import PerfectSeparationError


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


def test_family_class():
    prior = Prior("CheeseWhiz", holes=0, taste=-10)
    family = Family("cheese", prior, link="ferment", parent="holes")
    for name in ["name", "prior", "link", "parent"]:
        assert hasattr(family, name)


def test_prior_factory_init_from_default_config():
    pf = PriorFactory()
    for d in ["dists", "terms", "families"]:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    assert "normal" in pf.dists
    assert "common" in pf.terms
    assert "gaussian" in pf.families


def test_prior_factory_get_fail():
    # .get() must receive only, and only one, non None argument.
    pf = PriorFactory()
    with pytest.raises(ValueError):
        assert pf.get(dist="ñam", term="fri", family="frufi")
    with pytest.raises(ValueError):
        assert pf.get(dist="fali", term="fru")
    with pytest.raises(ValueError):
        assert pf.get()


def test_prior_factory_init_from_config():
    config_file = join(dirname(__file__), "data", "sample_priors.json")
    pf = PriorFactory(config_file)
    for d in ["dists", "terms", "families"]:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    config_dict = json.load(open(config_file, "r"))
    pf = PriorFactory(config_dict)
    for d in ["dists", "terms", "families"]:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    assert "feta" in pf.dists
    assert "hard" in pf.families
    assert "yellow" in pf.terms
    pf = PriorFactory(dists=config_dict["dists"])
    assert "feta" in pf.dists
    pf = PriorFactory(terms=config_dict["terms"])
    assert "yellow" in pf.terms
    pf = PriorFactory(families=config_dict["families"])
    assert "hard" in pf.families


def test_prior_retrieval():
    config_file = join(dirname(__file__), "data", "sample_priors.json")
    pf = PriorFactory(config_file)
    prior = pf.get(dist="asiago")
    assert prior.name == "Asiago"
    assert isinstance(prior, Prior)
    assert prior.args["hardness"] == 10
    with pytest.raises(KeyError):
        assert prior.args["holes"] == 4
    family = pf.get(family="hard")
    assert isinstance(family, Family)
    assert family.link == "grate"
    backup = family.prior.args["backup"]
    assert isinstance(backup, Prior)
    assert backup.args["flavor"] == 10000
    prior = pf.get(term="yellow")
    assert prior.name == "Swiss"

    # Test exception raising
    with pytest.raises(ValueError):
        pf.get(dist="apple")
    with pytest.raises(ValueError):
        pf.get(term="banana")
    with pytest.raises(ValueError):
        pf.get(family="cantaloupe")


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


def test_family_unsupported():
    family = Family("name", "prior", "link", "parent")
    with pytest.raises(ValueError):
        family._set_link("Empty")


def test_family_bad_type():
    data = pd.DataFrame({"x": [1], "y": [1]})

    with pytest.raises(ValueError):
        Model("y ~ x", data, family=0)

    with pytest.raises(ValueError):
        Model("y ~ x", data, family=set("gaussian"))

    with pytest.raises(ValueError):
        Model("y ~ x", data, family={"family": "gaussian"})


def test_family_unsupported_index_notation():
    data = pd.DataFrame({"x": [1], "y": [1]})
    with pytest.raises(ValueError):
        Model("y[1] ~ x", data, family="gaussian")


def test_complete_separation():
    data = pd.DataFrame({"y": [0] * 5 + [1] * 5, "g": ["a"] * 5 + ["b"] * 5})

    with pytest.raises(PerfectSeparationError):
        Model("y ~ g", data, family="bernoulli")

    # No error is raised
    priors = {"common": Prior("Normal", mu=0, sigma=10)}
    Model("y ~ g", data, family="bernoulli", priors=priors)


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
    assert model.terms["Intercept"].prior == prior
    assert model.terms["x"].prior == prior

    # Group-specific
    model.set_priors(group_specific=prior)
    assert model.terms["1|g"].prior == prior

    # By name
    model = Model("y ~ x + (1|g)", data)
    model.set_priors(priors={"x": prior})
    model.set_priors(priors={"1|g": prior})
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
    assert model.response.prior.args["sigma"] == priors["sigma"]

    priors = {"alpha": Prior("Uniform", lower=1, upper=20)}
    model = Model("y ~ x", data, family="negativebinomial", priors=priors)
    assert model.response.prior.args["alpha"] == priors["alpha"]

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma", priors=priors)
    assert model.response.prior.args["alpha"] == Prior("Uniform", lower=0, upper=50)

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma", priors=priors)
    assert model.response.prior.args["alpha"] == Prior("Uniform", lower=0, upper=50)


def test_set_response_prior():
    data = pd.DataFrame({"y": np.random.randint(3, 10, size=50), "x": np.random.normal(size=50)})

    priors = {"sigma": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data)
    model.set_priors(priors)
    assert model.response.prior.args["sigma"] == Prior("Uniform", lower=0, upper=50)

    priors = {"alpha": Prior("Uniform", lower=1, upper=20)}
    model = Model("y ~ x", data, family="negativebinomial")
    model.set_priors(priors)
    assert model.response.prior.args["alpha"] == Prior("Uniform", lower=1, upper=20)

    priors = {"alpha": Prior("Uniform", lower=0, upper=50)}
    model = Model("y ~ x", data, family="gamma")
    model.set_priors(priors)
    assert model.response.prior.args["alpha"] == Prior("Uniform", lower=0, upper=50)


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
