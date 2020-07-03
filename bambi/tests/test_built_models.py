import re

import arviz as az
import numpy as np
import pandas as pd
import pytest
import theano.tensor as tt

from bambi.models import Model, Term
from bambi.priors import Prior


@pytest.fixture(scope="module")
def crossed_data():
    """
    Random effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    Fixed effects:
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


def test_empty_model(crossed_data):
    model0 = Model(crossed_data)
    model0.add("Y ~ 0")
    model0.build(backend="pymc3")
    model0.fit(tune=0, samples=1)

    model1 = Model(crossed_data)
    model1.fit("Y ~ 0", run=False)
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1)

    # check that both models have same priors for fixed effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_intercept_only_model(crossed_data):
    # using fit
    model0 = Model(crossed_data)
    model0.fit("Y ~ 1", run=False)
    model0.build(backend="pymc3")
    model0.fit(tune=0, samples=1, init=None)

    # using add
    model1 = Model(crossed_data)
    model1.add("Y ~ 0")
    model1.add("1")
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1)

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_slope_only_model(crossed_data):
    # using fit
    model0 = Model(crossed_data)
    model0.fit("Y ~ 0 + continuous", run=False)
    model0.build(backend="pymc3")
    model0.fit(tune=0, samples=1, init=None)

    # using add
    model1 = Model(crossed_data)
    model1.add("Y ~ 0")
    model1.add("0 + continuous")
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_cell_means_parameterization(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit("Y ~ 0 + threecats", run=False)
    model0.build(backend="pymc3")
    model0.fit(tune=0, samples=1, init=None)

    # build model using add
    model1 = Model(crossed_data)
    model1.add("Y ~ 0")
    model1.add("0 + threecats")
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1)

    # check that design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )
    assert X0 == X1

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_3x4_fixed_anova(crossed_data):
    # add a four-level category that's perfectly crossed with threecats
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3

    # using fit, with intercept
    model0 = Model(crossed_data)
    model0.fit("Y ~ threecats*fourcats", run=False)
    model0.build(backend="pymc3")
    fitted0 = model0.fit(tune=0, samples=1, init=None)
    assert len(fitted0.posterior.data_vars) == 5

    # using fit, without intercept (i.e., 2-factor cell means model)
    model1 = Model(crossed_data)
    model1.fit("Y ~ 0 + threecats*fourcats", run=False)
    model1.build(backend="pymc3")
    fitted1 = model1.fit(tune=0, samples=1)
    assert len(fitted1.posterior.data_vars) == 4


def test_cell_means_with_covariate(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit("Y ~ 0 + threecats + continuous", run=False)
    model0.build(backend="pymc3")
    # model0.fit(tune=0, samples=1)

    # build model using add
    model1 = Model(crossed_data)
    model1.add("Y ~ 0")
    model1.add("0 + threecats")
    model1.add("0 + continuous")
    model1.build(backend="pymc3")
    # model1.fit(tune=0, samples=1)

    # check that design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )
    assert X0 == X1

    # check that threecats priors have finite variance
    assert not any(np.isinf(model0.terms["threecats"].prior.args["sigma"]))

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_many_fixed_many_random(crossed_data):
    # delete a few values to also test dropna=True functionality
    crossed_data_missing = crossed_data.copy()
    crossed_data_missing.loc[0, "Y"] = np.nan
    crossed_data_missing.loc[1, "continuous"] = np.nan
    crossed_data_missing.loc[2, "threecats"] = np.nan

    # build model using fit
    model0 = Model(crossed_data_missing, dropna=True)
    fitted = model0.fit(
        "Y ~ continuous + dummy + threecats",
        random=["0+threecats|subj", "1|item", "0+continuous|item", "dummy|item", "threecats|site"],
        backend="pymc3",
        init=None,
        tune=10,
        samples=10,
        chains=2,
    )
    # model0.build(backend='pymc3')
    # model0.fit(tune=0, samples=1)

    # build model using add(append=True)
    model1 = Model(crossed_data_missing, dropna=True)
    model1.add("Y ~ 1")
    model1.add("continuous")
    model1.add("dummy")
    model1.add("threecats")
    model1.add(random="0+threecats|subj")
    model1.add(random="1|item")
    model1.add(random="0+continuous|item")
    model1.add(random="dummy|item")
    model1.add(random="threecats|site")
    model1.build(backend="pymc3")

    # check that the random effects design matrices have the same shape
    X0 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.random_terms.values()
        ],
        axis=1,
    )
    X1 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model1.random_terms.values()
        ],
        axis=1,
    )
    assert X0.shape == X1.shape

    # check that the random effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that fixed effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )

    assert X0 == X1

    # check that models have same priors for fixed effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

    # check that fit and add models have same priors for random effects
    priors0 = {x.name: x.prior.args["sigma"].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args["sigma"].args for x in model1.terms.values() if x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_cell_means_with_many_random_effects(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit(
        "Y ~ 0 + threecats",
        random=["0+threecats|subj", "continuous|item", "dummy|item", "threecats|site"],
        run=False,
    )
    model0.build(backend="pymc3")
    # model0.fit(tune=0, samples=1)

    # build model using add(append=True)
    model1 = Model(crossed_data)
    model1.add("Y ~ 0")
    model1.add("0 + threecats")
    model1.add(random="0+threecats|subj")
    model1.add(random="1|item")
    model1.add(random="0+continuous|item")
    model1.add(random="dummy|item")
    model1.add(random="threecats|site")
    model1.build(backend="pymc3")
    # model1.fit(tune=0, samples=1)

    # check that the random effects design matrices have the same shape
    X0 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.random_terms.values()
        ],
        axis=1,
    )
    X1 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.random_terms.values()
        ],
        axis=1,
    )
    assert X0.shape == X1.shape

    # check that the random effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that fixed effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )
    assert X0 == X1

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # check that fit and add models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args["sigma"].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args["sigma"].args for x in model1.terms.values() if x.random}
    assert set(priors0) == set(priors1)


def test_logistic_regression(crossed_data):
    # build model using fit and pymc3
    model0 = Model(crossed_data)
    fitted0 = model0.fit(
        "threecats[b] ~ continuous + dummy",
        family="bernoulli",
        link="logit",
        backend="pymc3",
        tune=0,
        samples=1000,
    )
    # model0.build()
    # fitted0 = model0.fit()

    # build model using add
    model1 = Model(crossed_data)
    model1.add("threecats[b] ~ 1", family="bernoulli", link="logit")
    model1.add("continuous")
    model1.add("dummy")
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1)

    # build model using fit, pymc3 and theano link function
    model3 = Model(crossed_data)
    fitted3 = model3.fit(
        "threecats[b] ~ continuous + dummy",
        family="bernoulli",
        link=tt.nnet.sigmoid,
        backend="pymc3",
        tune=0,
        samples=1000,
    )

    # check that using a theano link function works
    assert np.allclose(az.summary(fitted0)["mean"], az.summary(fitted3)["mean"], atol=0.2)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that fixed effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )

    assert X0 == X1

    # check that models have same priors for fixed effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_poisson_regression(crossed_data):
    # build model using fit and pymc3
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model0 = Model(crossed_data)
    fitted = model0.fit(
        "count ~ threecats + continuous + dummy",
        family="poisson",
        backend="pymc3",
        tune=0,
        samples=1,
        init=None,
    )
    # model0.build()
    # model0.fit()

    # build model using add
    model1 = Model(crossed_data)
    model1.add("count ~ 1", family="poisson")
    model1.add("threecats")
    model1.add("continuous")
    model1.add("dummy")
    model1.build(backend="pymc3")
    model1.fit(tune=0, samples=1, init=None)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that fixed effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [tuple(t.data[:, lev]) for t in model0.fixed_terms.values() for lev in range(len(t.levels))]
    )
    X1 = set(
        [tuple(t.data[:, lev]) for t in model1.fixed_terms.values() for lev in range(len(t.levels))]
    )

    assert X0 == X1

    # check that models have same priors for fixed effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_laplace():
    data = pd.DataFrame(np.repeat((0, 1), (30, 60)), columns=["w"])
    model = Model(data=data)
    priors = {"Intercept": Prior("Uniform", lower=0, upper=1)}
    results = model.fit(
        "w ~ 1", family="bernoulli", link="identity", priors=priors, method="laplace"
    )
    mode_n = np.round(results["Intercept"][0], 2)
    std_n = np.round(results["Intercept"][1][0], 2)
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal((mode_n, std_n), (mode_a.item(), std_a.item()), decimal=2)
