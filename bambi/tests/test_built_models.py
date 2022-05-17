from os.path import dirname, join

import logging

import pytest

import numpy as np
import pandas as pd

from bambi import math
from bambi.models import Model
from bambi.priors import Prior


@pytest.fixture(scope="module")
def crossed_data():
    """
    Group specific effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    common effects:
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
def dm():
    """
    Data obtained from https://github.com/jswesner/nps_emergence/tree/v2_nps_emerge
    and used in Gamma GLM
    """
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "dm.csv"))
    return data


@pytest.fixture(scope="module")
def init_data():
    """
    Data used to test initialization method
    """
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "obs.csv"))
    return data


@pytest.fixture(scope="module")
def inhaler():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "inhaler.csv"))
    data["rating"] = pd.Categorical(data["rating"], categories=[1, 2, 3, 4])
    return data


@pytest.fixture(scope="module")
def categorical_family_categorical_predictor():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "categorical_family_categorical_predictor.csv"))
    return data


def test_empty_model(crossed_data):
    model0 = Model("Y ~ 0", crossed_data)
    model0.fit(tune=0, draws=1)


def test_intercept_only_model(crossed_data):
    model0 = Model("Y ~ 1", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_slope_only_model(crossed_data):
    model0 = Model("Y ~ 0 + continuous", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_cell_means_parameterization(crossed_data):
    model0 = Model("Y ~ 0 + threecats", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_3x4_common_anova(crossed_data):
    # add a four-level category that's perfectly crossed with threecats
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3

    # with intercept
    model0 = Model("Y ~ threecats*fourcats", crossed_data)
    fitted0 = model0.fit(tune=0, draws=1, init=None)
    assert len(fitted0.posterior.data_vars) == 5

    # without intercept (i.e., 2-factor cell means model)
    model1 = Model("Y ~ 0 + threecats*fourcats", crossed_data)
    fitted1 = model1.fit(tune=0, draws=1)
    assert len(fitted1.posterior.data_vars) == 4


def test_cell_means_with_covariate(crossed_data):
    model0 = Model("Y ~ 0 + threecats + continuous", crossed_data)
    model0.fit(tune=0, draws=1, init=None)

    # check that threecats priors have finite variance
    assert not (np.isinf(model0.terms["threecats"].prior.args["sigma"])).all()


def test_many_common_many_group_specific(crossed_data):
    # This test is kind of a mess, but it is very important, it checks lots of things.
    # delete a few values to also test dropna=True functionality
    crossed_data_missing = crossed_data.copy()
    crossed_data_missing.loc[0, "Y"] = np.nan
    crossed_data_missing.loc[1, "continuous"] = np.nan
    crossed_data_missing.loc[2, "threecats"] = np.nan

    # Here I'm comparing implicit/explicit intercepts for group specific effects work the same way.
    model0 = Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (1|item) + (0+continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model0.fit(
        init=None,
        tune=10,
        draws=10,
        chains=2,
    )

    model1 = Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model1.fit(
        tune=10,
        draws=10,
        chains=2,
    )
    # check that the group specific effects design matrices have the same shape
    X0 = pd.concat([pd.DataFrame(t.data) for t in model0.group_specific_terms.values()], axis=1)
    X1 = pd.concat([pd.DataFrame(t.data) for t in model1.group_specific_terms.values()], axis=1)
    assert X0.shape == X1.shape

    # check that the group specific effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )

    assert X0 == X1

    # check that models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

    # check that fit and add models have same priors for group specific effects
    priors0 = {x.name: x.prior.args["sigma"].args for x in model0.group_specific_terms.values()}
    priors1 = {x.name: x.prior.args["sigma"].args for x in model1.group_specific_terms.values()}

    # check dictionary keys
    assert set(priors0) == set(priors1)

    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_cell_means_with_many_group_specific_effects(crossed_data):
    # Group specific intercepts are added in different way, but the final result should be the same.
    formula = "Y ~" + "+".join(
        [
            "0",
            "threecats",
            "(threecats|subj)",
            "(1|subj)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(0 + threecats|site)",
            "(1|site)",
        ]
    )
    model0 = Model(formula, crossed_data)
    model0.fit(tune=0, draws=1)

    formula = "Y ~" + "+".join(
        [
            "0",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
    )
    model1 = Model(formula, crossed_data)
    model1.fit(tune=0, draws=1)

    # check that the group specific effects design matrices have the same shape
    X0 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.group_specific_terms.values()
        ],
        axis=1,
    )
    X1 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.group_specific_terms.values()
        ],
        axis=1,
    )
    assert X0.shape == X1.shape

    # check that the group specific effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    assert X0 == X1

    # check that fit and add models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    assert set(priors0) == set(priors1)

    # check that fit and add models have same priors for group specific effects
    priors0 = {
        x.name: x.prior.args["sigma"].args for x in model0.terms.values() if x.group_specific
    }
    priors1 = {
        x.name: x.prior.args["sigma"].args for x in model1.terms.values() if x.group_specific
    }
    assert set(priors0) == set(priors1)


def test_group_specific_categorical_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model("Y ~ continuous + (threecats:fourcats|site)", crossed_data)
    model.fit(tune=10, draws=10)


def test_logistic_regression_empty_index():
    data = pd.DataFrame({"y": np.random.choice(["a", "b"], 50), "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_logistic_regression_good_numeric():
    data = pd.DataFrame({"y": np.random.choice([1, 0], 50), "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_logistic_regression_bad_numeric():
    data = pd.DataFrame({"y": np.random.choice([1, 2], 50), "x": np.random.normal(size=50)})
    with pytest.raises(ValueError):
        model = Model("y ~ x", data, family="bernoulli")
        model.fit()


def test_logistic_regression_categoric():
    y = pd.Series(np.random.choice(["a", "b"], 50), dtype="category")
    data = pd.DataFrame({"y": y, "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_poisson_regression(crossed_data):
    # build model using fit and pymc3
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model0 = Model("count ~ dummy + continuous + threecats", crossed_data, family="poisson")
    model0.fit(tune=0, draws=1)

    # build model using add
    model1 = Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    model1.fit(tune=0, draws=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )

    assert X0 == X1

    # check that models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
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
    priors = {"Intercept": Prior("Uniform", lower=0, upper=1)}
    model = Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(method="laplace")
    mode_n = np.round(results["Intercept"][0], 2)
    std_n = np.round(results["Intercept"][1][0], 2)
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal((mode_n, std_n), (mode_a.item(), std_a.item()), decimal=2)


def test_prior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    # New default priors are too wide for this case... something to keep investigating
    model = Model(
        "count ~ threecats + continuous + dummy",
        crossed_data,
        family="poisson",
    )
    model.fit(tune=0, draws=2)
    pps = model.prior_predictive(draws=500)

    keys = ["Intercept", "threecats", "continuous", "dummy"]
    shapes = [(1, 500), (1, 500, 2), (1, 500), (1, 500)]

    for key, shape in zip(keys, shapes):
        assert pps.prior[key].shape == shape

    assert pps.prior_predictive["count"].shape == (1, 500, 120)
    assert pps.observed_data["count"].shape == (120,)

    pps = model.prior_predictive(draws=500, var_names=["count"])
    assert pps.groups() == ["prior_predictive", "observed_data"]

    pps = model.prior_predictive(draws=500, var_names=["Intercept"])
    assert pps.groups() == ["prior"]


def test_posterior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model = Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    fitted = model.fit(tune=0, draws=10, chains=2)
    pps = model.predict(fitted, kind="pps", inplace=False)

    assert pps.posterior_predictive["count"].shape == (2, 10, 120)

    pps = model.predict(fitted, kind="pps", inplace=True)

    assert pps is None
    assert fitted.posterior_predictive["count"].shape == (2, 10, 120)


def test_gamma_regression(dm):
    # simulated data
    np.random.seed(1234)
    N = 100
    x = np.random.uniform(-1, 1, N)
    a = 0.5
    b = 1.2
    y_true = np.exp(a + b * x)
    shape_true = 10  # alpha

    y = np.random.gamma(shape_true, y_true / shape_true, N)
    data = pd.DataFrame({"x": x, "y": y})
    model = Model("y ~ x", data, family="gamma", link="log")
    model.fit(draws=10, tune=10)

    # Real data, categorical predictor.
    data = dm[["order", "ind_mg_dry"]]
    model = Model("ind_mg_dry ~ order", data, family="gamma", link="log")
    model.fit(draws=10, tune=10)


def test_beta_regression():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "gasoline.csv"))
    model = Model("yield ~  temp + batch", data, family="beta", categorical="batch")
    model.fit(draws=10, tune=10, target_accept=0.9)


def test_t_regression():
    data = pd.DataFrame({"y": np.random.normal(size=100), "x": np.random.normal(size=100)})
    Model("y ~ x", data, family="t").fit(draws=10, tune=10)


def test_vonmises_regression():
    data = pd.DataFrame({"y": np.random.vonmises(0, 1, size=100), "x": np.random.normal(size=100)})
    Model("y ~ x", data, family="vonmises").fit(draws=10, tune=10)


def test_plot_priors(crossed_data):
    model = Model("Y ~ 0 + threecats", crossed_data)
    # Priors cannot be plotted until model is built.
    with pytest.raises(ValueError):
        model.plot_priors()
    model.build()
    model.plot_priors()


def test_model_graph(crossed_data):
    model = Model("Y ~ 0 + threecats", crossed_data)
    # Graph cannot be plotted until model is built.
    with pytest.raises(ValueError):
        model.graph()
    model.build()
    model.graph()


def test_potentials():
    data = pd.DataFrame(np.repeat((0, 1), (18, 20)), columns=["w"])
    priors = {"Intercept": Prior("Uniform", lower=0, upper=1)}
    potentials = [
        (("Intercept", "Intercept"), lambda x, y: math.switch(x < 0.45, y, -np.inf)),
        ("Intercept", lambda x: math.switch(x > 0.55, 0, -np.inf)),
    ]

    model = Model(
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
    pot0.__str__() == (
        "Elemwise{switch,no_inplace}(Elemwise{lt,no_inplace}.0, "
        "Intercept ~ Uniform, TensorConstant{-inf})"
    )
    pot1.__str__() == (
        "Elemwise{switch,no_inplace}(Elemwise{gt,no_inplace}.0, "
        "TensorConstant{0}, TensorConstant{-inf})"
    )


def test_binomial_regression():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    model = Model("prop(y, n) ~ x", data, family="binomial")
    model.fit(draws=10, tune=10)

    # Using constant instead of variable in data frame
    model = Model("prop(y, 62) ~ x", data, family="binomial")
    model.fit(draws=10, tune=10)


def test_init_fallback(init_data, caplog):
    model = Model("od ~ temp + (1|source) + 0", init_data)
    with caplog.at_level(logging.INFO):
        model.fit(draws=100, init="auto")
        assert "Initializing NUTS using jitter+adapt_diag..." in caplog.text
        assert "The default initialization" in caplog.text
        assert "Initializing NUTS using adapt_diag..." in caplog.text


def test_categorical_family(inhaler):
    model = Model("rating ~ period + carry + treat", inhaler, family="categorical")
    model.fit(draws=10, tune=10)


def test_categorical_family_varying_intercept(inhaler):
    model = Model("rating ~ period + carry + treat + (1|subject)", inhaler, family="categorical")
    model.fit(draws=10, tune=10)


def test_categorical_family_categorical_predictors(categorical_family_categorical_predictor):
    model = Model(
        "response ~ group + city", categorical_family_categorical_predictor, family="categorical"
    )
    model.fit(draws=10, tune=10)


def test_set_alias():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "predictor": np.random.normal(size=100),
            "group": np.random.choice(["A", "B", "C", "D"], size=100),
        }
    )
    model = Model("y ~ predictor + (predictor|group)", data)
    aliases = {
        "Intercept": "α",
        "predictor": "β",
        "1|group": "α_group",
        "predictor|group": "β_group",
        "sigma": "σ",
    }
    model.set_alias(aliases)
    model.build()
    new_names = set(["α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"])
    assert new_names.issubset(set(model.backend.model.named_vars))


def test_fit_include_mean(crossed_data):
    model = Model("Y ~ continuous*threecats", crossed_data)
    idata = model.fit(tune=400, draws=400, include_mean=True)
    assert idata.posterior["Y_mean"].shape[1:] == (400, 120)

    # Compare with the mean obtained with `model.predict()`
    mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

    model.predict(idata)
    predicted_mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

    assert np.array_equal(mean, predicted_mean)
