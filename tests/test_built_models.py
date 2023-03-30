from os.path import dirname, join

import logging

import pytest

import numpy as np
import pandas as pd
import pymc as pm

from scipy.special import expit

import bambi as bmb
from bambi.terms import GroupSpecificTerm


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


@pytest.fixture(scope="module")
def data_100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "n1": rng.normal(size=size),
            "n2": rng.normal(size=size),
            "n3": rng.normal(size=size),
            "b0": rng.binomial(n=1, p=0.5, size=size),
            "b1": rng.choice(["a", "b"], size=size),
            "count1": rng.poisson(lam=2, size=size),
            "count2": rng.poisson(lam=2, size=size),
            "cat1": rng.choice(list("MNOP"), size=size),
            "cat2": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


@pytest.fixture(scope="module")
def data_1000():
    size = 1000
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "n1": rng.normal(size=size),
            "n2": rng.normal(size=size),
            "n3": rng.normal(size=size),
            "b0": rng.binomial(n=1, p=0.5, size=size),
            "b1": rng.choice(["a", "b"], size=size),
            "count1": rng.poisson(lam=2, size=size),
            "count2": rng.poisson(lam=2, size=size),
            "cat1": rng.choice(list("MNOP"), size=size),
            "cat2": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


def test_empty_model(crossed_data):
    model0 = bmb.Model("Y ~ 0", crossed_data)
    model0.fit(tune=0, draws=1)


def test_intercept_only_model(crossed_data):
    model0 = bmb.Model("Y ~ 1", crossed_data)
    model0.fit(tune=0, draws=1)


def test_slope_only_model(crossed_data):
    model0 = bmb.Model("Y ~ 0 + continuous", crossed_data)
    model0.fit(tune=0, draws=1)


def test_cell_means_parameterization(crossed_data):
    model0 = bmb.Model("Y ~ 0 + threecats", crossed_data)
    model0.fit(tune=0, draws=1)


def test_3x4_common_anova(crossed_data):
    # add a four-level category that's perfectly crossed with threecats
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3

    # with intercept
    model0 = bmb.Model("Y ~ threecats*fourcats", crossed_data)
    fitted0 = model0.fit(tune=0, draws=1)
    assert len(fitted0.posterior.data_vars) == 5

    # without intercept (i.e., 2-factor cell means model)
    model1 = bmb.Model("Y ~ 0 + threecats*fourcats", crossed_data)
    fitted1 = model1.fit(tune=0, draws=1)
    assert len(fitted1.posterior.data_vars) == 4


def test_cell_means_with_covariate(crossed_data):
    model = bmb.Model("Y ~ 0 + threecats + continuous", crossed_data)
    model.build()
    # check that threecats priors have finite variance
    assert not np.isinf(model.response_component.terms["threecats"].prior.args["sigma"]).all()


def test_many_common_many_group_specific(crossed_data):
    # This test is kind of a mess, but it is very important, it checks lots of things.
    # delete a few values to also test dropna=True functionality
    crossed_data_missing = crossed_data.copy()
    crossed_data_missing.loc[0, "Y"] = np.nan
    crossed_data_missing.loc[1, "continuous"] = np.nan
    crossed_data_missing.loc[2, "threecats"] = np.nan

    # Here I'm comparing implicit/explicit intercepts for group specific effects work the same way.
    model0 = bmb.Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (1|item) + (0+continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model0.fit(
        tune=10,
        draws=10,
        chains=2,
    )

    model1 = bmb.Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model1.fit(
        tune=10,
        draws=10,
        chains=2,
    )
    # Check that the group specific effects design matrices have the same shape
    X0 = pd.concat(
        [pd.DataFrame(t.data) for t in model0.response_component.group_specific_terms.values()],
        axis=1,
    )
    X1 = pd.concat(
        [pd.DataFrame(t.data) for t in model1.response_component.group_specific_terms.values()],
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
    X0_list = []
    X1_list = []
    for term in model0.response_component.common_terms.values():
        if term.levels is not None:
            for level_idx in range(len(term.levels)):
                X0_list.append(tuple(term.data[:, level_idx]))
        else:
            X0_list.append(tuple(term.data))

    for term in model1.response_component.common_terms.values():
        if term.levels is not None:
            for level_idx in range(len(term.levels)):
                X1_list.append(tuple(term.data[:, level_idx]))
        else:
            X1_list.append(tuple(term.data))

    assert set(X0_list) == set(X1_list)

    # check that models have same priors for common effects
    priors0 = {
        x.name: x.prior.args
        for x in model0.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }
    priors1 = {
        x.name: x.prior.args
        for x in model1.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }

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
    priors0 = {
        x.name: x.prior.args["sigma"].args
        for x in model0.response_component.group_specific_terms.values()
    }
    priors1 = {
        x.name: x.prior.args["sigma"].args
        for x in model1.response_component.group_specific_terms.values()
    }

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
    model0 = bmb.Model(formula, crossed_data)
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
    model1 = bmb.Model(formula, crossed_data)
    model1.fit(tune=0, draws=1)

    # check that the group specific effects design matrices have the same shape
    X0 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.response_component.group_specific_terms.values()
        ],
        axis=1,
    )
    X1 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.response_component.group_specific_terms.values()
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
            for t in model0.response_component.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.response_component.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    assert X0 == X1

    # check that fit and add models have same priors for common effects
    priors0 = {
        x.name: x.prior.args
        for x in model0.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }
    priors1 = {
        x.name: x.prior.args
        for x in model1.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }
    assert set(priors0) == set(priors1)

    # check that fit and add models have same priors for group specific effects
    priors0 = {
        x.name: x.prior.args["sigma"].args
        for x in model0.response_component.terms.values()
        if isinstance(x, GroupSpecificTerm)
    }
    priors1 = {
        x.name: x.prior.args["sigma"].args
        for x in model1.response_component.terms.values()
        if isinstance(x, GroupSpecificTerm)
    }
    assert set(priors0) == set(priors1)


def test_group_specific_categorical_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = bmb.Model("Y ~ continuous + (threecats:fourcats|site)", crossed_data)
    model.fit(tune=10, draws=10)


def test_logistic_regression_empty_index(data_100):
    model = bmb.Model("b1 ~ n1", data_100, family="bernoulli")
    model.fit()


def test_logistic_regression_good_numeric(data_100):
    model = bmb.Model("b0 ~ n1", data_100, family="bernoulli")
    model.fit()


def test_logistic_regression_bad_numeric():
    error_msg = "Numeric response must be all 0 and 1 for 'bernoulli' family"
    rng = np.random.default_rng(1234)
    data = pd.DataFrame({"y": rng.choice([1, 2], 50), "x": rng.normal(size=50)})
    with pytest.raises(ValueError, match=error_msg):
        model = bmb.Model("y ~ x", data, family="bernoulli")
        model.fit()


@pytest.mark.parametrize(
    "args",
    [
        ("mcmc", {}),
        ("nuts_numpyro", {"chain_method": "vectorized"}),
        ("nuts_blackjax", {"chain_method": "vectorized"}),
    ],
)
def test_logistic_regression_categoric_alternative_samplers(data_100, args):
    model = bmb.Model("b1 ~ n1", data_100, family="bernoulli")
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])


@pytest.mark.parametrize(
    "args",
    [
        ("mcmc", {}),
        ("nuts_numpyro", {"chain_method": "vectorized"}),
        ("nuts_blackjax", {"chain_method": "vectorized"}),
    ],
)
def test_regression_alternative_samplers(data_100, args):
    model = bmb.Model("n1 ~ n2", data_100)
    model.fit(tune=50, draws=50, inference_method=args[0], **args[1])


def test_laplace_regression(data_100):
    bmb_model = bmb.Model("n1 ~ n2", data_100, family="laplace")
    bmb_model.fit()


def test_poisson_regression(crossed_data):
    # build model using fit and pymc
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model0 = bmb.Model("count ~ dummy + continuous + threecats", crossed_data, family="poisson")
    model0.fit(tune=0, draws=1)

    # build model using add
    model1 = bmb.Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    model1.fit(tune=0, draws=1)

    # check that term names agree
    assert set(model0.response_component.terms) == set(model1.response_component.terms)

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different

    X0_list = []
    X1_list = []
    for term in model0.response_component.common_terms.values():
        if term.levels is not None:
            for level_idx in range(len(term.levels)):
                X0_list.append(tuple(term.data[:, level_idx]))
        else:
            X0_list.append(tuple(term.data))

    for term in model1.response_component.common_terms.values():
        if term.levels is not None:
            for level_idx in range(len(term.levels)):
                X1_list.append(tuple(term.data[:, level_idx]))
        else:
            X1_list.append(tuple(term.data))

    assert set(X0_list) == set(X1_list)

    # check that models have same priors for common effects
    priors0 = {
        x.name: x.prior.args
        for x in model0.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }
    priors1 = {
        x.name: x.prior.args
        for x in model1.response_component.terms.values()
        if not isinstance(x, GroupSpecificTerm)
    }
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
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    model = bmb.Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(inference_method="laplace")
    mode_n = results.posterior["Intercept"].mean().item()
    std_n = results.posterior["Intercept"].std().item()
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal((mode_n, std_n), (mode_a.item(), std_a.item()), decimal=2)


def test_vi():
    data = pd.DataFrame(np.repeat((0, 1), (30, 60)), columns=["w"])
    priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
    model = bmb.Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(inference_method="vi", method="advi")
    samples = results.sample(1000).posterior["Intercept"]
    mode_n = samples.mean()
    std_n = samples.std()
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal(
        (mode_n.item(), std_n.item()), (mode_a.item(), std_a.item()), decimal=2
    )


def test_prior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    # New default priors are too wide for this case... something to keep investigating
    model = bmb.Model(
        "count ~ threecats + continuous + dummy",
        crossed_data,
        family="poisson",
    )
    model.build()
    pps = model.prior_predictive(draws=500, random_seed=1234)

    keys = ["Intercept", "threecats", "continuous", "dummy"]
    shapes = [(1, 500), (1, 500, 2), (1, 500), (1, 500)]

    for key, shape in zip(keys, shapes):
        assert pps.prior[key].shape == shape

    assert pps.prior_predictive["count"].shape == (1, 500, 120)
    assert pps.observed_data["count"].shape == (120,)

    pps = model.prior_predictive(draws=500, var_names=["count"], random_seed=1234)
    assert pps.groups() == ["prior_predictive", "observed_data"]

    pps = model.prior_predictive(draws=500, var_names=["Intercept"], random_seed=1234)
    assert pps.groups() == ["prior", "observed_data"]


def test_posterior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model = bmb.Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    fitted = model.fit(tune=0, draws=10, chains=2)
    pps = model.predict(fitted, kind="pps", inplace=False)

    assert pps.posterior_predictive["count"].shape == (2, 10, 120)

    pps = model.predict(fitted, kind="pps", inplace=True)

    assert pps is None
    assert fitted.posterior_predictive["count"].shape == (2, 10, 120)


def test_gamma_regression(dm):
    # simulated data
    rng = np.random.default_rng(1234)
    a, b, N, shape_true = 0.5, 1.2, 100, 10  # alpha
    x = rng.uniform(-1, 1, N)
    y_true = np.exp(a + b * x)

    y = rng.gamma(shape_true, y_true / shape_true, N)
    data = pd.DataFrame({"x": x, "y": y})
    model = bmb.Model("y ~ x", data, family="gamma", link="log")
    model.fit(draws=10, tune=10)

    # Real data, categorical predictor.
    data = dm[["order", "ind_mg_dry"]]
    model = bmb.Model("ind_mg_dry ~ order", data, family="gamma", link="log")
    model.fit(draws=10, tune=10)


def test_beta_regression():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "gasoline.csv"))
    model = bmb.Model("yield ~  temp + batch", data, family="beta", categorical="batch")
    model.fit(draws=10, tune=10, target_accept=0.9)


def test_t_regression(data_100):
    bmb.Model("n1 ~ n2", data_100, family="t").fit(draws=10, tune=10)


def test_vonmises_regression():
    rng = np.random.default_rng(1234)
    data = pd.DataFrame({"y": rng.vonmises(0, 1, size=100), "x": rng.normal(size=100)})
    bmb.Model("y ~ x", data, family="vonmises").fit(draws=10, tune=10)


def test_quantile_regression():
    rng = np.random.default_rng(1234)
    x = rng.uniform(2, 10, 100)
    y = 2 * x + rng.normal(0, 0.6 * x**0.75)
    data = pd.DataFrame({"x": x, "y": y})
    bmb_model0 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 9})
    idata0 = bmb_model0.fit()
    bmb_model0.predict(idata0)

    bmb_model1 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 0.1})
    idata1 = bmb_model1.fit()
    bmb_model1.predict(idata1)

    assert np.all(
        idata0.posterior["y_mean"].mean(("chain", "draw"))
        > idata1.posterior["y_mean"].mean(("chain", "draw"))
    )


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
    assert pot0.__str__() == (
        "Elemwise{switch,no_inplace}(Elemwise{lt,no_inplace}.0, " "Intercept, TensorConstant{-inf})"
    )
    assert pot1.__str__() == (
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

    model = bmb.Model("prop(y, n) ~ x", data, family="binomial")
    model.fit(draws=10, tune=10)

    # Using constant instead of variable in data frame
    model = bmb.Model("prop(y, 62) ~ x", data, family="binomial")
    model.fit(draws=10, tune=10)


@pytest.mark.skip(reason="this example no longer trigger the fallback to adapt_diag")
def test_init_fallback(init_data, caplog):
    model = bmb.Model("od ~ temp + (1|source) + 0", init_data)
    with caplog.at_level(logging.INFO):
        model.fit(draws=100, init="auto")
        assert "Initializing NUTS using jitter+adapt_diag..." in caplog.text
        assert "The default initialization" in caplog.text
        assert "Initializing NUTS using adapt_diag..." in caplog.text


def test_categorical_family(inhaler):
    model = bmb.Model("rating ~ period + carry + treat", inhaler, family="categorical")
    model.fit(draws=10, tune=10)


def test_categorical_family_varying_intercept(inhaler):
    model = bmb.Model("rating ~ period + carry + treat + (1|subject)", inhaler, family="categorical")
    model.fit(draws=10, tune=10)


def test_categorical_family_categorical_predictors(categorical_family_categorical_predictor):
    formula = "response ~ group + city"
    model = bmb.Model(formula, categorical_family_categorical_predictor, family="categorical")
    model.fit(draws=10, tune=10)


def test_set_alias(data_100):
    model = bmb.Model("n1 ~ n2 + (n2|cat1)", data_100)
    aliases = {
        "Intercept": "α",
        "n2": "β",
        "1|cat1": "α_group",
        "n2|cat1": "β_group",
        "sigma": "σ",
    }
    model.set_alias(aliases)
    model.build()
    new_names = set(["α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"])
    assert new_names.issubset(set(model.backend.model.named_vars))


def test_fit_include_mean(crossed_data):
    draws = 500
    model = bmb.Model("Y ~ continuous * threecats", crossed_data)
    idata = model.fit(tune=draws, draws=draws, include_mean=True)
    assert idata.posterior["Y_mean"].shape[1:] == (draws, 120)

    # Compare with the mean obtained with `model.predict()`
    mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

    model.predict(idata)
    predicted_mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

    assert np.array_equal(mean, predicted_mean)


def test_group_specific_splines():
    x_check = pd.DataFrame(
        {
            "x": [
                82.0,
                143.0,
                426.0,
                641.0,
                1156.0,
                986.0,
                365.0,
                187.0,
                254.0,
                550.0,
                101.0,
                661.0,
                327.0,
                119.0,
            ],
            "day": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] * 2,
            "y": [
                571.0,
                684.0,
                1652.0,
                2130.0,
                2455.0,
                1874.0,
                1288.0,
                1011.0,
                1004.0,
                1993.0,
                593.0,
                1986.0,
                1503.0,
                711.0,
            ],
        }
    )
    knots = np.array([191.0, 297.0, 512.5])

    model = bmb.Model("y ~ (bs(x, knots=knots, intercept=False, degree=1)|day)", data=x_check)
    model.build()


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


def test_zero_inflated_poisson():
    rng = np.random.default_rng(121195)

    # Basic intercept-only model
    x = np.concatenate([np.zeros(250), rng.poisson(lam=3, size=750)])
    df = pd.DataFrame({"response": x})

    model = bmb.Model("response ~ 1", df, family="zero_inflated_poisson")
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")

    # Distributional model
    x = np.sort(rng.uniform(0.2, 3, size=1000))

    b0, b1 = 0.2, 0.9
    a0, a1 = 2.5, -0.7
    mu = np.exp(b0 + b1 * x)
    psi = expit(a0 + a1 * x)

    y = pm.draw(pm.ZeroInflatedPoisson.dist(mu=mu, psi=psi))
    df = pd.DataFrame({"y": y, "x": x})

    formula = bmb.Formula("y ~ x", "psi ~ x")
    model = bmb.Model(formula, df, family="zero_inflated_poisson")
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")


def test_zero_inflated_binomial():
    rng = np.random.default_rng(121195)

    # Basic intercept-only model
    y = pm.draw(pm.ZeroInflatedBinomial.dist(p=0.5, n=30, psi=0.7), draws=500, random_seed=1234)
    df = pd.DataFrame({"y": y})
    model = bmb.Model("p(y, 30) ~ 1", df, family="zero_inflated_binomial")
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")

    # Distributional model
    x = np.sort(rng.uniform(0.2, 3, size=500))
    b0, b1 = -0.5, 0.5
    a0, a1 = 2, -0.7
    p = expit(b0 + b1 * x)
    psi = expit(a0 + a1 * x)

    y = pm.draw(pm.ZeroInflatedBinomial.dist(p=p, psi=psi, n=30))
    df = pd.DataFrame({"y": y, "x": x})

    formula = bmb.Formula("prop(y, 30) ~ x", "psi ~ x")
    model = bmb.Model(formula, df, family="zero_inflated_binomial")
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")


def test_zero_inflated_negativebinomial():
    rng = np.random.default_rng(121195)

    # Basic intercept-only model
    y = pm.draw(
        pm.ZeroInflatedNegativeBinomial.dist(mu=5, alpha=30, psi=0.7), draws=500, random_seed=1234
    )
    df = pd.DataFrame({"y": y})
    priors = {"alpha": bmb.Prior("HalfNormal", sigma=20)}
    model = bmb.Model("y ~ 1", df, family="zero_inflated_negativebinomial", priors=priors)
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")

    # Distributional model
    x = np.sort(rng.uniform(0.2, 3, size=500))
    b0, b1 = 0.5, 0.35
    a0, a1 = 2, -0.7
    mu = np.exp(b0 + b1 * x)
    psi = expit(a0 + a1 * x)

    y = pm.draw(pm.ZeroInflatedNegativeBinomial.dist(mu=mu, alpha=30, psi=psi))
    df = pd.DataFrame({"y": y, "x": x})

    priors = {"alpha": bmb.Prior("HalfNormal", sigma=20)}
    formula = bmb.Formula("y ~ x", "psi ~ x")
    model = bmb.Model(formula, df, family="zero_inflated_negativebinomial", priors=priors)
    idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
    model.predict(idata, kind="pps")