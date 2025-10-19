import pathlib
import re

import pytest

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm

from bambi.terms import GroupSpecificTerm

from helpers import assert_ip_dlogp


@pytest.fixture(scope="module")
def data_n100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "b1": rng.binomial(n=1, p=0.5, size=size),
            "n1": rng.poisson(lam=2, size=size),
            "n2": rng.poisson(lam=2, size=size),
            "y1": rng.normal(size=size),
            "y2": rng.normal(size=size),
            "y3": rng.normal(size=size),
            "cat2": rng.choice(["a", "b"], size=size),
            "cat4": rng.choice(list("MNOP"), size=size),
            "cat5": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


@pytest.fixture(scope="module")
def cat_response_cat_preds_data():
    data = pd.read_csv(
        pathlib.Path(__file__).parent / "data" / "categorical_family_categorical_predictor.csv"
    )
    return data


@pytest.fixture(scope="module")
def zi_count_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 30, 70
    y = np.concatenate([np.zeros(n1), rng.poisson(3, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def zi_bounded_count_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 40, 60
    y = np.concatenate([np.zeros(n1), rng.binomial(n=30, p=0.6, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def zi_continuous_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 40, 60
    y = np.concatenate([np.zeros(n1), rng.gamma(shape=2, scale=3, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def truncated_data():
    rng = np.random.default_rng(12345)
    slope, intercept, sigma, N = 1, 0, 2, 200
    x = rng.uniform(-10, 10, N)
    y = rng.normal(loc=slope * x + intercept, scale=sigma)
    bounds = [-5, 5]
    keep = (y >= bounds[0]) & (y <= bounds[1])
    xt = x[keep]
    yt = y[keep]

    return pd.DataFrame({"x": xt, "y": yt})


class FitPredictParent:
    def fit(self, model, **kwargs):
        self.assert_ip_dlogp(model)
        return model.fit(tune=200, draws=200, chains=2, **kwargs)

    def predict_oos(self, model, idata, data=None):
        # Reuse the original data
        if data is None:
            data = model.data.head()
        return model.predict(idata, kind="response", data=data, inplace=False)

    def assert_ip_dlogp(self, model):
        model.build()
        assert_ip_dlogp(model)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestGaussian(FitPredictParent):
    def test_intercept_only_model(self, data_crossed):
        model = bmb.Model("Y ~ 1", data_crossed)
        idata = self.fit(model)
        self.predict_oos(model, idata)

    def test_slope_only_model(self, data_crossed):
        model = bmb.Model("Y ~ 0 + continuous", data_crossed)
        idata = self.fit(model)
        self.predict_oos(model, idata)

    def test_cell_means_parameterization(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats", data_crossed)
        idata = self.fit(model)
        assert list(idata.posterior["threecats_dim"]) == ["a", "b", "c"]
        self.predict_oos(model, idata)

    def test_2_factors_saturated(self, data_crossed):
        model = bmb.Model("Y ~ threecats*fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "sigma",
        }
        assert list(idata.posterior["threecats_dim"].values) == ["b", "c"]
        assert list(idata.posterior["fourcats_dim"].values) == ["b", "c", "d"]
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        ]
        self.predict_oos(model, idata)

    def test_2_factors_no_intercept(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats*fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "sigma",
        }
        assert list(idata.posterior["threecats_dim"].values) == ["a", "b", "c"]
        assert list(idata.posterior["fourcats_dim"].values) == ["b", "c", "d"]
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        ]
        self.predict_oos(model, idata)

    def test_2_factors_cell_means(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats:fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"threecats:fourcats", "sigma"}
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "a, a",
            "a, b",
            "a, c",
            "a, d",
            "b, a",
            "b, b",
            "b, c",
            "b, d",
            "c, a",
            "c, b",
            "c, c",
            "c, d",
        ]
        self.predict_oos(model, idata)

    def test_cell_means_with_covariate(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats + continuous", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"threecats", "continuous", "sigma"}
        assert list(idata.posterior["threecats_dim"].values) == ["a", "b", "c"]
        self.predict_oos(model, idata)

    def test_many_common_many_group_specific(self, data_crossed):
        # Comparing implicit/explicit intercepts for group specific effects work the same way.
        terms0 = [
            "continuous",
            "dummy",
            "threecats",
            "(threecats|subj)",
            "(1|item)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
        terms1 = [
            "continuous",
            "dummy",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]

        model0 = bmb.Model("Y ~ " + " + ".join(terms0), data_crossed)
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), data_crossed)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

        # Check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [pd.DataFrame(t.data) for t in model0.components["mu"].group_specific_terms.values()],
            axis=1,
        )
        X1 = pd.concat(
            [pd.DataFrame(t.data) for t in model1.components["mu"].group_specific_terms.values()],
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
        for term in model0.components["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.components["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.components["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.components["mu"].terms.values()
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
            for x in model0.components["mu"].group_specific_terms.values()
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.components["mu"].group_specific_terms.values()
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

    def test_cell_means_with_many_group_specific_effects(self, data_crossed):
        # Group specific intercepts are added in different way, but the final result should be the same.
        terms0 = [
            "0",
            "threecats",
            "(threecats|subj)",
            "(1|subj)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(0 + threecats|site)",
            "(1|site)",
        ]

        terms1 = [
            "0",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
        model0 = bmb.Model("Y ~ " + " + ".join(terms0), data_crossed)
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), data_crossed)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

        # check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [
                (
                    pd.DataFrame(t.data)
                    if not isinstance(t.data, dict)
                    else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
                )
                for t in model0.components["mu"].group_specific_terms.values()
            ],
            axis=1,
        )
        X1 = pd.concat(
            [
                (
                    pd.DataFrame(t.data)
                    if not isinstance(t.data, dict)
                    else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
                )
                for t in model0.components["mu"].group_specific_terms.values()
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
                for t in model0.components["mu"].common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        X1 = set(
            [
                tuple(t.data[:, lev])
                for t in model1.components["mu"].common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        assert X0 == X1

        # check that fit and add models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.components["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.components["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

        # check that fit and add models have same priors for group specific effects
        priors0 = {
            x.name: x.prior.args["sigma"].args
            for x in model0.components["mu"].terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.components["mu"].terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

    def test_group_specific_categorical_interaction(self, data_crossed):
        model = bmb.Model("Y ~ continuous + (threecats:fourcats|site)", data_crossed)
        idata = self.fit(model)
        self.predict_oos(model, idata)

        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "continuous",
            "sigma",
            "1|site_sigma",
            "threecats:fourcats|site_sigma",
            "1|site",
            "threecats:fourcats|site",
        }
        assert set(idata.posterior["threecats:fourcats|site"].coords) == {
            "chain",
            "draw",
            "site__factor_dim",
            "threecats:fourcats__expr_dim",
        }
        assert set(idata.posterior["1|site"].coords) == {"chain", "draw", "site__factor_dim"}
        assert set(idata.posterior["1|site_sigma"].coords) == {"chain", "draw"}
        assert set(idata.posterior["threecats:fourcats|site_sigma"].coords) == {
            "chain",
            "draw",
            "threecats:fourcats__expr_dim",
        }

        assert set(idata.posterior["threecats:fourcats__expr_dim"].values) == {
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        }
        assert set(idata.posterior["site__factor_dim"].values) == {"0", "1", "2", "3", "4"}

    def test_fit_include_mean(self, data_crossed):
        draws = 100
        model = bmb.Model("Y ~ continuous * threecats", data_crossed)
        idata = model.fit(tune=draws, draws=draws, chains=2, include_response_params=True)
        assert idata.posterior["mu"].shape[1:] == (draws, 120)

        # Compare with the mean obtained with `model.predict()`
        mean = idata.posterior["mu"].stack(sample=("chain", "draw")).values.mean(1)

        model.predict(idata)
        predicted_mean = idata.posterior["mu"].stack(sample=("chain", "draw")).values.mean(1)

        assert np.array_equal(mean, predicted_mean)

    def test_group_specific_splines(self):
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
        idata = self.fit(model)
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBernoulli(FitPredictParent):
    def assert_posterior_predictive_range(self, model, idata):
        y_name = model.response_component.term.name
        y_posterior_predictive = idata.posterior_predictive[y_name].to_numpy()
        assert set(np.unique(y_posterior_predictive)) == {0, 1}

    def assert_mean_range(self, idata):
        y_mean_name = "p"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert ((0 < y_mean_posterior) & (y_mean_posterior < 1)).all()

    def test_bernoulli_empty_index(self, data_n100):
        model = bmb.Model("b1 ~ 1 + y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata)
        self.assert_posterior_predictive_range(model, idata)

        # out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(idata)
        self.assert_posterior_predictive_range(model, idata)

    def test_bernoulli_good_numeric(self, data_n100):
        model = bmb.Model("b1 ~ y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata)
        self.assert_posterior_predictive_range(model, idata)

    def test_bernoulli_bad_numeric(self, data_n100):
        error_msg = "Numeric response must be all 0 and 1 for 'bernoulli' family"
        with pytest.raises(ValueError, match=error_msg):
            model = bmb.Model("y1 ~ y2", data_n100, family="bernoulli")
            self.fit(model)

    def test_categorical_group_specific(self, data_n100):
        # see https://github.com/bambinos/bambi/issues/447

        formula = "b1 ~ 0 + cat2 + y2 + (0 + cat2|cat5) + (0 + y2| cat4 + cat5)"
        model = bmb.Model(formula, data_n100, family="bernoulli")
        idata = self.fit(model)
        idata = self.predict_oos(model, idata)

        self.assert_mean_range(idata)
        self.assert_posterior_predictive_range(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBinomial(FitPredictParent):
    def assert_mean_range(self, idata):
        y_mean_name = "p"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert ((0 < y_mean_posterior) & (y_mean_posterior < 1)).all()

    def test_binomial_regression(self, data_beetle):
        model = bmb.Model("prop(y, n) ~ x", data_beetle, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata)
        y_reshaped = data_beetle["n"].to_numpy()[None, None, :]

        assert (idata.posterior_predictive["prop(y, n)"].to_numpy() <= y_reshaped).all()
        assert (0 <= idata.posterior_predictive["prop(y, n)"].to_numpy()).all()

        y_reshaped = data_beetle["n"].to_numpy()[None, None, :3]
        idata = self.predict_oos(model, idata, data=model.data.head(3))
        self.assert_mean_range(idata)
        assert (idata.posterior_predictive["prop(y, n)"].to_numpy() <= y_reshaped).all()

        # Test log-likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_beetle, inplace=False)
        assert (
            (idata_2.log_likelihood["prop(y, n)"] == idata.log_likelihood["prop(y, n)"])
            .all()
            .item()
        )

    def test_binomial_regression_constant(self, data_beetle):
        # Uses a constant instead of variable in data frame
        model = bmb.Model("p(y, 62) ~ x", data_beetle, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata)
        assert (idata.posterior_predictive["p(y, 62)"].to_numpy() <= 62).all()
        assert (0 <= idata.posterior_predictive["p(y, 62)"].to_numpy()).all()

        # Out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(idata)
        assert (idata.posterior_predictive["p(y, 62)"].to_numpy() <= 62).all()

        # Test log-likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_beetle, inplace=False)
        assert (idata_2.log_likelihood["p(y, 62)"] == idata.log_likelihood["p(y, 62)"]).all().item()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestPoisson(FitPredictParent):
    def assert_mean_range(self, idata):
        y_mean_name = "mu"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert (y_mean_posterior > 0).all()

    def test_poisson_regression(self, data_crossed):
        data_crossed["count"] = (data_crossed["Y"] - data_crossed["Y"].min()).round()
        model0 = bmb.Model("count ~ dummy + continuous + threecats", data_crossed, family="poisson")
        idata0 = self.fit(model0)
        idata0 = self.predict_oos(model0, idata0)
        self.assert_mean_range(idata0)

        # build model using add
        model1 = bmb.Model("count ~ threecats + continuous + dummy", data_crossed, family="poisson")
        idata1 = self.fit(model1)
        idata1 = self.predict_oos(model1, idata1)
        self.assert_mean_range(idata1)

        # check that term names agree
        assert set(model0.components["mu"].terms) == set(model1.components["mu"].terms)

        # check that common effect design matrices are the same,
        # even if term names / level names / order of columns is different

        X0_list = []
        X1_list = []
        for term in model0.components["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.components["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.components["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.components["mu"].terms.values()
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

        # Now test prior predictive
        pps = model1.prior_predictive(draws=200, random_seed=1234)

        keys = ["Intercept", "threecats", "continuous", "dummy"]
        shapes = [(1, 200), (1, 200, 2), (1, 200), (1, 200)]

        for key, shape in zip(keys, shapes):
            assert pps.prior[key].shape == shape

        assert pps.prior_predictive["count"].shape == (1, 200, 120)
        assert pps.observed_data["count"].shape == (120,)

        pps = model1.prior_predictive(draws=200, var_names=["count"], random_seed=1234)
        assert pps.groups() == ["prior_predictive", "observed_data"]

        pps = model1.prior_predictive(draws=200, var_names=["Intercept"], random_seed=1234)
        assert pps.groups() == ["prior", "observed_data"]

        # Now test posterior predictive
        # Fit again to make sure we fix the number of chains
        idata = model1.fit(tune=50, draws=50, chains=2)
        pps = model1.predict(idata, kind="response", inplace=False)
        assert pps.posterior_predictive["count"].shape == (2, 50, 120)

        pps = model1.predict(idata, kind="response", inplace=True)
        assert pps is None
        assert idata.posterior_predictive["count"].shape == (2, 50, 120)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestNegativeBinomial(FitPredictParent):
    # To Do: Could be modified to follow the same format than the others
    def test_predict_negativebinomial(self, data_n100):

        model = bmb.Model("n1 ~ y1", data_n100, family="negativebinomial")
        idata = self.fit(model)

        model.predict(idata, kind="response_params")
        assert (0 < idata.posterior["mu"]).all()

        model.predict(idata, kind="response")
        assert (np.equal(np.mod(idata.posterior_predictive["n1"].values, 1), 0)).all()

        model.predict(idata, kind="response_params", data=data_n100.iloc[:20, :])
        assert (0 < idata.posterior["mu"]).all()

        model.predict(idata, kind="response", data=data_n100.iloc[:20, :])
        assert (np.equal(np.mod(idata.posterior_predictive["n1"].values, 1), 0)).all()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestLaplace(FitPredictParent):
    def test_laplace_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="laplace")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "b"}
        assert (idata.posterior["b"] > 0).all().item()

        idata = self.predict_oos(model, idata)
        assert "mu" in idata.posterior


@pytest.mark.usefixtures("mock_pymc_sample")
class TestGamma(FitPredictParent):
    def test_gamma_regression(self, data_n100):
        # Construct a positive variable
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ y2 + y3 + n1 + cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "y3", "n1", "cat4", "alpha"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["o"] >= 0).all().item()

        # Compute log likelihood
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_n100, inplace=False)
        assert (idata.log_likelihood["o"] == idata_2.log_likelihood["o"]).all().item()

    def test_gamma_regression_categoric(self, data_n100):
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ 0 + cat2:cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"cat2:cat4", "alpha"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["o"] >= 0).all().item()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBeta(FitPredictParent):
    def test_beta_regression(self, data_gasoline):
        model = bmb.Model(
            "yield ~  temp + batch", data_gasoline, family="beta", categorical="batch"
        )
        idata = self.fit(model, target_accept=0.9, random_seed=1234)

        # To Do: Could be adjusted but this is what we had before
        model.predict(idata, kind="response_params")
        model.predict(idata, kind="response")

        assert (0 < idata.posterior["mu"]).all() & (idata.posterior["mu"] < 1).all()
        assert (0 <= idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] <= 1
        ).all()

        model.predict(idata, kind="response_params", data=data_gasoline.iloc[:20, :])
        model.predict(idata, kind="response", data=data_gasoline.iloc[:20, :])

        assert (0 < idata.posterior["mu"]).all() & (idata.posterior["mu"] < 1).all()
        assert (0 <= idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] <= 1
        ).all()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestStudentT(FitPredictParent):
    def test_t_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="t")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "nu", "sigma"}
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestVonMises(FitPredictParent):
    def test_vonmises_regression(self):
        rng = np.random.default_rng(1234)
        data = pd.DataFrame({"y": rng.vonmises(0, 1, size=100), "x": rng.normal(size=100)})
        model = bmb.Model("y ~ x", data, family="vonmises")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "x", "kappa"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["y"].min() >= -np.pi).item() and (
            idata.posterior_predictive["y"].max() <= np.pi
        ).item()


# NOTE: We don't use mock_pymc_sample because the assertion needs actual posterior samples to work.
class TestAsymmetricLaplace(FitPredictParent):
    # This test doesn't follow the previous pattern but it works...
    def test_quantile_regression(self):
        rng = np.random.default_rng(1234)
        x = rng.uniform(2, 10, 100)
        y = 2 * x + rng.normal(0, 0.6 * x**0.75)
        data = pd.DataFrame({"x": x, "y": y})
        bmb_model0 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 9})
        idata0 = bmb_model0.fit(chains=2)
        bmb_model0.predict(idata0)

        bmb_model1 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 0.1})
        idata1 = bmb_model1.fit(chains=2)
        bmb_model1.predict(idata1)

        assert np.all(
            idata0.posterior["mu"].mean(("chain", "draw"))
            > idata1.posterior["mu"].mean(("chain", "draw"))
        )


@pytest.mark.usefixtures("mock_pymc_sample")
class TestCategorical(FitPredictParent):
    # assert pps.shape[-1] == inhaler.shape[0]
    def assert_mean_sum(self, model, idata):
        y_mean_name = "p"
        y_dim = model.response_component.term.name + "_dim"
        y_mean_posterior = idata.posterior[y_mean_name]
        assert np.allclose(y_mean_posterior.sum(y_dim).to_numpy(), 1)

    def assert_mean_range(self, model, idata):
        y_mean_name = "p"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert ((0 < y_mean_posterior) & (y_mean_posterior < 1)).all()

    def assert_posterior_predictive_range(self, model, idata, n):
        y_name = model.response_component.term.name
        y_posterior_predictive = idata.posterior_predictive[y_name].to_numpy()
        assert set(np.unique(y_posterior_predictive)).issubset(set(range(n)))

    def test_basic(self, data_inhaler):
        model = bmb.Model("rating ~ period + carry + treat", data_inhaler, family="categorical")
        idata = self.fit(model)

        for name in ["Intercept", "period", "carry", "treat"]:
            assert list(idata.posterior[name].coords) == ["chain", "draw", "rating_reduced_dim"]

        assert list(idata.posterior.coords["rating_reduced_dim"].values) == ["2", "3", "4"]

        idata = self.predict_oos(model, idata)
        assert list(idata.posterior["p"].coords) == [
            "chain",
            "draw",
            "__obs__",
            "rating_dim",
        ]
        assert list(idata.posterior.coords["rating_dim"].values) == ["1", "2", "3", "4"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_posterior_predictive_range(model, idata, len(np.unique(data_inhaler["rating"])))

    def test_varying_intercept(self, data_inhaler):
        formula = "rating ~ period + carry + treat + (1|subject)"
        model = bmb.Model(formula, data_inhaler, family="categorical")
        idata = self.fit(model)

        for name in ["Intercept", "period", "carry", "treat"]:
            assert set(idata.posterior[name].coords) == {"chain", "draw", "rating_reduced_dim"}

        assert set(idata.posterior["1|subject"].coords) == {
            "chain",
            "draw",
            "rating_reduced_dim",
            "subject__factor_dim",
        }

        assert (
            idata.posterior["subject__factor_dim"].values
            == np.unique(data_inhaler["subject"]).astype(str)
        ).all()

        assert list(idata.posterior.coords["rating_reduced_dim"].values) == ["2", "3", "4"]

        idata = self.predict_oos(model, idata)
        assert set(idata.posterior["p"].coords) == {
            "chain",
            "draw",
            "__obs__",
            "rating_dim",
        }
        assert list(idata.posterior.coords["rating_dim"].values) == ["1", "2", "3", "4"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_posterior_predictive_range(model, idata, len(np.unique(data_inhaler["rating"])))

    def test_categorical_predictors(self, cat_response_cat_preds_data):
        formula = "response ~ group + city"
        model = bmb.Model(formula, cat_response_cat_preds_data, family="categorical")
        idata = self.fit(model)

        assert set(idata.posterior["group"].coords) == {
            "chain",
            "draw",
            "response_reduced_dim",
            "group_dim",
        }
        assert set(idata.posterior["city"].coords) == {
            "chain",
            "draw",
            "response_reduced_dim",
            "city_dim",
        }
        assert list(idata.posterior["group_dim"].values) == ["group 2", "group 3"]
        assert list(idata.posterior["city_dim"].values) == ["Rosario", "San Luis"]
        assert list(idata.posterior["response_reduced_dim"].values) == ["B", "C", "D"]

        idata = self.predict_oos(model, idata)
        assert list(idata.posterior["response_dim"].values) == ["A", "B", "C", "D"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_posterior_predictive_range(model, idata, 4)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestZeroInflatedFamilies(FitPredictParent):
    @pytest.mark.parametrize(
        "formula, data_name, family, priors",
        [  # Zero Inflated Poisson
            (bmb.Formula("y ~ x"), "zi_count_data", "zero_inflated_poisson", None),
            (bmb.Formula("y ~ x", "psi ~ x"), "zi_count_data", "zero_inflated_poisson", None),
            # Zero Inflated Negative Binomial
            (
                bmb.Formula("y ~ x"),
                "zi_count_data",
                "zero_inflated_negativebinomial",
                {"alpha": bmb.Prior("HalfNormal", sigma=20)},
            ),
            (
                bmb.Formula("y ~ x", "psi ~ x"),
                "zi_count_data",
                "zero_inflated_negativebinomial",
                {"alpha": bmb.Prior("HalfNormal", sigma=20)},
            ),
            # Zero Inflated Binomial
            (bmb.Formula("p(y, 30) ~ 1"), "zi_bounded_count_data", "zero_inflated_binomial", None),
            (
                bmb.Formula("p(y, 30) ~ 1", "psi ~ x"),
                "zi_bounded_count_data",
                "zero_inflated_binomial",
                None,
            ),
        ],
    )
    def test_family(self, formula, data_name, family, priors, request):
        data = request.getfixturevalue(data_name)
        model = bmb.Model(formula, data, priors=priors, family=family)
        idata = self.fit(model)
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestHurdle(FitPredictParent):
    @pytest.mark.parametrize(
        "data_name, family",
        [
            ("zi_count_data", "hurdle_poisson"),
            ("zi_count_data", "hurdle_negativebinomial"),
            ("zi_continuous_data", "hurdle_gamma"),
            ("zi_continuous_data", "hurdle_lognormal"),
        ],
    )
    def test_hurlde_families(self, data_name, family, request):
        # To access 'data' which is a fixture
        data = request.getfixturevalue(data_name)
        model = bmb.Model("y ~ 1", data, family=family)
        idata = self.fit(model, random_seed=1234)
        self.predict_oos(model, idata)


# NOTE: We don't use mock_pymc_sample because the assertion needs actual posterior samples to work.
class TestOrdinal(FitPredictParent):
    @pytest.mark.parametrize(
        "family, link",
        [
            ("cumulative", "logit"),
            ("cumulative", "probit"),
            ("cumulative", "cloglog"),
            ("sratio", "logit"),
            ("sratio", "probit"),
            ("sratio", "cloglog"),
        ],
    )
    def test_ordinal_families(self, data_inhaler, family, link):
        # To have both numeric and categoric predictors
        data_inhaler["carry"] = pd.Categorical(data_inhaler["carry"])
        model = bmb.Model("rating ~ period + carry + treat", data_inhaler, family=family, link=link)
        idata = self.fit(model, random_seed=1234)
        idata = self.predict_oos(model, idata)

        assert np.allclose(idata.posterior["p"].sum("rating_dim").to_numpy(), 1)
        assert set(np.unique(idata.posterior_predictive["rating"])).issubset({0, 1, 2, 3})

    def test_cumulative_family_priors(self, data_inhaler):
        priors = {
            "threshold": bmb.Prior(
                "Normal",
                mu=[-0.5, 0, 0.5],
                sigma=1.5,
                transform=pm.distributions.transforms.ordered,
            )
        }
        model = bmb.Model(
            "rating ~ 0 + period + carry + treat", data_inhaler, family="cumulative", priors=priors
        )
        idata = self.fit(model, random_seed=1234)
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestCensoredResponses(FitPredictParent):
    def test_model_with_intercept(self, data_kidney):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age",
            data_kidney,
            family="weibull",
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)

    def test_model_without_intercept(self, data_kidney):
        priors = {
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 0 + sex + age",
            data_kidney,
            family="weibull",
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)

    def test_model_with_group_specific_effects(self, data_kidney):
        # Model 3, with group-specific effects
        priors = {
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
            "sex": bmb.Prior("Normal", mu=0, sigma=1),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "1|patient": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("InverseGamma", alpha=5, beta=10)
            ),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age + (1|patient)",
            data_kidney,
            family="weibull",
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestTruncatedResponse(FitPredictParent):
    def test_truncated_response(self, truncated_data):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "x": bmb.Prior("Normal", mu=0, sigma=1),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        }
        model = bmb.Model("truncated(y, -5, 5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # PyMC seems to automatically dispatch to TruncatedNormal
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.TruncatedNormal.rv_type)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestConstrainedResponse(FitPredictParent):
    def test_constrained_response(self, truncated_data):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "x": bmb.Prior("Normal", mu=0, sigma=1),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        }
        model = bmb.Model("constrained(y, -5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.posterior_predictive["constrained(y, -5)"].to_numpy().min() > -5

        model = bmb.Model("constrained(y, ub=5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.posterior_predictive["constrained(y, ub=5)"].to_numpy().max() < 5

        model = bmb.Model("constrained(y, -5, 5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.posterior_predictive["constrained(y, -5, 5)"].to_numpy().min() > -5
        assert idata.posterior_predictive["constrained(y, -5, 5)"].to_numpy().max() < 5


@pytest.mark.usefixtures("mock_pymc_sample")
class TestMultinomial(FitPredictParent):
    def assert_posterior_predictive(self, model, idata):
        y_name = model.response_component.term.name
        y_posterior_predictive = idata.posterior_predictive[y_name].to_numpy()
        assert (y_posterior_predictive.sum(-1).var((0, 1)) == 0).all()

    def test_intercept_only(self, data_multinomial):
        model = bmb.Model("c(y1, y2, y3, y4) ~ 1", data_multinomial, family="multinomial")
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_posterior_predictive(model, idata)

        # Out of sample with different number of rows, see issue #845
        idata = self.predict_oos(model, idata, data=model.data.sample(frac=0.8, random_state=1211))
        self.assert_posterior_predictive(model, idata)

    def test_numerical_predictors(self, data_multinomial):
        model = bmb.Model(
            "c(y1, y2, y3, y4) ~ treat + carry", data_multinomial, family="multinomial"
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_posterior_predictive(model, idata)

        # Log likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_multinomial, inplace=False)
        name = "c(y1, y2, y3, y4)"
        assert (idata.log_likelihood[name] == idata_2.log_likelihood[name]).all().item()

    def test_categorical_predictors(self, data_multinomial):
        data_multinomial["treat"] = data_multinomial["treat"].replace({-0.5: "A", 0.5: "B"})
        data_multinomial["carry"] = data_multinomial["carry"].replace({-1: "a", 0: "b", 1: "c"})

        model = bmb.Model(
            "c(y1, y2, y3, y4) ~ treat + carry", data_multinomial, family="multinomial"
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_posterior_predictive(model, idata)

    def test_group_specific_effects(self):
        data = pd.DataFrame(
            {
                "state": ["A", "B", "C"],
                "y1": [35298, 1885, 5775],
                "y2": [167328, 20731, 21564],
                "y3": [212682, 37716, 20222],
                "y4": [37966, 5196, 3277],
            }
        )

        model = bmb.Model(
            "c(y1, y2, y3, y4) ~ 1 + (1 | state)", data, family="multinomial", noncentered=False
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_posterior_predictive(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestDirichletMultinomial(FitPredictParent):
    def assert_posterior_predictive(self, model, idata):
        y_name = model.response_component.term.name
        y_posterior_predictive = idata.posterior_predictive[y_name].to_numpy()
        assert (y_posterior_predictive.sum(-1).var((0, 1)) == 0).all()

    def test_intercept_only(self, data_multinomial):
        model = bmb.Model("c(y1, y2, y3, y4) ~ 1", data_multinomial, family="dirichlet_multinomial")
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        self.assert_posterior_predictive(model, idata)

        # Out of sample with different number of rows, see issue #845
        idata = self.predict_oos(model, idata, data=model.data.sample(frac=0.8, random_state=1211))
        self.assert_posterior_predictive(model, idata)

    def test_predictor(self, data_multinomial):
        model = bmb.Model(
            "c(y1, y2, y3, y4) ~ 0 + treat", data_multinomial, family="dirichlet_multinomial"
        )
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        self.assert_posterior_predictive(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBetaBinomial(FitPredictParent):
    def test_basic(self, data_beetle):
        model = bmb.Model("prop(y, n) ~ x", data_beetle, family="beta_binomial")
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        n = data_beetle["n"].to_numpy()
        assert np.all(
            idata.posterior_predictive["prop(y, n)"].values <= n[np.newaxis, np.newaxis, :]
        )


def test_wald_family(data_n100, mock_pymc_sample):
    data_n100["y"] = np.exp(data_n100["y1"])
    priors = {"common": bmb.Prior("Normal", mu=0, sigma=1)}
    model = bmb.Model("y ~ y2", data_n100, family="wald", link="log", priors=priors)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)

    model.predict(idata, kind="response_params")
    model.predict(idata, kind="response")

    assert (0 < idata.posterior["mu"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()

    model.predict(idata, kind="response_params", data=data_n100.iloc[:20, :])
    model.predict(idata, kind="response", data=data_n100.iloc[:20, :])

    assert (0 < idata.posterior["mu"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()


def test_predict_include_group_specific(data_random_n100):
    model = bmb.Model("binary_num ~ 1 + (1|continuous1)", data_random_n100, family="bernoulli")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    idata_1 = model.predict(
        idata, data=data_random_n100, inplace=False, include_group_specific=True
    )
    idata_2 = model.predict(
        idata, data=data_random_n100, inplace=False, include_group_specific=False
    )

    assert not np.isclose(idata_1.posterior["p"].values, idata_2.posterior["p"].values).all()

    # Since it's an intercept-only model, predictions are the same for all observations if
    # we drop group-specific terms.
    assert (idata_2.posterior["p"] == idata_2.posterior["p"][:, :, 0]).all()

    # When we include group-specific terms, these predictions are different
    assert not (idata_1.posterior["p"] == idata_1.posterior["p"][:, :, 0]).all()


def test_predict_offset(mock_pymc_sample):
    # Simple case
    data = bmb.load_data("carclaims")
    model = bmb.Model("numclaims ~ offset(np.log(exposure))", data, family="poisson", link="log")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata)
    model.predict(idata, kind="response")

    # More complex case
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.poisson(20, size=100),
            "x": rng.normal(size=100),
            "group": np.tile(np.arange(10), 10),
        }
    )
    data["time"] = data["y"] - rng.normal(loc=1, size=100)
    model = bmb.Model("y ~ offset(np.log(time)) + x + (1 | group)", data, family="poisson")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata)
    model.predict(idata, kind="response")


def test_predict_new_groups_fail(data_sleepstudy, mock_pymc_sample):
    model = bmb.Model("Reaction ~ 1 + Days + (1 + Days | Subject)", data_sleepstudy)
    idata = model.fit(chains=2)
    model.build()
    assert_ip_dlogp(model)

    df_new = data_sleepstudy.head(10).reset_index(drop=True)
    df_new["Subject"] = "xxx"
    to_match = "There are new groups for the factors ('Subject',) and 'sample_new_groups' is False."
    with pytest.raises(ValueError, match=re.escape(to_match)):
        model.predict(idata, data=df_new)


@pytest.mark.parametrize(
    "data,formula,family,df_new",
    [
        (
            "data_sleepstudy",
            "Reaction ~ 1 + Days + (1 + Days | Subject)",
            "gaussian",
            pd.DataFrame({"Days": [1, 2, 3], "Subject": ["x", "y", "z"]}),
        ),
        (
            "data_inhaler",
            "rating ~ 1 + period + treat + (1 + treat|subject)",
            "categorical",
            pd.DataFrame(
                {
                    "subject": [1, 999],
                    "rating": [1, 1],
                    "treat": [0.5, 0.5],
                    "period": [0.5, 0.5],
                    "carry": [0, 0],
                }
            ),
        ),
        (
            "data_crossed",
            "Y ~ 0 + threecats + (0 + threecats | subj)",
            "gaussian",
            pd.DataFrame({"threecats": ["a", "a"], "subj": ["0", "11"]}),
        ),
    ],
)
def test_predict_new_groups(data, formula, family, df_new, request, mock_pymc_sample):
    data = request.getfixturevalue(data)
    model = bmb.Model(formula, data, family=family)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, data=df_new, sample_new_groups=True)


def test_weighted(mock_pymc_sample):
    weights = 1 + np.random.poisson(lam=3, size=100)
    y = np.random.exponential(scale=3, size=100)
    data = pd.DataFrame({"w": weights, "y": y})
    model = bmb.Model("weighted(y, w) ~ 1", data, family="exponential")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, kind="response")
    model.predict(idata, kind="response", data=data)
